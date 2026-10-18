from improvised_tooluse.tasks.task import Task, AffordanceRegion, quat_distance_wxyz
from contextlib import nullcontext
import time
import numpy as np
import open3d as o3d
import trimesh
from dataclasses import dataclass
from sapien import Pose, physx
from scipy.spatial import ConvexHull, cKDTree
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2quat
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from real_world_client import RobotClient


def voxel_points(points, resolution=0.002):
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        raise ValueError("Empty perceived surface")
    _, indices = np.unique(np.floor(points / resolution).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(indices)]


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Cannot estimate a direction from degenerate perception")
    return vector / length


@dataclass
class ToolEnd:
    tip: np.ndarray
    axis: np.ndarray
    radius: float
    straight_length: float
    surface: np.ndarray
    distal_path_length: float


def connected_tool_end(surface, action_points, resolution=0.002):
    """Find the distal marked endpoint along a radius-limited surface graph.

    A graph on sampled points tolerates mesh seams without shortcutting across
    a hook's wide mouth. The opposite end is inferred from the full surface,
    without grasp labels. Missing connectivity is an error, not an axis-sign guess.
    """
    points = voxel_points(surface, resolution)
    action = np.asarray(action_points, dtype=float)
    if min(len(points), len(action)) < 4:
        raise ValueError("Insufficient points for connected endpoint estimation")
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(18, len(points)), distance_upper_bound=3 * resolution)
    rows = np.broadcast_to(np.arange(len(points))[:, None], neighbors.shape)
    valid = np.isfinite(distances) & (distances > 0)
    graph = coo_matrix((distances[valid], (rows[valid], neighbors[valid])), shape=(len(points), len(points))).tocsr()
    graph = graph.maximum(graph.T)
    action_indices = tree.query(action)[1]
    # A zero-cost source attached to every action point finds the surface end
    # furthest from the marked region by connected distance, even on a hook.
    source = len(points)
    seeds = np.unique(action_indices)
    edges = graph.tocoo()
    extended = coo_matrix((np.r_[edges.data, np.zeros(2 * len(seeds))],
        (np.r_[edges.row, np.full(len(seeds), source), seeds],
         np.r_[edges.col, seeds, np.full(len(seeds), source)])),
        shape=(source + 1, source + 1)).tocsr()
    from_action = dijkstra(extended, directed=False, indices=source)[:source]
    connected = np.isfinite(from_action)
    if np.count_nonzero(connected) < 0.8 * len(points):
        raise ValueError("Perceived tool surface is disconnected from the lifting region")
    root = int(np.argmax(np.where(connected, from_action, -np.inf)))
    if from_action[root] < 3 * resolution:
        raise ValueError("Lifting region does not identify a distinct tool end")
    path = dijkstra(graph, directed=False, indices=root)
    action_path = path[action_indices]
    finite = np.isfinite(action_path)
    if np.count_nonzero(finite) < 0.8 * len(action):
        raise ValueError("Perceived lifting region spans disconnected tool surfaces")
    distal = action[finite][np.argmax(action_path[finite])]
    # Include the reconstructed cap, even if its last few millimetres lack a label.
    local = np.linalg.norm(points - distal, axis=1) < 0.012
    local &= np.isfinite(path)
    end_path = path[local]
    cap = points[local & (path >= np.quantile(end_path, 0.95))].mean(axis=0)
    tip_seed = int(tree.query(cap)[1])

    def tip_component(mask):
        """Keep only the surface connected to the tip inside this neighborhood.

        Nearby branches can meet farther back along the tool. Connectivity must
        be checked after cropping, so that meeting point cannot join separate
        tines or the two arms of a hook inside the local shaft fit.
        """
        indices = np.flatnonzero(mask)
        seed = np.flatnonzero(indices == tip_seed)
        if not len(seed):
            return indices[:0]
        _, labels = connected_components(graph[indices][:, indices], directed=False)
        return indices[labels == labels[seed[0]]]

    nearby = points[tip_component(np.linalg.norm(points - cap, axis=1) < 0.018)]
    if len(nearby) < 8:
        raise ValueError("Insufficient endpoint surface for a local shaft fit")
    values, vectors = np.linalg.eigh(np.cov(nearby.T))
    if values[-1] < 1.5 * values[-2]:
        raise ValueError("Local tool endpoint has no reliable insertion axis")
    axis = vectors[:, -1]
    if axis @ (cap - nearby.mean(axis=0)) < 0:
        axis = -axis
    # Estimate the centre of the shaft and its terminal axial position locally.
    projection = nearby @ axis
    end = np.quantile(projection, 0.98)
    shaft = nearby[(projection >= end - 0.014) & (projection <= end - 0.004)]
    if len(shaft) < 4:
        shaft = nearby
    centre = shaft.mean(axis=0)
    tip = centre + axis * (end - centre @ axis)
    radial = shaft - centre - np.outer((shaft - centre) @ axis, axis)
    radius = max(resolution, float(np.quantile(np.linalg.norm(radial, axis=1), 0.9)))

    # Stop usable insertion length where the centreline bends or the tool widens.
    delta = points - tip
    axial = -(delta @ axis)
    radial_vectors = delta + np.outer(axial, axis)
    radial_distance = np.linalg.norm(radial_vectors, axis=1)
    straight_length = 0.0
    corridor = radial_distance < max(0.015, 3 * radius)
    for depth in np.arange(0.004, min(0.15, float(np.max(axial))) + 0.001, 0.003):
        # Grow the connected shaft from the tip. Other branches must not count
        # toward its width until their joining surface enters the measured span.
        shaft_indices = tip_component(corridor & (axial <= depth + 0.0025))
        section_indices = shaft_indices[np.abs(axial[shaft_indices] - depth) < 0.0025]
        section = radial_vectors[section_indices]
        if len(section) < 3:
            break
        centre_offset = np.linalg.norm(section.mean(axis=0))
        section_radius = np.quantile(np.linalg.norm(section - section.mean(axis=0), axis=1), 0.9)
        if centre_offset > max(radius, 0.004) or section_radius > 1.8 * radius:
            break
        straight_length = float(depth)
    if straight_length < 0.007:
        raise ValueError("No usable straight segment behind the perceived tool tip")
    return ToolEnd(tip, unit(axis), radius, straight_length, points, float(np.max(action_path[finite])))


def insertion_depths(handle_points, center, direction, end, uncertainty):
    """Require the tip to clear the far face while the bend stays outside the near face."""
    axial = (np.asarray(handle_points) - center) @ direction
    near, far = np.quantile(axial, [0.02, 0.98])
    minimum = float(far + end.radius + uncertainty)
    maximum = float(near + end.straight_length - uncertainty)
    if maximum < minimum:
        return []
    # The midpoint gives tolerance to under/over-insertion; the other two
    # options retain margins at both faces instead of using patch length.
    return sorted(set([minimum, (minimum + maximum) / 2, maximum]))


def predicted_support(surface, target_surface, handle_points, rotation, tip, inserted_tip, radius,
                      *, target_tree=None, handle_tree=None, rotated_points=None):
    """Estimate whether a marked handle underside supports an upward tool motion.

    This is a geometric score on perceived surfaces, not simulated engagement.
    It rejects contacts with the unmarked body before contact with the handle.
    """
    rotated = ((surface - tip) @ rotation.T
               if rotated_points is None else np.asarray(rotated_points))
    tool = rotated + inserted_tip
    target_tree = cKDTree(target_surface) if target_tree is None else target_tree
    handle_tree = cKDTree(handle_points) if handle_tree is None else handle_tree
    # Only upward-facing material can provide an upward normal force.
    upper_handle = np.max(handle_points[:, 2]) - inserted_tip[2]
    maximum = min(0.12, max(0.03, upper_handle + 0.015))
    heights = np.arange(0.003, maximum + 0.003, 0.003)
    # Keep all points that could enter the 2.5 mm contact radius at any height.
    possible = np.all((tool + [0, 0, heights[0]] <= target_tree.maxes + 0.0025)
                      & (tool + [0, 0, heights[-1]] >= target_tree.mins - 0.0025), axis=1)
    tool = tool[possible]
    if len(tool) < 3:
        return None
    for height in heights:
        moved = tool + [0, 0, height]
        distance, indices = target_tree.query(moved, distance_upper_bound=0.0025)
        contacts = distance < 0.0025
        if np.count_nonzero(contacts) < 3:
            continue
        target_contact = target_surface[indices[contacts]]
        tool_contact = moved[contacts]
        on_handle = handle_tree.query(target_contact)[0] < max(0.006, radius)
        if np.count_nonzero(on_handle) < 3:
            return None
        # Target material above the tool gives a possible load-bearing contact.
        upward_gap = target_contact[:, 2] - tool_contact[:, 2]
        good = on_handle & (upward_gap > 0)
        if np.count_nonzero(good) < 3:
            return None
        lateral_extent = np.linalg.norm(np.ptp(tool_contact[good, :2], axis=0))
        return float(height), float(lateral_extent)
    return None


def insertion_frame(direction):
    """A right-handed frame with X along insertion and Y as close to up as possible."""
    x = np.asarray(direction, dtype=float).copy()
    length = np.linalg.norm(x)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Insertion direction must be finite and nonzero")
    x /= length
    up = np.array([0., 0., 1.])
    if abs(x @ up) > 0.95:
        up = np.array([0., 1., 0.])
    y = up - x * (up @ x)
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))

def handle_opening(points, normal, center, surface_points=None):
    """Estimate the opening from perceived points, optionally including observed surface points."""
    basis = insertion_frame(normal)[:, 1:]
    projected = (points - center) @ basis
    hull = ConvexHull(projected)
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    grid = np.stack(np.meshgrid(*[np.linspace(a, b, 61) for a, b in zip(lo, hi)]), axis=-1).reshape(-1, 2)
    equations = hull.equations
    margins = -(grid @ equations[:, :2].T + equations[:, 2])
    grid = grid[np.all(margins > 0, axis=1)]
    if not len(grid):
        raise ValueError("Target affordance does not enclose a handle opening")
    material = projected
    if surface_points is not None:
        axial = (points - center) @ normal
        surface_axial = (surface_points - center) @ normal
        local = surface_points[(surface_axial >= axial.min() - 0.005)
                               & (surface_axial <= axial.max() + 0.005)]
        if len(local):
            material = (local - center) @ basis
    surface_clearance = cKDTree(material).query(grid)[0]
    boundary_clearance = -(grid @ equations[:, :2].T + equations[:, 2]).max(axis=1)
    clearance = np.minimum(surface_clearance, boundary_clearance)
    best = int(np.argmax(clearance))
    return center + basis @ grid[best], float(clearance[best])


def gripper_pose_for_tip(grasp_pose, perceived_tip, desired_tip, rotation):
    """Move a perceived tool point assuming the supplied grasp frame stays rigid."""
    displacement = Pose(desired_tip, mat2quat(rotation)) * Pose(perceived_tip).inv()
    return displacement * grasp_pose


@dataclass
class InsertionCandidate:
    tip: np.ndarray
    rotation: np.ndarray
    center: np.ndarray
    direction: np.ndarray
    depth: float
    support_height: float
    support_width: float
    score: float
    grasp: np.ndarray = None
    depth_margin: float = 0.0


class LiftTask(Task):
    def __init__(self, simu_env, motion_planner, detector, tool, target,
                 verbose=False, is_soft_gripper=False):
        super().__init__(simu_env, motion_planner, detector, tool, target,
                         verbose, is_soft_gripper)
        self.perceived_reconstructions = detector.reconstruction_results_processed
        self.perceived_observations = detector.obs
        self.grasp_scores = np.asarray(detector.grasp_scores).reshape(-1)
        # Simulation material setting only; no object state feeds motion commands.
        # Give each tool shape its own material to avoid changing shared defaults.
        for component in self.tool.components:
            if isinstance(component, physx.PhysxRigidBaseComponent):
                for shape in component.get_collision_shapes():
                    shape.set_physical_material(physx.PhysxMaterial(
                        static_friction=1.5,
                        dynamic_friction=1.5,
                        restitution=shape.get_physical_material().restitution,
                    ))

    def _perceived_surface(self, affordance):
        """Match semantic points to a perception reconstruction, never an actor mesh."""
        points = np.asarray(affordance.points)
        reconstructions = self.perceived_reconstructions
        if not reconstructions:
            raise ValueError("Lift needs full perceived tool and target reconstructions")
        name = min(reconstructions, key=lambda key: np.median(cKDTree(
            np.asarray(reconstructions[key]["mesh_world"].vertices)).query(points)[0]))
        result = reconstructions[name]
        mesh = result["mesh_world"]
        sampled, _ = trimesh.sample.sample_surface(
            mesh, max(20000, int(np.ceil(mesh.area / 0.001**2))), seed=0)
        surface = np.vstack((mesh.vertices, sampled))
        # This residual is an uncertainty heuristic, not a calibrated error bound.
        residual = 0.005
        pointmap = self.perceived_observations.get("front_pointmap")
        mask = result.get("mask")
        if pointmap is not None and mask is not None:
            observed = np.asarray(pointmap)[np.asarray(mask, dtype=bool)]
            observed = observed[np.all(np.isfinite(observed), axis=1)]
            if len(observed):
                residual = float(np.median(cKDTree(surface).query(observed)[0]))
        return surface, residual

    def _geometry_candidates(self):
        tool_surface, _ = self._perceived_surface(self.tool_aff_pcd)
        target_surface, _ = self._perceived_surface(self.target_aff_pcd)
        points, colors = np.asarray(self.tool_aff_pcd.points), np.asarray(self.tool_aff_pcd.colors)
        end = connected_tool_end(tool_surface, points[colors[:, 1] > 0])
        if self.verbose:
            surface_pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(voxel_points(tool_surface, 0.004)))
            surface_pcd.paint_uniform_color([0.65, 0.65, 0.65])
            tip_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            tip_marker.translate(end.tip)
            tip_marker.paint_uniform_color([1.0, 0.0, 0.0])
            o3d.visualization.draw_geometries(
                [surface_pcd, self.tool_aff_pcd, tip_marker,
                 AffordanceRegion.make_vector(end.tip, end.axis, length=0.05)],
                window_name="Tool tip (red) and outward tip direction (blue)")
        region = LiftEDRegion.from_perception(self.target_aff_pcd, self.verbose,
                                              surface_points=target_surface)
        handle = np.asarray(region.pcd.points)
        # Fixed 2 mm margin per side for all lifting cases.
        uncertainty = 0.002
        print(f"Perceived tip {end.tip}, local axis {end.axis}; radius={end.radius:.4f}, "
              f"straight length={end.straight_length:.4f}, uncertainty margin={uncertainty:.4f} m", flush=True)
        print(f"Handle opening {region.center}, radius={region.clearance:.4f} m", flush=True)
        slack = region.clearance - end.radius - uncertainty
        if slack <= 0:
            print("No insertion fits the perceived opening with the uncertainty margin")
            return []
        target_tree = cKDTree(target_surface)
        handle_tree = cKDTree(handle)
        candidates = []
        rejected = dict(depth=0, support=0)
        # Target-side order remains random; the tool endpoint sign is geometric.
        for side in np.random.permutation([-1., 1.]):
            direction = side * region.normal_direction
            frame = insertion_frame(direction)
            offset = min(0.004, slack / 2)
            centers = [region.center + frame[:, 1:] @ np.asarray(x) * offset
                       for x in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]]
            for roll in np.arange(8) * np.pi / 4:
                c, s = np.cos(roll), np.sin(roll)
                rotation = frame @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ insertion_frame(end.axis).T
                rotated = (end.surface - end.tip) @ rotation.T
                for center in centers:
                    depths = insertion_depths(handle, center, direction, end, uncertainty)
                    if not depths:
                        rejected["depth"] += 1
                    for depth in depths:
                        finish = center + direction * depth
                        support = predicted_support(end.surface, target_surface, handle,
                                                    rotation, end.tip, finish, end.radius,
                                                    target_tree=target_tree, handle_tree=handle_tree,
                                                    rotated_points=rotated)
                        if support is None:
                            rejected["support"] += 1
                            continue
                        height, width = support
                        depth_margin = min(depth - min(depths), max(depths) - depth)
                        # A boundary-depth candidate has no axial tolerance.
                        score = depth_margin + min(width, 0.03) * 0.1 - height * 0.025
                        candidates.append(InsertionCandidate(end.tip, rotation, center,
                            direction, depth, height, width, score, depth_margin=depth_margin))
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        print(f"Geometry: {len(candidates)} candidates; rejected {rejected}", flush=True)
        return candidates

    def _planned_grasp(self, raw):
        """Mirror the shared grasp convention using perception and robot kinematics."""
        raw = raw.copy()
        if self.is_soft_gripper:
            offset = np.eye(4)
            offset[:3, 3] = [0, 0, -0.135]
            raw = raw @ offset
        pose = Pose(raw[:3, 3], mat2quat(raw[:3, :3]))
        options = [pose * Pose(q=euler2quat(0, 0, a)) for a in (np.pi / 2, -np.pi / 2)]
        current = self.simu_env.end_effector.pose.q
        # Match the base planner's second-option tie break.
        return options[0] if quat_distance_wxyz(options[0].q, current) < quat_distance_wxyz(options[1].q, current) else options[1]

    def _ik_chain(self, poses, seed):
        """Non-executing feasibility filter in the existing, user-approved planner."""
        backend = self.planner.planner
        q = np.asarray(seed).copy()
        cost = 0.
        for pose, contact in poses:
            with self.planner.ignore_collisions() if contact else nullcontext():
                status, solution = backend.IK(backend._transform_goal_to_wrt_base(pose), q,
                    n_init_qpos=6, return_closest=True, threshold=1e-3)
            if status != "Success":
                return None
            solution = np.asarray(solution).reshape(-1)
            cost += float(np.linalg.norm(solution[:7] - q[:7]))
            q = solution
        return q, cost

    def _lift_distance(self, candidate):
        """Rise from the inserted tip to the perceived handle top, plus 8 cm."""
        points = np.asarray(self.target_aff_pcd.points)
        handle = points[np.asarray(self.target_aff_pcd.colors)[:, 2] > 0]
        handle = handle[np.all(np.isfinite(handle), axis=1)]
        if not len(handle):
            raise ValueError("Cannot calculate lift height without perceived handle points")
        inserted_tip = candidate.center + candidate.direction * candidate.depth
        return max(0.0, float(handle[:, 2].max() - inserted_tip[2])) + 0.08

    def _rank_candidates(self, geometry, limit):
        if not geometry or limit <= 0:
            return []
        from dataclasses import replace
        self.planner.update_from_simulation()
        initial = self.planner.robot.get_qpos().copy()
        seen, grasps, ranked = set(), [], []
        try:
            for index in np.argsort(-self.grasp_scores.reshape(-1)):
                raw = self.grasping_poses[index]
                key = raw.round(6).tobytes()
                if key in seen or raw[:3, 2] @ [0, 0, -1] < self.grasping_threshold:
                    continue
                seen.add(key)
                grasp = self._planned_grasp(raw)
                result = self._ik_chain([(grasp * Pose([0, 0, -0.1]), False),
                    (grasp, True), (grasp * Pose([0, 0, -0.2]), True)], initial)
                if result is not None:
                    grasps.append((raw, grasp, result[0], result[1], float(self.grasp_scores[index])))
            print(f"Grasp IK: {len(grasps)}/{len(seen)} distinct grasps feasible", flush=True)
            if not grasps:
                return []
            grasps.sort(key=lambda g: np.linalg.norm(g[0][:3, 3] - geometry[0].tip)
                        + 0.002 * g[3] - 0.01 * g[4])
            checks = 0
            # Cover different grasp poses and insertion geometries within the
            # pair-check budget. The previous grasp-outer loop spent all 200
            # checks on the first grasp when geometry had >200 candidates.
            for offset in range(len(grasps)):
                for candidate_index, candidate in enumerate(geometry):
                    raw, grasp, seed, grasp_cost, quality = grasps[
                        (candidate_index + offset) % len(grasps)]
                    checks += 1
                    pose = lambda position: gripper_pose_for_tip(grasp, candidate.tip, position, candidate.rotation)
                    start = candidate.center - candidate.direction * 0.08
                    finish = candidate.center + candidate.direction * candidate.depth
                    # The screw planner checks the full insertion path. IK
                    # preflight only needs its start and final poses.
                    chain = [(pose(start), False), (pose(finish), False)]
                    lift_distance = self._lift_distance(candidate)
                    lift_steps = max(1, int(np.ceil(lift_distance / 0.04)))
                    chain += [(Pose([0, 0, z]) * pose(finish), True)
                              for z in np.linspace(0, lift_distance, lift_steps + 1)[1:]]
                    result = self._ik_chain(chain, seed)
                    if result is not None:
                        lever = np.linalg.norm(raw[:3, 3] - candidate.tip)
                        score = candidate.score - 0.002 * lever - 0.0001 * (result[1] + grasp_cost) + 0.0001 * quality
                        ranked.append(replace(candidate, grasp=raw.copy(), score=score))
                    if checks >= 200 or len(ranked) >= limit:
                        break
                if checks >= 200 or len(ranked) >= limit:
                    break
            ranked.sort(key=lambda candidate: candidate.score, reverse=True)
            print(f"Pair IK: {len(ranked)}/{checks} feasible grasp–insertion pairs", flush=True)
            return ranked
        finally:
            self.planner.update_from_simulation()

    def _move_gripper(self, goal, motions, *, screw=True, contact=False):
        """Execute a planned command without simulated object-pose feedback."""
        self.planner.update_from_simulation()
        try:
            status, trajectory = self.planner.move_to_pose(
                goal, use_screw=screw, ignore_collisions=contact,
                qpos_step=0.01, return_traj=True,
            )
        except RuntimeError as error:
            if str(error) != "Fail to parameterize path":
                raise
            print("Planner could not parameterize the path; resampling")
            return False
        if status != 0:
            return False
        self.simu_env.simulate(0.15)
        motions.append((trajectory, 0.15))
        return True

    def execute_task(self, client: RobotClient = None, num_attempts: int = 50):
        if num_attempts <= 0:
            return -1
        candidates = self._rank_candidates(self._geometry_candidates(), num_attempts)
        checkpoint = self.simu_env.pack()
        for trial, candidate in enumerate(candidates):
            self._selected_grasp = candidate.grasp
            print(f"Lift attempt {trial + 1}/{len(candidates)}")
            try:
                _, pre_grasp, grasp, retreat = self.plan_grasp_sequence(num_attempts=1)
            except RuntimeError as error:
                if not (str(error).startswith("Failed to plan a complete grasp sequence")
                        or str(error) == "Fail to parameterize path"):
                    raise
                print(f"{error} Resampling the grasp.")
                self.retrieve(checkpoint)
                continue
            self.simu_env.simulate(0.15)
            tip, rotation = candidate.tip, candidate.rotation
            center, direction, depth = candidate.center, candidate.direction, candidate.depth

            attachment_pose = self.grasp_pose_at_attachment
            if attachment_pose is None:
                raise RuntimeError("Missing robot FK measurement at grasp attachment")
            offset_mm = np.linalg.norm(attachment_pose.p - self.grasp_pose.p) * 1000
            angle_deg = np.degrees(quat_distance_wxyz(attachment_pose.q, self.grasp_pose.q))
            print(f"Attachment frame from robot FK: {offset_mm:.2f} mm, {angle_deg:.2f} deg "
                  "from requested grasp", flush=True)

            def gripper_pose(position):
                # Perceived tool geometry and robot FK at attachment share the
                # same time reference. The simulator tool transform is unused.
                return gripper_pose_for_tip(attachment_pose, tip, position, rotation)

            motions = []
            print(f"Insertion direction {direction}, depth {depth:.3f} m; "
                  f"axial tolerance {candidate.depth_margin:.4f} m, "
                  f"predicted support travel {candidate.support_height:.3f} m")
            preinsert = gripper_pose(center - direction * 0.08)
            if not self._move_gripper(preinsert, motions, screw=False):
                self.retrieve(checkpoint)
                continue

            insert_pose = gripper_pose(center + direction * depth)
            if not self._move_gripper(insert_pose, motions, contact=False):
                self.retrieve(checkpoint)
                continue

            # Without sensed object poses or force feedback, insertion and
            # lifting are open-loop. Do not infer engagement from actor poses.
            lift_distance = self._lift_distance(candidate)
            print(f"Adaptive lift distance: {lift_distance:.4f} m "
                  "(perceived handle top minus inserted tip height, plus 0.08 m)", flush=True)
            if not self._move_gripper(Pose([0, 0, lift_distance]) * insert_pose, motions, contact=True):
                self.retrieve(checkpoint)
                continue

            # Use the simulation evaluation to advance through the already
            # scored candidates when a complete lift does not raise the target.
            if client is None and not self.evaluate():
                self.retrieve(checkpoint)
                continue

            self.retrieve(checkpoint)
            self.simu_env.start_recording()
            self.simu_env.create_viewer()
            self.simu_env.follow_path(pre_grasp)
            self.simu_env.follow_path(grasp)
            self.simu_env.attach_object(self.tool)
            self.simu_env.close_gripper()
            self.simu_env.follow_path(retreat, final_wait=0.15)
            for trajectory, wait in motions:
                self.simu_env.follow_path(trajectory, final_wait=wait)
            if client is None:
                return 0
            if input("Now execute the planned trajectory? y/n: ").lower() != "y":
                self.retrieve(checkpoint)
                continue
            client.execute_trajectory(pre_grasp)
            input("next")
            client.execute_trajectory(grasp)
            input("next")
            client.close_gripper()
            client.execute_trajectory(retreat)
            time.sleep(0.15)
            for trajectory, wait in motions:
                input("next")
                client.execute_trajectory(trajectory)
                time.sleep(wait)
            return 0
        self.retrieve(checkpoint)
        return -1

    def sample_grasping_pose(self):
        if getattr(self, "_selected_grasp", None) is not None:
            self.grasp_mat = self._selected_grasp.copy()
            return self.grasp_mat
        # Do not let the base grasp planner loop forever on non-top-down poses.
        if not getattr(self, "_grasp_order", None):
            eligible = [i for i, pose in enumerate(self.grasping_poses)
                        if pose[:3, 2] @ np.array([0., 0., -1.]) >= self.grasping_threshold]
            if not eligible:
                raise RuntimeError("No top-down grasp candidates available for lifting")
            self._grasp_order = list(np.random.permutation(eligible))
        self.grasp_mat = self.grasping_poses[self._grasp_order.pop()]
        return self.grasp_mat

    def evaluate(self):
        """Benchmark scoring only; this result never triggers a planning retry."""
        lifted = self.target.pose.p[2] - self.target_initial_pose.p[2]
        print(f"Lifted height: {lifted:.4f} m")
        return lifted > 0.015


@dataclass
class LiftEDRegion(AffordanceRegion):
    center: np.ndarray
    pcd: o3d.geometry.PointCloud
    normal_direction: np.ndarray
    clearance: float = 0.0

    @classmethod
    def from_perception(cls, pcd_aff, verbose=False, *, surface_points=None):
        indices = np.flatnonzero(np.asarray(pcd_aff.colors)[:, 2] > 0)
        pcd = pcd_aff.select_by_index(indices.tolist())
        if len(pcd.points) < 4:
            raise ValueError("Not enough target handle affordance points")
        obb = pcd.get_oriented_bounding_box()
        normal = np.array(obb.R[:, np.argmin(obb.extent)])
        center, clearance = handle_opening(np.asarray(pcd.points), normal, np.array(obb.center), surface_points)
        if verbose:
            o3d.visualization.draw_geometries(
                [pcd_aff, cls.make_vector(center, normal)],
                window_name="Target handle normal (both insertion directions tested)")
        return cls(center, pcd, normal, clearance)
