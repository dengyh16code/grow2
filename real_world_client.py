import time
import zmq

def make_req_socket(robot_ip: str, port: int = 5555):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)

    # Optional timeouts so it won't hang forever if server dies
    sock.setsockopt(zmq.RCVTIMEO, 10000)  # 10s wait for reply
    sock.setsockopt(zmq.SNDTIMEO, 2000)   # 2s wait to send

    sock.connect(f"tcp://{robot_ip}:{port}")
    return sock


def send_and_wait(sock, msg: dict) -> dict:
    """
    REQ/REP rule: send once, then recv once.
    """
    sock.send_json(msg)
    reply = sock.recv_json()
    return reply

def convert_planning_result_to_traj(planning_result):
    times = planning_result['time']
    positions = planning_result['position']
    velocity = planning_result['velocity']
    acceleration = planning_result['acceleration']
    traj_points = []
    for i in range(0, len(times)):
        point = {
            "t": float(times[i]), 
            "pos": [float(p) for p in positions[i]],
            "vel": [float(v) for v in velocity[i]],
            "acc": [float(a) for a in acceleration[i]],
        }
        traj_points.append(point)
    return traj_points

class RobotClient: 
    def __init__(self, robot_ip: str = "172.16.0.50", port: int = 5555):
        self.sock = make_req_socket(robot_ip, port)
        self.joint_names = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]
    
    def execute_trajectory(self, planning_result):
        points = convert_planning_result_to_traj(planning_result)
        msg = {"type": "traj", "joint_names": self.joint_names, "points": points}
        print("[CLIENT] Send trajectory ...")
        reply = send_and_wait(self.sock, msg)
        print("[CLIENT] Reply:", reply)
        status = reply['ok']
        if not status:
            self.go_home()
            exit()
        return status
    
    def close_gripper(self):
        print("[CLIENT] Send gripper grasp ...")
        reply = send_and_wait(self.sock, {"type": "hand", "command": "close"})
        print("[CLIENT] Reply:", reply)
        status = reply['ok']
        return status
    
    def open_gripper(self):
        print("[CLIENT] Send gripper open ...")
        reply = send_and_wait(self.sock, {"type": "hand", "command": "open"})
        print("[CLIENT] Reply:", reply)
        status = reply['ok']
        return status
    
    def go_home(self):
        print("[CLIENT] Send go home ...")
        reply = send_and_wait(self.sock, {"type": "home"})
        print("[CLIENT] Reply:", reply)
        status = reply['ok']
        return status
    
if __name__ == "__main__":
    client = RobotClient()
    while True:
        command = input("Enter command (o, c, h, q): ")
        if command == "o":
            client.open_gripper()
        elif command == "c":
            client.close_gripper()
        elif command == "h":
            client.go_home()
        elif command == "q":
            break
