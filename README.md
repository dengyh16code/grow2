# Improvised Tool Use

Code for the CoRL 2026 paper
[**GROW²: Grounding Which and Where for Robot Tool Use**](https://www.yuhongdeng.com/grow2.html).

A method for grounding object affordances to enable **improvised robotic tool use**.
Given an instruction such as
*"pound the nail"* with no hammer available, the system selects a substitute object,
identifies its 3D affordance region, and plans a Franka arm motion in
[SAPIEN](https://sapien.ucsd.edu/).

Supported tasks: `lift`, `pound`, `sweep`, `cut`, `pour`.

## Installation

On **Linux x86_64**, create the Conda environment (Python 3.11 and the dependencies
in [`requirements.txt`](requirements.txt)):

```bash
git clone https://github.com/dengyh16code/improvised_tooluse.git
cd improvised_tooluse
conda env create -f environment.yml
conda activate improvised_tooluse
```

The pinned MPlib wheel targets Python 3.11 (`cp311`). For another Python version,
choose a matching wheel from the [nature21/MPlib releases](https://github.com/nature21/MPlib/releases/tag/publish),
replace its URL in `requirements.txt`, and update the Python version in `environment.yml`.
Check that the remaining dependencies support your chosen version.

Run project commands from the repository root. Check the installation:

```bash
python -m pip check
python -c "import sapien, mplib, open3d; print(sapien.__version__)"
```

Before running a case, set up the [configuration](#configuration),
[API keys](#api-keys) and [remote services](#remote-services).

## Configuration

[`config.yaml`](config.yaml) contains server endpoints, model choices, paths and
rendering settings. Use `--config PATH` to load another configuration.

| Setting | Purpose / options |
|---|---|
| `models.vlm` | object parsing and tool selection |
| `models.bounding_box` | 2D grounding |
| `registration_way` | `icp` (local, current default) or `foundation_pose` (optional remote service) |

FoundationPose is optional. With `registration_way: icp`, pose registration runs
locally, so you can skip installing and starting FoundationPose. Set
`registration_way: foundation_pose` to use its remote service instead.

### API keys

Export the credentials for your selected models:

```bash
export OPENAI_API_KEY=...            # gpt
export DASHSCOPE_API_KEY=...         # qwen3.7-plus
export DASHSCOPE_WORKSPACE_ID=...    # Qwen workspace ID
```

## Remote services

The [`server/`](server/) folder provides HTTP wrappers for these models:

| Model repository | Purpose | Server script | Port used below |
|---|---|---|---|
| [SAM3](https://github.com/facebookresearch/sam3) | segmentation | [`sam3_server.py`](server/sam3_server.py) | 4054 |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | 3D reconstruction | [`sam3D_server.py`](server/sam3D_server.py) | 4056 |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) (optional) | pose registration and tracking | [`foundation_pose_server.py`](server/foundation_pose_server.py) | 4052 |
| [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) | grasp sampling | [`grasping_server.py`](server/grasping_server.py) | 4058 |

### Install and launch

Clone the required model repositories on your GPU server and follow their
installation and checkpoint instructions. Give each model its own environment or
container, with Flask installed (`python -m pip install flask`).

Copy the SAM3 and SAM3D scripts into their model repository roots. Only install
FoundationPose and copy its script if using `registration_way: foundation_pose`.
Place `grasping_server.py` in the inner `contact_graspnet/` directory, alongside
`config_utils.py` and `contact_grasp_estimator.py`. Run the commands below from each
model's repository root.
Use a separate terminal or tmux session for each server, waiting for startup to
finish before launching the next.

**SAM3** — from `sam3/`:

```bash
cp /path/to/improvised_tooluse/server/sam3_server.py .
export HF_HOME="$HOME/.cache/huggingface"
python sam3_server.py --ip 0.0.0.0 --port 4054
```

**SAM3D** — from `sam-3d-objects/`, with the downloaded model configuration at
`checkpoints/hf/pipeline.yaml`:

```bash
cp /path/to/improvised_tooluse/server/sam3D_server.py .
python sam3D_server.py --ip 0.0.0.0 --port 4056
```

**FoundationPose (optional)** — only for `registration_way: foundation_pose`,
from `FoundationPose/`:

```bash
cp /path/to/improvised_tooluse/server/foundation_pose_server.py .
python foundation_pose_server.py --ip 0.0.0.0 --port 4052
```

**Contact-GraspNet** — from the outer `contact_graspnet/` repository directory:

```bash
cp /path/to/improvised_tooluse/server/grasping_server.py contact_graspnet/
# Set self.ckpt_dir in the copied script to your downloaded checkpoint directory.
python contact_graspnet/grasping_server.py --ip 0.0.0.0 --port 4058
```

### Update config.yaml

Replace the example hosts below with your server IPs or hostnames in
[`config.yaml`](config.yaml). Ports must match the launch commands:

```yaml
servers:
  sam3:
    host: 192.168.1.10
    port: 4054
  sam3d:
    host: 192.168.1.11
    port: 4056
  foundationpose:  # Only used with registration_way: foundation_pose.
    host: 192.168.1.12
    port: 4052
  grasp:
    host: 192.168.1.13
    port: 4058
```

Use a reachable host without `http://`; `0.0.0.0` is only a server bind address.
For containers or port forwarding, use the port exposed to the client.

## Usage

```bash
python main.py --case_id 0
```

Each `case_<id>.json` in [`test_case/`](test_case/) defines an instruction, task and scene.

| Flag | Effect |
|---|---|
| `--case_id N` | select a case |
| `--config PATH` | use another configuration |
| `--load_cache` | reuse perception results from `<task_type>_result/` |
| `--preview_scene` | view the scene without running the pipeline |
| `--perception_only` | skip planning |
| `--render` | watch execution in the viewer |
| `--record` | save execution to `trial_<case_id>.mp4` |
| `--verbose` | visualize perception intermediates |

Runs are headless by default but still require GPU rendering through Vulkan.
`--record` saves successful execution; combine it with `--render` to watch as well.

## Real-robot execution

For a physical Franka and RealSense camera, install the additional hardware dependencies
(`pyrealsense2`, plus `pybullet` for `convex_obj.py`) and run:

```bash
python main_real_world.py --instruction "pour into the bowl" --task_type pour
```

Add `--load_cache` to reuse `result/` for the same scene and instruction.

## Repository layout

```text
environment.yml         # Conda environment
requirements.txt        # pinned Python dependencies
config.yaml             # runtime configuration
main.py                 # simulation entry point
main_real_world.py      # real-robot entry point
server/                 # model server scripts
improvised_tooluse/
├── affordance.py       # affordance detection
├── simu_env.py         # SAPIEN scene setup
├── planner.py          # motion planning
├── vlm_client.py       # VLM queries
├── vfm_client.py       # vision-model server clients
├── geom_utils.py       # geometry helpers
├── config.py           # configuration loading
├── prompt/             # VLM prompts
└── tasks/              # task implementations
```
