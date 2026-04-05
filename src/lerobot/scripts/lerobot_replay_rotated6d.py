# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

```
lerobot-replay --dataset.repo_id='' --dataset.root /home/hpc/VLA/package_data/single/merge_data_1_4 --dataset.episode=0 --robot.type single_piper --robot.port ' '
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    reachy2,
    so_follower,
    unitree_g1,
    single_piper,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True

# rotate6D转欧拉角
def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2] # 0 2 4
    a2 = v6[..., 1:6:2] # 1 3 5
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz',degrees=True)

def process_position(position):
    """
    position: 长度为10的list或np.ndarray
    返回新的position: 长度10
    """
    position = np.asarray(position)
    if len(position) != 10:
        raise ValueError("position must have length 10")

    # 前3个数保持不动
    first_three = position[:3]

    # 3:9 的6个数 -> rotate6d_to_xyz -> 3个数   单位换算成0.001°
    rotated_three = rotate6d_to_xyz(position[3:9])

    # 第10个数保持不动
    last_one = position[9]

    # 拼回去
    new_position = np.concatenate([first_three, rotated_three, [last_one]])
    return new_position


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    # 创建机械臂对象
    robot = make_robot_from_config(cfg.robot)
    # 创建数据集
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    print(actions)



    # epos = episode_frames.select_columns('observation.state_epos')
    # for i in range(700):
    #     epos = episode_frames.select_columns('observation.state_epos')[i]
    #     print("epos数据", epos)

    # # print("epos数据",epos)
    # exit(0)

    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(len(episode_frames)):
        start_episode_t = time.perf_counter()

        action_array = actions[idx][ACTION]
        # action_array = epos[idx]['observation.state_epos']
        action = {}

        for i, name in enumerate(dataset.features[ACTION]["names"]):
            action[name] = action_array[i]

        robot_obs = robot.get_observation()
        # print("观测动作action",robot_obs)

        processed_action = robot_action_processor((action, robot_obs))

        print("处理后的action",processed_action)

        _ = robot.send_action(processed_action)
        # _ = robot.send_action(action)

        dt_s = time.perf_counter() - start_episode_t
        precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
        time.sleep(0.3)

    robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
