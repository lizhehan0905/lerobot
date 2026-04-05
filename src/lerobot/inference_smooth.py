import numpy as np
import torch
import time
from collections import deque
import requests
import threading
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

# 导入机器人模型
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    # moving_dual_piper,
)
# 导入相机相关模块
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.utils.robot_utils import busy_wait
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
# 导入策略模型
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
# from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
# 导入piper SDK
from piper_sdk import *
from scipy.spatial.transform import Rotation as R


@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置"""
    robot: RobotConfig
    # 是否使用策略控制机器人
    policy: PreTrainedConfig | None = None
    # 模型检查点路径
    ckpt_path: str = None
    # 任务名称
    task: str = None

    def __post_init__(self):
        """初始化后处理，从命令行参数获取预训练模型路径"""
        # 从命令行参数获取策略路径
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            # 从预训练路径加载配置
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
        return ["policy"]


def enable_fun(piper: C_PiperInterface):
    '''
    使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序
    '''
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        
    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)


# 欧拉角 转 rotate6D 输入:q 角度
def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=True).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


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
    b3 = np.cross(b1, b2, axis=-1)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz',degrees=True)


def process_position(position):
    position = np.asarray(position)

    if position.shape[-1] != 10:
        raise ValueError(f"Last dimension must be 10 (got {position.shape[-1]})")

    first_three = position[..., :3]
    rotated_three = rotate6d_to_xyz(position[..., 3:9])
    last_one = position[..., 9:10]

    new_position = np.concatenate([first_three, rotated_three, last_one], axis=-1)
    return new_position


class ArmSmoother:
    def __init__(self, smooth_window=11, angle_indices=[3,4,5]):
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.pad = self.smooth_window // 2
        self.window = np.ones(self.smooth_window) / self.smooth_window
        self.angle_indices = angle_indices
        self.max_angle = 180000
        self.min_angle = -self.max_angle
        self.full_circle = self.max_angle - self.min_angle
        self.angle_clamp_range = (-180, 180)

    def _unwrap_angle(self, values):
        """角度解缠绕"""
        unwrapped = np.copy(values).astype(np.float64)
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i - 1]
            if diff > self.max_angle:
                unwrapped[i:] -= self.full_circle
            elif diff < -self.max_angle:
                unwrapped[i:] += self.full_circle
        return unwrapped

    def _wrap_angle(self, values):
        """角度重缠绕"""
        wrapped = np.copy(values)
        wrapped = wrapped % self.full_circle
        wrapped[wrapped >= self.max_angle] -= self.full_circle
        return wrapped.astype(np.float64)

    def _angle_weighted_average(self, angle1, angle2, weight1, weight2):
        """计算两个角度的加权平均, 兼容-180~180°环形"""
        scale = self.max_angle / 180
        norm_angle1 = angle1 / scale
        norm_angle2 = angle2 / scale

        total_weight = weight1 + weight2
        w1 = weight1 / total_weight
        w2 = weight2 / total_weight

        rad1 = np.radians(norm_angle1)
        rad2 = np.radians(norm_angle2)
        avg_cos = w1 * np.cos(rad1) + w2 * np.cos(rad2)
        avg_sin = w1 * np.sin(rad1) + w2 * np.sin(rad2)

        avg_rad = np.arctan2(avg_sin, avg_cos)
        avg_angle_norm = np.degrees(avg_rad)

        clamp_min, clamp_max = self.angle_clamp_range
        if avg_angle_norm > clamp_max:
            avg_angle_norm -= 360
        elif avg_angle_norm < clamp_min:
            avg_angle_norm += 360
        return avg_angle_norm * scale

    def smooth_column(self, col_data, need_angle_wrap):
        """
        单列平滑
        :param need_angle_wrap: 是否需要角度解缠绕/重缠绕
        """
        if len(col_data) < self.smooth_window:
            return col_data.astype(np.float64)

        col_processed = self._unwrap_angle(col_data) if need_angle_wrap else col_data.astype(np.float64)
        col_padded = np.pad(col_processed, pad_width=(self.pad, self.pad), mode='reflect')
        smoothed_col = np.convolve(col_padded, self.window, mode='valid')

        if need_angle_wrap:
            smoothed_col = self._wrap_angle(smoothed_col)
        return smoothed_col

    def smooth(self, data):
        """
        多列数据平滑
        :param data: 2D数组
        """
        if len(data) < self.smooth_window:
            return data.astype(np.float64)

        smoothed_batch = np.copy(data).astype(np.float64)

        for dim in range(data.shape[-1]):
            need_wrap = dim in self.angle_indices
            smoothed_col = self.smooth_column(data[:, dim], need_wrap)
            smoothed_batch[:, dim] = smoothed_col
        return smoothed_batch

    def merge_with_queue(self, smoothed_batch, position_queue):
        """合并新老动作批次"""
        merged_batch = smoothed_batch.copy()
        queue_len = len(position_queue)
        if queue_len == 0:
            return merged_batch

        queue_len = min(queue_len, len(merged_batch))
        queue_actions = np.array(list(position_queue)[:queue_len])

        total_parts = queue_len + 1
        step = 1.0 / total_parts
        queue_weights = np.array([1.0 - step * (i+1) for i in range(queue_len)])
        batch_weights = 1.0 - queue_weights

        for i in range(queue_len):
            wq, wb = queue_weights[i], batch_weights[i]
            for j in range(smoothed_batch.shape[-1]):
                if j in self.angle_indices:
                    merged_batch[i,j] = self._angle_weighted_average(queue_actions[i,j], smoothed_batch[i,j], wq, wb)
                    # print(f"queue {queue_actions[i, j]} smooth {smoothed_batch[i, j]} merged {merged_batch[i, j]}")
                else:
                    merged_batch[i,j] = wq * queue_actions[i,j] + wb * smoothed_batch[i,j]
        return merged_batch

class RobotInferenceSystem:
    """机器人推理系统，封装线程化的采集+推理、动作执行逻辑"""
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.robot = None
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.device = "cuda"
        
        self.running = False                        # 线程控制标志
        self.infer_collect_thread = None            # 采集+推理合并线程
        self.execute_thread = None                  # 动作执行线程

        self.SMOOTH_WINDOW = 9              # 平滑窗口大小
        self.INFERENCE_PERIOD = 1.5         # 采集+推理周期，秒
        self.EXECUTION_PERIOD = 0.05        # 执行周期，秒
                
        self.smoother = ArmSmoother(self.SMOOTH_WINDOW)

        self.pos_factor = 1000000
        self.angel_factor = 1000
        self.grapper_factor = 70000

    def init_robot(self):
        """初始化机器人并完成使能"""
        # 创建机器人实例并连接
        self.robot = make_robot_from_config(self.cfg.robot)
        if not self.robot.is_connected:
            self.robot.connect()

        # 初始化策略模型
        self._init_policy()

        # 机器人使能
        self._enable_robot()

        # 等待使能完成
        time.sleep(2.0)

    def _init_policy(self):
        """初始化策略模型"""
        if self.cfg.task is None:
            raise ValueError("You need to provide a task name.")

        ckpt_path = self.cfg.ckpt_path
        print(f"ckpt_path : {ckpt_path}")

        # 根据策略类型创建模型
        policy_type = self.cfg.policy.type
        if policy_type == "act":
            self.policy = ACTPolicy.from_pretrained(ckpt_path)
        elif policy_type == "diffusion":
            self.policy = DiffusionPolicy.from_pretrained(ckpt_path)
        elif policy_type == "smolvla":
            self.policy = SmolVLAPolicy.from_pretrained(ckpt_path)
        elif policy_type == "groot":
            self.policy = GrootPolicy.from_pretrained(ckpt_path)
        elif policy_type == "xvla":
            self.policy = XVLAPolicy.from_pretrained(ckpt_path)
        elif policy_type == "pi05":
            self.policy = PI05Policy.from_pretrained(ckpt_path)
        else:
            raise ValueError("You need to provide a valid policy between act/diffusion/smolvla/xvla.")

        self.position_queue = deque(maxlen=self.policy.config.n_action_steps)  # 位置队列
        self.position_queue_lock = threading.Lock()

        self.preprocess, self.postprocess = make_pre_post_processors(self.policy.config, ckpt_path)

        self.policy.eval()
        self.policy.to(self.device)
        self.policy.reset()

    def _enable_robot(self):
        """根据机器人类型执行使能操作"""
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            # 单臂机器人
            self.robot.piper.EnableArm(7)
            enable_fun(piper=self.robot.piper)
            self.robot.piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
            self.robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def grapper_attenuation(self, raw_value, full_open, mid=0.4, steepness=4):
        x = np.clip(raw_value, 0, full_open) / full_open
        y = (x / mid) ** steepness * mid if x <= mid else x
        return int(y * full_open)

    def _collect_and_infer(self):
        """采集+推理线程函数"""
        while self.running:
            start_t = time.perf_counter()
            try:
                with self.position_queue_lock:
                    action_queue_len_pre = len(self.position_queue)

                observation = self.robot.get_observation()
                observation_frame = {}

                state_values = []
                for key, value in observation.items():
                    if key.endswith('.pos'):
                        state_values.append(np.float32(value))
                    elif isinstance(value, np.ndarray) and value.ndim == 3:
                        observation_frame[f'observation.images.{key}'] = value
                observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)  # (10,)

                for name in observation_frame:
                    if "image" in name:
                        obs = observation_frame[name].astype(np.float32) / 255.0  # 归一化
                        obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
                    else:
                        obs = observation_frame[name]
                    obs = np.expand_dims(obs, axis=0)
                    observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=self.device)
                observation_frame["task"] = [self.cfg.task]

                observation_frame = self.preprocess(observation_frame)
                with torch.inference_mode():
                    actions = self.policy.predict_action_chunk(observation_frame)[:, : self.policy.config.n_action_steps]
                actions = self.postprocess(actions)
                actions_np = actions.squeeze(0).cpu().detach().numpy()
                position = process_position(actions_np.tolist())  # (50, 7)

                part1 = position[..., :3] * self.pos_factor
                part2 = position[..., 3:6] * self.angel_factor
                part3 = position[..., 6:7] * self.grapper_factor
                scaled_position = np.concatenate([part1, part2, part3], axis=-1)
                smoothed_batch = np.copy(scaled_position)

                # 平滑
                smoothed_batch = self.smoother.smooth(smoothed_batch)
                with self.position_queue_lock:
                    executed_steps = action_queue_len_pre - len(self.position_queue)
                    smoothed_batch = smoothed_batch[executed_steps:]
                    smoothed_batch = self.smoother.merge_with_queue(smoothed_batch, self.position_queue)

                # 放入动作队列
                with self.position_queue_lock:
                    self.position_queue.clear()
                    smoothed_batch = smoothed_batch.astype(np.int64)
                    for pos in smoothed_batch:
                        self.position_queue.append(pos)
                
                dt = time.perf_counter() - start_t
                print(f"inference over. cost time:{dt} s.")
                wait_time = self.INFERENCE_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"采集+推理线程异常: {e}")
                time.sleep(0.001)

    def _execute_action(self):
        """动作执行线程函数（独立线程，保证控制实时性）"""
        last_queue_len = 1000  # 大数
        is_new_batch = False
        while self.running:
            start_t = time.perf_counter()
            position = None
            try:
                # 获取动作数据，阻塞等待，直到有数据
                while self.running:
                    with self.position_queue_lock:
                        if self.position_queue:
                            is_new_batch = False
                            if last_queue_len < len(self.position_queue):
                                is_new_batch = True
                            last_queue_len = len(self.position_queue)
                            position = self.position_queue.popleft()
                            break
                    time.sleep(0.001)
                if not self.running:
                    break

                # 执行动作
                # print(f"queue len: {last_queue_len}")
                print(f"{'new ' if is_new_batch else ''}{position}")
                self._execute_robot_action(position)

                dt = time.perf_counter() - start_t
                wait_time = self.EXECUTION_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"动作执行线程异常: {e}")
                time.sleep(0.001)

    def _execute_robot_action(self, position):
        """根据机器人类型执行具体的动作控制"""
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            self.robot.piper.EndPoseCtrl(*(int(x) for x in position[:6]))
            gripper = self.grapper_attenuation(position[6], self.grapper_factor)
            self.robot.piper.GripperCtrl(gripper, 1000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type for action execution: {robot_type}")
        
    def start(self):
        if self.running:
            return
        self.running = True

        self.init_robot()

        self.infer_collect_thread = threading.Thread(target=self._collect_and_infer, daemon=True)
        self.execute_thread = threading.Thread(target=self._execute_action, daemon=True)
        self.infer_collect_thread.start()
        self.execute_thread.start()

        print("所有线程已启动，开始机器人控制循环...")

    def stop(self):
        """停止所有线程并清理资源"""
        self.running = False

        # 等待线程结束
        if self.infer_collect_thread:
            self.infer_collect_thread.join(timeout=1.0)
        if self.execute_thread:
            self.execute_thread.join(timeout=1.0)
        print("所有线程已停止")


@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """推理入口函数"""
    # 创建推理系统实例
    inference_system = RobotInferenceSystem(cfg)

    try:
        # 启动线程化推理
        inference_system.start()
        
        # 主进程空闲运行
        inference_time_s = 3600  # 推理总时间, 1小时
        start_time = time.time()
        while time.time() - start_time < inference_time_s:
            if not inference_system.running:
                break
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        print("\n接收到停止信号，正在关闭系统...")
    finally:
        # 停止所有线程
        inference_system.stop()
        print("推理系统已正常关闭")


if __name__ == "__main__":
    inference()

'''

alias c='clear'
alias s='ls'
alias e='exit'
alias ll='ls -hl'
alias cs='clear; ls'
alias cdb='cd ..'
alias cdbb='cd ../..'
alias cdbbb='cd ../../..'
alias cdf='cd -'

python lerobot/inference_smooth.py --robot.type=single_piper --robot.id=single_piper_robot --robot.port=can1 --robot.cameras="{wrist: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" --policy.compile_model=false  --robot.ctrl_mode="eef" --policy.type=pi05 --ckpt_path=/home/hpc/VLA/lerobot_models/pi05/2_28_rotate6d_300/24000/pretrained_model  --task="Pick up the express package and put it on the tray."

'''