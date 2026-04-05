import numpy as np
import torch
import time
from typing import Any, Dict
from queue import Queue
import threading
import cv2

from lerobot.policies.go1.evaluate.deploy_single import *

# 导入机器人模型
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    moving_dual_piper,
)
# 导入相机相关模块
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.utils.robot_utils import busy_wait
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
# 导入策略模型
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser

# 导入piper SDK
from piper_sdk import *
import torch
from PIL import Image

import faulthandler
faulthandler.enable(open("crash.log", "w"))

import cv2
import matplotlib.pyplot as plt
import os
import traceback


# 创建两个文件夹
top_image = "/home/hpc/VLA/lerobot/src/lerobot/obs_image_output/top"
wrist_image = "/home/hpc/VLA/lerobot/src/lerobot/obs_image_output/wrist"
os.makedirs(top_image, exist_ok=True)
os.makedirs(wrist_image, exist_ok=True)


# 初始化计数器
counter = 0

def save_images(obs, path1,prefix="frame",):
    global counter
    
    # 生成带序列号的文件名
    filename = f"{prefix}_{counter:06d}.png"  # 6位数字，如 frame_000001.png
    
    # 保存到两个文件夹
    path1 = os.path.join(top_image, filename)
    
    # 转换为PIL图片
    pil_image = to_pil_image(obs)
    
    # 保存到两个文件夹
    pil_image.save(path1)
    
    print(f"已保存: {path1}")
    counter += 1
    return pil_image


# 客户端
class GO1Server:
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
    ) -> Path:
        self.model = GO1Infer(
            model_path=model_path,
            data_stats_path=data_stats_path,
        )

    def run(self,payload: Dict[str, Any] ) -> List:
        actions = self.model.inference(payload)
        return actions.tolist() if hasattr(actions, "tolist") else actions

@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置"""
    robot: RobotConfig
    # 是否使用策略控制机器人
    # policy: PreTrainedConfig | None = None
    # 模型检查点路径
    # ckpt_path: str = None
    # 任务名称
    task: str = None

    def __post_init__(self):
        """初始化后处理，从命令行参数获取预训练模型路径"""
        # 从命令行参数获取策略路径
        policy_path = parser.get_path_arg("policy")
        # if policy_path:
        #     cli_overrides = parser.get_cli_overrides("policy")
        #     # 从预训练路径加载配置
        #     self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        #     self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
        return ["policy"]

class ActionQueue:
    """
    用于存储动作序列的类。
    - 初始化时传入一个 [N, 7] 的 numpy 数组
    - 内部使用 Queue 存储每个单独的动作
    - 提供 get_next_action() 方法，每次取出一个 [7] 动作
    """

    def __init__(self):
        # 创建队列
        self.queue = Queue()
        self.lock = threading.Lock()

    def get_next_action(self) -> np.ndarray:
        """
        从队列中取出下一个动作
        返回:
            np.ndarray: 形状为 [7] 的动作
        """
        if self.queue.empty():
            print("⚠️ 动作队列已空，返回 None")
            return None
        with self.lock:
            action = self.queue.get()
            return action

    def remaining(self) -> int:
        """返回队列中剩余的动作数量"""
        with self.lock:
            return self.queue.qsize()

    def add_actions(self, new_actions: np.ndarray):
        with self.lock:

            # 先清空队列（安全方式）
            # while not self.queue.empty():
            #     try:
            #         self.queue.get_nowait()  # 非阻塞取出元素
            #     except Exception:
            #         break

            # 确保输入形状正确
            assert new_actions.ndim == 2 and new_actions.shape[1] == 7, \
                f"动作维度应为 [N, 7]，但实际为 {new_actions.shape}"

            # 从第10个开始（索引10起）
            # new_actions = new_actions[10:20]  # 丢弃前10个动作
            # for i in range(new_actions.shape[0]):
            #     self.queue.put(new_actions[i])

            # 只取前 20 个
            # 逐个加入队列
            for i in range(min(20, new_actions.shape[0])):
                self.queue.put(new_actions[i])

            # 打印当前状态
            print(f"✅ 已清空旧动作，新增 {new_actions.shape[0]} 个动作，现在共 {self.queue.qsize()} 个。")


def to_pil_image(obs):
    """
    将 (H, W, 3) 的 float32 或 float64 图像安全转换为 PIL.Image 对象。
    自动归一化并保证通道顺序正确。
    """
    # 确保是 numpy 数组
    obs = np.array(obs)

    # 如果是 float 类型，则将值规范到 [0, 255]
    if np.issubdtype(obs.dtype, np.floating):
        # 若范围在 [0,1]，则乘255
        if obs.max() <= 1.0:
            obs = obs * 255.0
        obs = np.clip(obs, 0, 255).astype(np.uint8)

    # 如果通道顺序是 (3, H, W)，则转置
    if obs.ndim == 3 and obs.shape[0] == 3 and obs.shape[-1] != 3:
        obs = np.transpose(obs, (1, 2, 0))  # -> (H, W, 3)

    # 确保是 RGB 三通道
    if obs.ndim == 3 and obs.shape[2] == 3:
        return Image.fromarray(obs, mode="RGB")
    elif obs.ndim == 2:
        return Image.fromarray(obs, mode="L")
    else:
        raise ValueError(f"不支持的图像形状: {obs.shape}")

class ARMRunner:
    def __init__(self,cfg:InferenceConfig):
        self.cfg = cfg
        self.fps = 30
        self.inference_time_s = 10000
        self.factor = 57324.840764   # 弧度→控制信号缩放
        self.speed = 30

        self.gripper_val_mutiple = 1
        # 检查任务名称是否提供
        if cfg.task == None:
            raise ValueError("You need to provide a task name.")
        else:
            self.task = cfg.task

        self._enable_flag = False

        # 相机图像发布器字典
        self._init_robot()

        self.model_path: Union[str, Path] = '/home/hpc/VLA/models/GO-1/go1train_0105_merge1_4/checkpoint-160000'
        self.data_stats_path: Union[str, Path] = '/home/hpc/VLA/models/GO-1/go1train_0105_merge1_4/checkpoint-160000/dataset_stats.json'  #获取训练集的数据集统计，平均值，标准差等

        self.server = GO1Server(self.model_path, self.data_stats_path)

        self.action_queue = ActionQueue()

        # 线程相关变量
        self._control_running = False
        self._control_thread = None

    def _init_robot(self):
        print("Initializing ARM...")
        self.robot = make_robot_from_config(self.cfg.robot)
        # 机器人连接
        
        if not self.robot.is_connected:
            self.robot.connect()
        if self.cfg.robot.type == "single_piper":
            # 机器人使能
            self._enable_fun(self.robot.piper)
            # 设置初始位置
            joints = [0, 0, 0, 0, 0, 0, 0]
            self.joint_ctrl(self.robot.piper,joints,30)
            self._enable_flag = True            
        time.sleep(2.0)
        print("Robot enabled successfully.")

    def _enable_fun(self,piper: C_PiperInterface):
        '''
        使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        
        # 循环检查使能状态
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            # 检查所有电机的使能状态
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("使能状态:", enable_flag)
            # 使能机械臂
            piper.EnableArm(7)
            # 控制夹爪
            piper.GripperCtrl(0, 1000, 0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            
        # 如果超时则退出程序
        if(elapsed_time_flag):
            print("程序自动使能超时,退出程序")
            exit(0)


    def joint_ctrl(self,piper ,joint_data,speed):
        """机械臂关节角回调函数

        Args:
            joint_data (): 
        """
        # 以前是一个service变量
        # self.block_ctrl_flag
        if joint_data is not None:
            factor = 57324.840764 #1000*180/3.14
            factor = 1000 * 180 / np.pi
            joint_0 = round(joint_data[0]*factor)
            joint_1 = round(joint_data[1]*factor)
            joint_2 = round(joint_data[2]*factor)
            joint_3 = round(joint_data[3]*factor)
            joint_4 = round(joint_data[4]*factor)
            joint_5 = round(joint_data[5]*factor)
            if(len(joint_data) >= 7):
                joint_6 = round(joint_data[6]*1000*1000)
                joint_6 = joint_6 * self.gripper_val_mutiple
                if(joint_6>80000): joint_6 = 80000
                if(joint_6<0): joint_6 = 0
            else: joint_6 = None
            if(self._enable_flag):
                # 设定电机速度
                piper.MotionCtrl_2(0x01, 0x01,speed,0x00)
                # 给定关节角位置
                piper.JointCtrl(joint_0, joint_1, joint_2, 
                                        joint_3, joint_4, joint_5)
                # 如果末端夹爪存在，则发送末端夹爪控制
                if(joint_6 is not None):
                    if abs(joint_6)<200:
                        joint_6=0
                    # 默认1N
                    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    def run(self):
        print("进入推理循环...")

        self.start_control_thread()
        try:
            # 主控制循环
            for _ in range(self.inference_time_s * self.fps):
                
                start_loop_t = time.perf_counter()  # 记录循环开始时间
                
                # 获取当前观测数据（状态和图像）
                observation = self.robot.get_observation()
                observation_frame = {}
                state_values = []
                state_values_epos = []
                # 处理观测数据
                for key, value in observation.items():
                    # 把所有带.pos的位置信息合并成状态向量
                    if key.endswith('.pos'):
                        state_values.append(np.float32(value))
                    if key.endswith('.epos'):
                        state_values_epos.append(np.float32(value))

                    # 处理图像数据（HWC格式的ndarray）
                    elif isinstance(value, np.ndarray) and value.ndim == 3:
                        observation_frame[f'observation.images.{key}'] = value

                # 添加合并后的状态向量
                observation_frame['observation.state'] = np.array(state_values, dtype=np.float32) #这里是获取关节
                # observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)  #这里是获取末端位姿

                # 数据预处理
                for name in observation_frame:
                    # print(name)
                    if "image" in name:
                        # 图像预处理：归一化并转换通道顺序
                        obs = observation_frame[name].astype(np.uint8) 
                        # print(f"键名: {name}, 形状: {obs.shape}, 类型: {type(obs)}")

                        # 转换为 PIL.Image
                        obs = to_pil_image(obs)
                        

                    else:
                        obs = observation_frame[name]  # 状态数据直接使用
                    
                    observation_frame[name] = obs

                # 添加任务名称
                if self.task == None:
                    raise ValueError("You need to provide a task name.")
                observation_frame["task"] = self.task

                payload = {
                    "top": observation_frame["observation.images.top"],
                    "right": observation_frame["observation.images.wrist"],
                    "left": observation_frame["observation.images.wrist"],
                    "instruction": observation_frame["task"],
                    "state": observation_frame["observation.state"],
                    "ctrl_freqs": np.array([30]),
                }
                print("state状态",observation_frame["observation.state"])

                # save_images(observation_frame["observation.images.top"],top_image)
                # save_images(observation_frame["observation.images.wrist"],wrist_image)
                
                actions = np.array(self.server.run(payload))[:,:7]

                self.action_queue.add_actions(actions)
                time.sleep(2)
                # for i in range(actions.shape[0]):
                #     action = actions[]
                #     self.queue.put(actions[i]) 

                # 打印耗时（毫秒）
                print(f"生成30个action所需的时间: {(time.perf_counter() - start_loop_t)*1000:.3f} ms")


        except Exception as e:
            print(f"Error in main loop: {e}")
            
        finally:
            # C. 无论因为报错还是正常退出，这里都会执行
            self.stop_control_thread()

    # --- 1. 把之前的 start 逻辑放这里 ---
    def start_control_thread(self):
        if self._control_running:
            return
        
        print("开始控制线程...")
        self._control_running = True
        
        # 根据配置选择控制模式
        target_func = self.joint_ctrl_loop
        
        self._control_thread = threading.Thread(target=target_func)
        self._control_thread.daemon = True
        self._control_thread.start()

    def stop_control_thread(self):
        print("终止控制线程...")
        self._control_running = False  # 通知子线程退出 while 循环
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        print("控制线程停止.")

    def joint_ctrl_loop(self):
        rate = 1.0 / 20.0  # 30Hz
        print("关节控制开始.")
        while self._control_running:    # 使用标志位代替 while True

            t0 = time.perf_counter()    # 使用更高精度的计时器
            try:
                action = self.action_queue.get_next_action()
                if action is not None:
                    # print("joint 控制 action",action)
                    pass
                    # 调试阶段，暂不控制
                    self.joint_ctrl(self.robot.piper, action,self.speed)
                else:
                    pass
            except Exception as e:
                # 捕获异常防止线程静默退出
                print(f"[Error] inside pos_ctrl_loop: {e}")
                traceback.print_exc()
                # 可以选择在这里 break 停止机器人，或者 continue 重试

            # 频率控制
            dt = time.perf_counter() - t0
            if rate - dt > 0:
                time.sleep(rate - dt)
                print("sleep了多少秒",rate-dt)

@parser.wrap()
def inference(cfg: InferenceConfig):
    """主推理函数"""
    runner = ARMRunner(cfg)
    # print(1)
    runner.run()

if __name__ == "__main__":
    inference()

"""
分析：现在推理时间是320ms左右,咱们设定的频率是30HZ执行一次控制，也就是29.8ms,也就是推理一次的时间，大概执行了10次控制

"""
