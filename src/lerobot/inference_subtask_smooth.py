# 导入必要的库
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import time  # 时间操作
from collections import deque  # 双端队列，用于存储历史动作
import requests  # HTTP请求库
import threading  # 多线程支持
from PIL import Image  # 图像处理
from transformers import AutoProcessor, AutoModelForImageTextToText  # HuggingFace transformers

# 导入LeRobot相关配置和模块
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

# 导入机器人模型相关模块
from lerobot.robots import (  # noqa: F401
    Robot,  # 机器人基类
    RobotConfig,  # 机器人配置类
    make_robot_from_config,  # 从配置创建机器人实例的工厂函数
    single_piper,  # 单臂Piper机器人
    # moving_dual_piper,  # 双臂移动机器人(暂未使用)
)

# 导入相机相关模块
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # 相机配置类
)
from dataclasses import asdict, dataclass  # 数据类装饰器
from lerobot.utils.robot_utils import busy_wait  # 忙等待函数
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

# 导入策略模型类
from lerobot.policies.act.modeling_act import ACTPolicy  # ACT策略
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # Diffusion策略
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # SmolVLA策略
from lerobot.policies.groot.modeling_groot import GrootPolicy  # Groot策略

# from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # XVLA策略(暂未使用)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # PI05策略
from lerobot.configs.policies import PreTrainedConfig  # 预训练配置
from lerobot.configs import parser  # 命令行参数解析器

# 导入Piper机械臂SDK
from piper_sdk import *
from scipy.spatial.transform import Rotation as R  # 旋转矩阵处理


@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置"""

    robot: RobotConfig  # 机器人配置对象
    policy: PreTrainedConfig | None = None  # 策略配置对象(可选)
    ckpt_path: str = None  # 模型检查点路径
    task: str = None  # 任务名称
    qwen3_vl_path: str = None  # 子任务规划模型路径
    subtask_planning_period: float = 1.0  # 子任务规划周期(秒)

def __post_init__(self):
    """初始化后处理，从命令行参数获取预训练模型路径"""
    # 从命令行参数获取策略路径
    policy_path = parser.get_path_arg("policy")
    if policy_path:
        # 获取命令行覆盖参数
        cli_overrides = parser.get_cli_overrides("policy")
        # 从预训练路径加载策略配置
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path  # 保存预训练模型路径

@classmethod
def __get_path_fields__(cls) -> list[str]:
    """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
    return ["policy"]


def enable_fun(piper: C_PiperInterface):
    """
    使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序

    参数:
        piper: Piper机械臂接口对象

    流程:
        1. 循环尝试使能机械臂
        2. 检查所有6个关节电机的驱动使能状态
        3. 如果5秒内未使能成功则退出程序
    """
    enable_flag = False  # 使能状态标志
    timeout = 5  # 超时时间(秒)
    start_time = time.time()  # 记录开始时间
    elapsed_time_flag = False  # 超时标志

    # 等待机械臂使能成功
    while not (enable_flag):
        elapsed_time = time.time() - start_time  # 计算已用时间
        # 检查6个关节电机是否全部使能成功
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("使能状态:", enable_flag)
        piper.EnableArm(7)  # 发送使能命令(参数7表示使能所有关节)
        if elapsed_time > timeout:  # 超时检查
            print("超时....")
            elapsed_time_flag = True  # 设置超时标志
            enable_flag = True  # 退出循环
            break
        time.sleep(1)  # 每秒检查一次

    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)  # 超时则退出程序


# 欧拉角转rotate6D表示
# rotate6D是一种6维旋转表示法，由两个正交归一化的3维向量组成
# 输入:q - 欧拉角 (弧度或度，取决于pattern参数)
#      pattern - 欧拉角旋转顺序，如"xyz"
# 输出:rotate6D表示 (6,)
def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=True).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


# rotate6D转欧拉角
# 将6维旋转表示转换回3维欧拉角
# 输入:v6 - rotate6D表示 (..., 6)
# 输出:欧拉角 (..., 3)
def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    # 提取两个正交归一化的3维向量
    # a1: [v6[...,0], v6[...,2], v6[...,4]] - 第一个向量的xyz
    # a2: [v6[...,1], v6[...,3], v6[...,5]] - 第二个向量的xyz
    a1 = v6[..., 0:5:2]  # 0 2 4
    a2 = v6[..., 1:6:2]  # 1 3 5
    # 归一化第一个向量
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    # 计算第二个向量在第一个向量方向上的投影
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    # 计算与第一个向量正交的分量并归一化
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    # 计算第三个向量(叉乘得到垂直于前两个向量的向量)
    b3 = np.cross(b1, b2, axis=-1)
    # 组装成3x3旋转矩阵
    rot_mats = np.stack((b1, b2, b3), axis=-1)  # shape (..., 3, 3)
    # 将旋转矩阵转换为欧拉角
    return R.from_matrix(rot_mats).as_euler("xyz", degrees=True)


def process_position(position):
    """处理位置数据，将rotate6D转换为欧拉角

    参数:
        position: 位置数据，shape为(..., 10)
                 - 前3个: 位置(x,y,z)
                 - 中间6个: rotate6D旋转表示
                 - 最后一个: 夹爪位置

    返回:
        处理后的位置数据，shape为(..., 10)
                - 前3个: 位置(x,y,z)
                - 中间3个: 欧拉角(roll,pitch,yaw)
                - 最后一个: 夹爪位置
    """
    position = np.asarray(position)

    if position.shape[-1] != 10:
        raise ValueError(f"Last dimension must be 10 (got {position.shape[-1]})")

    # 提取位置部分
    first_three = position[..., :3]
    # 转换旋转表示(从rotate6D到欧拉角)
    rotated_three = rotate6d_to_xyz(position[..., 3:9])
    # 提取夹爪位置
    last_one = position[..., 9:10]

    # 合并所有部分
    new_position = np.concatenate([first_three, rotated_three, last_one], axis=-1)
    return new_position


class ArmSmoother:
    """机械臂动作平滑器

    功能:
        1. 对动作序列进行滑动窗口平滑
        2. 处理角度的周期性(unwrap/wrap)问题
        3. 合并历史动作队列与新动作序列

    使用场景:
        机器人执行策略输出动作时，由于推理频率较低，
        需要对动作进行平滑处理以获得更流畅的运动轨迹
    """

    def __init__(self, smooth_window=11, angle_indices=[3, 4, 5]):
        """初始化平滑器

        参数:
            smooth_window: 滑动窗口大小，必须为奇数
            angle_indices: 需要特殊角度处理的维度索引(欧拉角索引)
        """
        # 确保窗口大小为奇数
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.pad = self.smooth_window // 2  # 填充大小
        self.window = np.ones(self.smooth_window) / self.smooth_window  # 均匀权重窗口
        self.angle_indices = angle_indices  # 角度维度索引
        self.max_angle = 180000  # 最大角度(0.001度为单位)
        self.min_angle = -self.max_angle  # 最小角度
        self.full_circle = self.max_angle - self.min_angle  # 完整一圈的角度范围
        self.angle_clamp_range = (-180, 180)  # 角度重缠绕范围

    def _unwrap_angle(self, values):
        """角度解缠绕

        处理角度的周期性 discontinuity 问题
        例如: 从179度到-179度， naive处理会是358度的巨大跳跃
        解缠绕后会是连续的: 179, 178, ... , -178, -179

        参数:
            values: 角度数组

        返回:
            解缠绕后的角度数组
        """
        unwrapped = np.copy(values).astype(np.float64)
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i - 1]  # 计算相邻角度差
            if diff > self.max_angle:  # 正向跨越边界(如179到-179)
                unwrapped[i:] -= self.full_circle  # 加上完整一圈
            elif diff < -self.max_angle:  # 负向跨越边界(如-179到179)
                unwrapped[i:] += self.full_circle  # 减去完整一圈
        return unwrapped

    def _wrap_angle(self, values):
        """角度重缠绕

        将解缠绕后的角度重新映射到指定范围
        例如: 将角度限制在 -180 到 180 度之间

        参数:
            values: 角度数组

        返回:
            重缠绕后的角度数组
        """
        wrapped = np.copy(values)
        wrapped = wrapped % self.full_circle  # 模运算
        wrapped[wrapped >= self.max_angle] -= self.full_circle  # 大于等于180度的减去360度
        return wrapped.astype(np.float64)

    def _angle_weighted_average(self, angle1, angle2, weight1, weight2):
        """计算两个角度的加权平均, 兼容-180~180°环形

        由于角度是环形的，直接线性平均会产生错误结果
        例如: -170和170的正确平均应该是180(或-180)，而不是0

        参数:
            angle1, angle2: 要平均的两个角度
            weight1, weight2: 对应的权重

        返回:
            加权平均后的角度
        """
        scale = self.max_angle / 180  # 缩放因子
        norm_angle1 = angle1 / scale  # 归一化到[-1,1]
        norm_angle2 = angle2 / scale

        total_weight = weight1 + weight2
        w1 = weight1 / total_weight  # 归一化权重
        w2 = weight2 / total_weight

        # 转换为弧度
        rad1 = np.radians(norm_angle1)
        rad2 = np.radians(norm_angle2)
        # 使用三角函数进行角度平均
        avg_cos = w1 * np.cos(rad1) + w2 * np.cos(rad2)
        avg_sin = w1 * np.sin(rad1) + w2 * np.sin(rad2)

        avg_rad = np.arctan2(avg_sin, avg_cos)  # 计算平均角度
        avg_angle_norm = np.degrees(avg_rad)  # 转换回角度

        # 钳制到[-180, 180]范围
        clamp_min, clamp_max = self.angle_clamp_range
        if avg_angle_norm > clamp_max:
            avg_angle_norm -= 360
        elif avg_angle_norm < clamp_min:
            avg_angle_norm += 360
        return avg_angle_norm * scale

    def smooth_column(self, col_data, need_angle_wrap):
        """单列数据平滑

        使用滑动窗口对单列数据进行卷积平滑

        参数:
            col_data: 单列数据数组 (n,)
            need_angle_wrap: 是否需要角度解缠绕/重缠绕处理

        返回:
            平滑后的数据数组
        """
        if len(col_data) < self.smooth_window:
            return col_data.astype(np.float64)

        # 如果需要角度处理，先解缠绕
        col_processed = self._unwrap_angle(col_data) if need_angle_wrap else col_data.astype(np.float64)
        # 填充数据以处理边界
        col_padded = np.pad(col_processed, pad_width=(self.pad, self.pad), mode="reflect")
        # 卷积平滑
        smoothed_col = np.convolve(col_padded, self.window, mode="valid")

        if need_angle_wrap:
            # 如果需要角度处理，平滑后重缠绕
            smoothed_col = self._wrap_angle(smoothed_col)
        return smoothed_col

    def smooth(self, data):
        """多列数据平滑

        对2D数据(多行多列)的每一列分别进行平滑处理

        参数:
            data: 2D数组 (n_rows, n_cols)

        返回:
            平滑后的2D数组
        """
        if len(data) < self.smooth_window:
            return data.astype(np.float64)

        smoothed_batch = np.copy(data).astype(np.float64)

        # 对每一列进行平滑处理
        for dim in range(data.shape[-1]):
            need_wrap = dim in self.angle_indices  # 判断该列是否需要角度处理
            smoothed_col = self.smooth_column(data[:, dim], need_wrap)
            smoothed_batch[:, dim] = smoothed_col
        return smoothed_batch

    def merge_with_queue(self, smoothed_batch, position_queue):
        """合并新老动作批次

        将历史动作队列与新平滑后的动作批次进行加权融合
        这样可以使动作过渡更加平滑，避免突变

        参数:
            smoothed_batch: 平滑后的新动作批次 (n, 10)
            position_queue: 历史动作队列(deque)

        返回:
            融合后的动作批次
        """
        merged_batch = smoothed_batch.copy()
        queue_len = len(position_queue)
        if queue_len == 0:
            return merged_batch

        # 取较小长度避免数组越界
        queue_len = min(queue_len, len(merged_batch))
        # 转换为numpy数组
        queue_actions = np.array(list(position_queue)[:queue_len])

        # 计算权重: 队列越靠前的动作权重越大
        total_parts = queue_len + 1
        step = 1.0 / total_parts
        queue_weights = np.array([1.0 - step * (i + 1) for i in range(queue_len)])
        batch_weights = 1.0 - queue_weights  # 新批次权重

        # 加权融合
        for i in range(queue_len):
            wq, wb = queue_weights[i], batch_weights[i]
            for j in range(smoothed_batch.shape[-1]):
                if j in self.angle_indices:
                    # 角度使用角度加权平均
                    merged_batch[i, j] = self._angle_weighted_average(
                        queue_actions[i, j], smoothed_batch[i, j], wq, wb
                    )
                else:
                    # 位置直接线性加权
                    merged_batch[i, j] = wq * queue_actions[i, j] + wb * smoothed_batch[i, j]
        return merged_batch


class RobotInferenceSystem:
    """机器人推理系统，封装线程化的采集+推理、动作执行逻辑

    设计思路:
        1. 双线程架构: 一个线程负责图像采集和策略推理
                      另一个线程负责动作执行(保证实时性)
        2. 队列缓冲: 使用deque作为动作队列，解耦推理和执行
        3. 动作平滑: 对策略输出的动作进行滑动窗口平滑处理

    线程分工:
        - _collect_and_infer: 采集相机图像 -> 策略推理 -> 动作平滑 -> 放入队列
        - _execute_action: 从队列取动作 -> 发送给机器人执行
    """

    def __init__(self, cfg: InferenceConfig):
        """初始化推理系统

        参数:
            cfg: 推理配置对象
        """
        self.cfg = cfg
        self.robot = None  # 机器人对象
        self.policy = None  # 策略模型
        self.preprocess = None  # 预处理函数
        self.postprocess = None  # 后处理函数
        self.device = "cuda"  # 运行设备

        self.running = False  # 线程运行标志
        self.infer_collect_thread = None  # 采集+推理线程
        self.execute_thread = None  # 动作执行线程
        self.subtask_planning_thread = None  # 子任务规划线程

        self.SMOOTH_WINDOW = 9  # 平滑窗口大小
        self.INFERENCE_PERIOD = 1.5  # 采集+推理周期(秒)
        self.EXECUTION_PERIOD = 0.05  # 执行周期(秒)，约20Hz
        self.SUBTASK_PLANNING_PERIOD = cfg.subtask_planning_period  # 子任务规划周期(秒)

        self.smoother = ArmSmoother(self.SMOOTH_WINDOW)  # 动作平滑器

        # 动作数据缩放因子(将归一化动作转换为实际电机指令)
        self.pos_factor = 1000000  # 位置缩放因子
        self.angel_factor = 1000  # 角度缩放因子(0.001度)
        self.grapper_factor = 70000  # 夹爪缩放因子

        # 子任务规划相关
        self.subtask_model = None  # 子任务规划模型
        self.subtask_processor = None  # 子任务规划处理器
        self.current_subtask = None  # 当前子任务
        self.subtask_lock = threading.Lock()  # 子任务线程锁

    def init_robot(self):
        """初始化机器人并完成使能

        流程:
            1. 创建机器人实例并连接
            2. 初始化策略模型
            3. 机器人使能
            4. 等待使能完成
        """
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
        """初始化策略模型

        根据策略类型创建对应的模型实例，并加载预训练权重
        """
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
        # elif policy_type == "xvla":
        #     self.policy = XVLAPolicy.from_pretrained(ckpt_path)
        elif policy_type == "pi05":
            self.policy = PI05Policy.from_pretrained(ckpt_path)
        else:
            raise ValueError("You need to provide a valid policy between act/diffusion/smolvla.")

        # 创建动作队列(用于存储历史动作)
        self.position_queue = deque(maxlen=self.policy.config.n_action_steps)
        self.position_queue_lock = threading.Lock()  # 队列线程锁

        # 创建预处理器和后处理器
        self.preprocess, self.postprocess = make_pre_post_processors(self.policy.config, ckpt_path)

        # 设置模型为推理模式并移到指定设备
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.reset()  # 重置策略状态

        # 初始化子任务规划模型
        self._init_subtask_planning_model()

    def _init_subtask_planning_model(self):
        """初始化子任务规划多模态大模型
        
        加载Qwen3-VL-2B子任务规划模型，用于根据当前图像确定下一个子任务
        """
        qwen3_vl_path = self.cfg.qwen3_vl_path
        if qwen3_vl_path is None:
            print("未配置子任务规划模型路径，跳过加载")
            return
            
        print(f"加载子任务规划模型: {qwen3_vl_path}")
        
        # 加载多模态大模型
        self.subtask_model = AutoModelForImageTextToText.from_pretrained(
            qwen3_vl_path, dtype="auto", device_map="auto"
        )
        self.subtask_processor = AutoProcessor.from_pretrained(qwen3_vl_path)
        
        # 设置为推理模式
        self.subtask_model.eval()
        
        # 初始化当前子任务
        self.current_subtask = self.cfg.task
        print(f"初始子任务: {self.current_subtask}")

    def _enable_robot(self):
        """根据机器人类型执行使能操作

        使能机械臂的各个关节电机，使其可以响应控制命令
        """
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            # 使能机械臂
            self.robot.piper.EnableArm(7)
            # 等待并检测使能状态
            enable_fun(piper=self.robot.piper)
            # 设置运动控制参数
            self.robot.piper.MotionCtrl_2(0x01, 0x00, 20, 0x00)
            # 设置夹爪初始开合程度
            self.robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def grapper_attenuation(self, raw_value, full_open, mid=0.4, steepness=4):
        """夹爪开合度非线性衰减

        将线性输入转换为非线性输出，使夹爪开合更符合实际物理特性

        参数:
            raw_value: 原始输入值
            full_open: 夹爪全开最大值
            mid: 非线性转折点
            steepness: 陡度(控制曲线形状)

        返回:
            衰减后的夹爪控制值
        """
        x = np.clip(raw_value, 0, full_open) / full_open  # 归一化到[0,1]
        # 使用分段函数: x较小时使用幂函数，x较大时使用线性
        y = (x / mid) ** steepness * mid if x <= mid else x
        return int(y * full_open)

    def _collect_and_infer(self):
        """采集+推理线程函数

        循环执行:
            1. 获取机器人观测(相机图像+关节状态)
            2. 策略推理得到动作
            3. 动作平滑处理
            4. 放入动作队列
            5. 控制推理频率
        """
        while self.running:
            start_t = time.perf_counter()
            try:
                # 获取执行前队列长度(用于计算已执行动作数)
                with self.position_queue_lock:
                    action_queue_len_pre = len(self.position_queue)

                # 获取机器人观测数据
                observation = self.robot.get_observation()
                observation_frame = {}

                # 处理观测数据
                state_values = []
                for key, value in observation.items():
                    if key.endswith(".pos"):
                        # 关节位置数据
                        state_values.append(np.float32(value))
                    elif isinstance(value, np.ndarray) and value.ndim == 3:
                        # 相机图像数据
                        observation_frame[f"observation.images.{key}"] = value
                # 状态数据: 位置+夹爪等共10维
                observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)  # (10,)

                # 预处理观测数据(图像归一化、通道转换等)
                for name in observation_frame:
                    if "image" in name:
                        obs = observation_frame[name].astype(np.float32) / 255.0  # 归一化到[0,1]
                        obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW格式
                    else:
                        obs = observation_frame[name]
                    obs = np.expand_dims(obs, axis=0)  # 添加batch维度
                    observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                # 获取当前子任务(线程安全)
                with self.subtask_lock:
                    current_task = self.current_subtask
                observation_frame["task"] = [current_task]  # 添加任务描述

                # 执行预处理器
                observation_frame = self.preprocess(observation_frame)

                # 策略推理
                with torch.inference_mode():
                    # 预测动作 chunk: (batch, n_action_steps, action_dim)
                    actions = self.policy.predict_action_chunk(observation_frame)[
                        :, : self.policy.config.n_action_steps
                    ]

                # 执行后处理器
                actions = self.postprocess(actions)
                actions_np = actions.squeeze(0).cpu().detach().numpy()

                # 处理位置数据(rotate6D -> 欧拉角)
                position = process_position(actions_np.tolist())  # (50, 7)

                # 缩放动作数据到电机控制范围
                part1 = position[..., :3] * self.pos_factor  # 位置: 转换为微米级
                part2 = position[..., 3:6] * self.angel_factor  # 角度: 转换为0.001度
                part3 = position[..., 6:7] * self.grapper_factor  # 夹爪: 转换为控制值
                scaled_position = np.concatenate([part1, part2, part3], axis=-1)
                smoothed_batch = np.copy(scaled_position)

                # 动作平滑处理
                smoothed_batch = self.smoother.smooth(smoothed_batch)

                # 合并历史动作与新动作
                with self.position_queue_lock:
                    executed_steps = action_queue_len_pre - len(self.position_queue)  # 已执行步数
                    smoothed_batch = smoothed_batch[executed_steps:]  # 跳过已执行的
                    smoothed_batch = self.smoother.merge_with_queue(smoothed_batch, self.position_queue)

                # 放入动作队列
                with self.position_queue_lock:
                    self.position_queue.clear()  # 清空旧队列
                    smoothed_batch = smoothed_batch.astype(np.int64)
                    for pos in smoothed_batch:
                        self.position_queue.append(pos)

                # 控制采集+推理频率
                dt = time.perf_counter() - start_t
                print(f"inference over. cost time:{dt} s.")
                wait_time = self.INFERENCE_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"采集+推理线程异常: {e}")
                time.sleep(0.001)

    def _execute_action(self):
        """动作执行线程函数(独立线程，保证控制实时性)

        循环执行:
            1. 从动作队列获取动作(阻塞等待)
            2. 发送给机器人执行
            3. 控制执行频率
        """
        last_queue_len = 1000  # 上次队列长度
        is_new_batch = False  # 是否有新批次
        while self.running:
            start_t = time.perf_counter()
            position = None
            try:
                # 阻塞等待获取动作数据
                while self.running:
                    with self.position_queue_lock:
                        if self.position_queue:
                            is_new_batch = False
                            if last_queue_len < len(self.position_queue):
                                is_new_batch = True  # 检测到新批次
                            last_queue_len = len(self.position_queue)
                            position = self.position_queue.popleft()  # 从队列取出动作
                            break
                    time.sleep(0.001)
                if not self.running:
                    break

                # 打印动作信息
                print(f"{'new ' if is_new_batch else ''}{position}")
                # 执行动作
                self._execute_robot_action(position)

                # 控制执行频率
                dt = time.perf_counter() - start_t
                wait_time = self.EXECUTION_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"动作执行线程异常: {e}")
                time.sleep(0.001)

    def _execute_robot_action(self, position):
        """根据机器人类型执行具体的动作控制

        参数:
            position: 动作位置数据 (7,) - [x, y, z, roll, pitch, yaw, gripper]
        """
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            # 控制末端位姿(前6个关节位置)
            self.robot.piper.EndPoseCtrl(*(int(x) for x in position[:6]))
            # 控制夹爪开合(应用非线性衰减)
            gripper = self.grapper_attenuation(position[6], self.grapper_factor)
            self.robot.piper.GripperCtrl(gripper, 1000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type for action execution: {robot_type}")

    def _plan_subtask(self):
        """子任务规划线程函数
        
        定期调用多模态大模型，根据当前相机图像确定下一个子任务
        
        循环执行:
            1. 获取俯视相机图像
            2. 调用Qwen3-VL模型进行子任务规划
            3. 更新当前子任务
            4. 控制规划频率
        """
        if self.subtask_model is None or self.subtask_processor is None:
            return
            
        # 系统提示词，定义三个子任务
        SYSTEM_PROMPT = "You are a robot action planner. Given a top-down image of a robotic workspace, output the next single sub-task for the robot arm. Now there are three subtasks, and the task prompts are as follows: Task one: Grab the package and place it on the pallet. Task two: Flip the package if the barcode is not facing up. Task three: Grab the package and place it into the box. You are required to output a task prompt based on the current image and strictly output one of the three prompts above."
        
        while self.running:
            start_t = time.perf_counter()
            try:
                # 获取俯视相机图像
                observation = self.robot.get_observation()
                image_top = None
                for key, value in observation.items():
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        # 假设top相机包含'top'关键字
                        if 'top' in key.lower():
                            image_top = value
                            break
                
                if image_top is None:
                    print("未找到俯视相机图像")
                    time.sleep(0.1)
                    continue
                
                # 构建对话消息
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_top},
                            {"type": "text", "text": "What is the next sub-task the robot arm should perform?"},
                        ],
                    }
                ]
                
                # 准备推理输入
                inputs = self.subtask_processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.subtask_model.device)
                
                # 推理生成子任务
                with torch.inference_mode():
                    generated_ids = self.subtask_model.generate(**inputs, max_new_tokens=128)
                
                # 解析输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.subtask_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                # 更新当前子任务(线程安全)
                new_subtask = "You are a parcel sorter. " + output_text[0]
                with self.subtask_lock:
                    if new_subtask != self.current_subtask:
                        print(f"子任务变化: {self.current_subtask} -> {new_subtask}")
                        self.current_subtask = new_subtask
                        
                        # 子任务变化时重置策略
                        self.policy.reset()
                
                # 控制规划频率
                dt = time.perf_counter() - start_t
                print(f"子任务规划完成. 耗时: {dt:.2f}s")
                wait_time = self.SUBTASK_PLANNING_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)
                    
            except Exception as e:
                print(f"子任务规划线程异常: {e}")
                time.sleep(0.1)

    def start(self):
        """启动推理系统

        流程:
            1. 初始化机器人
            2. 创建并启动三个工作线程(采集推理、执行动作、子任务规划)
            3. 进入主循环
        """
        if self.running:
            return
        self.running = True

        # 初始化机器人(连接、加载策略、使能)
        self.init_robot()

        # 创建工作线程
        self.infer_collect_thread = threading.Thread(target=self._collect_and_infer, daemon=True)
        self.execute_thread = threading.Thread(target=self._execute_action, daemon=True)
        
        # 只有当子任务规划模型可用时才创建该线程
        if self.subtask_model is not None:
            self.subtask_planning_thread = threading.Thread(target=self._plan_subtask, daemon=True)

        # 启动线程
        self.infer_collect_thread.start()
        self.execute_thread.start()
        if self.subtask_planning_thread is not None:
            self.subtask_planning_thread.start()

        print("所有线程已启动，开始机器人控制循环...")

    def stop(self):
        """停止所有线程并清理资源

        流程:
            1. 设置停止标志
            2. 等待线程结束
        """
        self.running = False

        # 等待线程结束
        if self.infer_collect_thread:
            self.infer_collect_thread.join(timeout=1.0)
        if self.execute_thread:
            self.execute_thread.join(timeout=1.0)
        if self.subtask_planning_thread:
            self.subtask_planning_thread.join(timeout=1.0)
        print("所有线程已停止")


@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """推理入口函数

    参数:
        cfg: 推理配置对象

    流程:
        1. 创建推理系统实例
        2. 启动系统(初始化机器人、启动线程)
        3. 主进程空闲等待
        4. 接收中断信号后优雅关闭
    """
    # 创建推理系统实例
    inference_system = RobotInferenceSystem(cfg)

    try:
        # 启动线程化推理
        inference_system.start()

        # 主进程空闲运行(1小时)
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

"""

# 命令行Aliases(用于快速操作)
alias c='clear'
alias s='ls'
alias e='exit'
alias ll='ls -hl'
alias cs='clear; ls'
alias cdb='cd ..'
alias cdbb='cd ../..'
alias cdbbb='cd ../../..'
alias cdf='cd -'

# 启动命令示例(包含子任务规划)
python lerobot/inference_smooth.py \
    --robot.type=single_piper \
    --robot.id=single_piper_robot \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --policy.compile_model=false \
    --robot.ctrl_mode="eef" \
    --policy.type=pi05 \
    --ckpt_path=/home/hpc/VLA/lerobot_models/pi05/2_28_rotate6d_300/24000/pretrained_model \
    --task="Pick up the express package and put it on the tray." \
    --qwen3_vl_path=/home/hpc/VLA/models/Qwen3-vl-2B-subtask/result \
    --subtask_planning_period=10.0

"""
