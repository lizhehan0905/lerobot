import numpy as np
from scipy.spatial.transform import Rotation as R

# 欧拉角 转 rotate6D 输入:q 角度
def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=True).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz',degrees=True)


# 原始欧拉角（假设是XYZ顺序）
euler_xyz = np.array([-160.944, -80.708, -163.636])  # 度

scipy_euler = euler_to_rotate6d(euler_xyz)
print("scipy_euler Rotate6D:", scipy_euler)

uuuu = rotate6d_to_xyz(scipy_euler)
print(uuuu)
