import numpy as np


def fix_all_coordinates_to_top_right():
    # 1. 原始测量坐标 (RTK测量的背部X号位置) [X, Y, Z]
    # 对应布局：
    # K (左上)   F (右上)
    # A (左下)   Z (右下)
    raw_coords = {
        "K (Top-Left)": np.array([-1325727.185726, 5323473.758561, 3243296.345430]),
        "F (Top-Right)": np.array([-1325727.209639, 5323473.847246, 3243296.182657]),
        "A (Bottom-Left)": np.array([-1325726.930646, 5323473.816573, 3243296.330301]),
        "Z (Bottom-Right)": np.array([-1325726.965898, 5323473.906766, 3243296.189143])
    }

    # 2. 定义修正参数 (单位: 米)
    # 垂直偏移: 从背部X号中心到手机顶部边框的距离
    offset_up = 0.055  # 向上 5.5 cm

    # 水平偏移: 从背部X号中心(左右居中)到右侧边框的距离
    # 假设手机宽度约 7.5cm，一半即 3.75cm
    offset_right = 0.0375  # 向右 3.75 cm

    # 3. 计算方向向量 (基于正确的平行布局)

    # 【向上方向】：由 A (后/下) 指向 K (前/上)
    vec_up_raw = raw_coords["K (Top-Left)"] - raw_coords["A (Bottom-Left)"]
    unit_up = vec_up_raw / np.linalg.norm(vec_up_raw)  # 归一化

    # 【向右方向】：由 K (左) 指向 F (右)
    vec_right_raw = raw_coords["F (Top-Right)"] - raw_coords["K (Top-Left)"]
    unit_right = vec_right_raw / np.linalg.norm(vec_right_raw)  # 归一化

    # 4. 计算总位移向量 (这是一个固定的空间向量)
    # 总位移 = (向上方向 * 垂直距离) + (向右方向 * 水平距离)
    shift_vector = (unit_up * offset_up) + (unit_right * offset_right)

    print(f"{'=' * 15} 修正参数 (修正到右上角) {'=' * 15}")
    print(f"垂直偏移 (向上): {offset_up * 100} cm")
    print(f"水平偏移 (向右): {offset_right * 100} cm")
    print(f"应用位移向量 (XYZ): {np.round(shift_vector, 4)}\n")

    print(f"{'=' * 20} 所有手机修正后真值坐标 {'=' * 20}")
    print(f"{'手机代号':<20} | {'X Coordinate':<15} | {'Y Coordinate':<15} | {'Z Coordinate':<15}")
    print("-" * 75)

    # 5. 循环计算并输出
    # 假设所有手机摆放方向一致，则应用同一个 shift_vector
    for name, pos in raw_coords.items():
        new_pos = pos + shift_vector
        print(f"{name:<20} | {new_pos[0]:.6f}      | {new_pos[1]:.6f}      | {new_pos[2]:.6f}")


# 运行函数
if __name__ == "__main__":
    fix_all_coordinates_to_top_right()