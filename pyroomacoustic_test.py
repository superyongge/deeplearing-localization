import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
import os

# 确保输出目录存在
fig_dir = "output/figures"
os.makedirs(fig_dir, exist_ok=True)

# L形房间的角落坐标
corners = np.array([[0,0], [0,3], [5,3], [5,1], [3,1], [3,0]]).T  # [x,y]

# 创建并绘制2D房间
room_2d = pra.Room.from_corners(corners)

# 绘制2D房间
fig, ax = plt.subplots(figsize=(10, 8))
room_2d.plot(ax=ax)
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 4])
ax.set_title('L-Shaped Room (2D View)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.grid(True)

# 在房间中添加标注
ax.text(1, 1.5, 'L-Shaped Room', fontsize=12, ha='center')

# 添加尺寸标注
# 水平尺寸
ax.annotate('', xy=(0, -0.2), xytext=(5, -0.2), arrowprops=dict(arrowstyle='<->'))
ax.text(2.5, -0.4, '5m', ha='center')
# 垂直尺寸
ax.annotate('', xy=(-0.2, 0), xytext=(-0.2, 3), arrowprops=dict(arrowstyle='<->'))
ax.text(-0.4, 1.5, '3m', va='center', rotation=90)

# 保存2D图像
plt.tight_layout()
plt.savefig(f"{fig_dir}/room_2d_layout.png", dpi=300, bbox_inches='tight')
print(f"2D Room layout saved to: {fig_dir}/room_2d_layout.png")

# 创建并绘制3D房间
room_3d = pra.Room.from_corners(corners)
room_3d.extrude(2.0)  # 设置房间高度为2米

# 创建3D图
fig3d = plt.figure(figsize=(12, 10))
ax3d = fig3d.add_subplot(111, projection='3d')
room_3d.plot(ax=ax3d)

# 设置3D图的属性
ax3d.set_xlim([0, 5])
ax3d.set_ylim([0, 3])
ax3d.set_zlim([0, 2])
ax3d.set_title('L-Shaped Room (3D View)')
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')

# 保存3D图像
plt.tight_layout()
plt.savefig(f"{fig_dir}/room_3d_layout.png", dpi=300, bbox_inches='tight')
print(f"3D Room layout saved to: {fig_dir}/room_3d_layout.png")

# 添加声源和麦克风阵列，并绘制完整的模拟场景
try:
    # 尝试载入音频文件
    audio_file = "output/audio/source_1_original.wav"
    if os.path.exists(audio_file):
        fs, signal = wavfile.read(audio_file)
    else:
        # 如果文件不存在，创建一个简单的信号
        fs = 16000
        duration = 2
        t = np.arange(0, duration, 1/fs)
        signal = 0.5 * np.sin(2 * np.pi * 500 * t)
        
    # 创建房间并添加声源和麦克风
    room_sim = pra.Room.from_corners(corners, fs=fs, ray_tracing=True, air_absorption=True)
    
    # 添加声源
    source_pos = [1.0, 1.0, 1.0]
    room_sim.add_source(source_pos, signal=signal)
    
    # 添加麦克风阵列
    mic_pos = [3.5, 2.0, 1.0]  # 麦克风阵列中心位置
    R = pra.beamforming.circular_2D_array(center=[mic_pos[0], mic_pos[1]], M=8, phi0=0, radius=0.04)
    R = np.vstack((R, np.ones(8) * mic_pos[2]))  # 添加z坐标
    room_sim.add_microphone_array(pra.MicrophoneArray(R, fs))
    
    # 绘制带有声源和麦克风的房间
    fig_sim = plt.figure(figsize=(12, 10))
    ax_sim = fig_sim.add_subplot(111, projection='3d')
    room_sim.plot(ax=ax_sim)
    
    # 突出显示声源和麦克风阵列
    ax_sim.scatter([source_pos[0]], [source_pos[1]], [source_pos[2]], color='red', s=100, marker='*', label='Source')
    ax_sim.scatter(R[0, :], R[1, :], R[2, :], color='blue', s=50, marker='o', label='Microphones')
    
    # 设置图的属性
    ax_sim.set_xlim([0, 5])
    ax_sim.set_ylim([0, 3])
    ax_sim.set_zlim([0, 2])
    ax_sim.set_title('Room Simulation with Source and Microphones')
    ax_sim.set_xlabel('X (m)')
    ax_sim.set_ylabel('Y (m)')
    ax_sim.set_zlabel('Z (m)')
    ax_sim.legend()
    
    # 保存模拟场景图像
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/room_simulation.png", dpi=300, bbox_inches='tight')
    print(f"Room simulation saved to: {fig_dir}/room_simulation.png")
    
except Exception as e:
    print(f"无法创建完整模拟场景: {e}")

print("所有房间布局图已成功生成！")
