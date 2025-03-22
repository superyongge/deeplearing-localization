import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
import soundfile as sf
import os
import shutil
from matplotlib import rcParams
import matplotlib

# 设置matplotlib参数，使用不依赖中文字体的方式
matplotlib.rcParams['axes.unicode_minus'] = False
# 不设置中文字体，使用英文标签代替，避免中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']

class UniformCircularArray:
    """均匀圆形麦克风阵列(UCA)实现"""
    
    def __init__(self, radius=0.02, n_mics=4):
        """
        初始化均匀圆形阵列
        
        参数:
            radius: 阵列半径(米)，默认为2cm
            n_mics: 麦克风数量，默认为4
        """
        self.radius = radius
        self.n_mics = n_mics
        self.positions = self._calculate_positions()
        
    def _calculate_positions(self):
        """计算麦克风位置"""
        angles = np.linspace(0, 2*np.pi, self.n_mics, endpoint=False)
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        z = np.zeros(self.n_mics)
        
        # 返回3D坐标
        return np.vstack((x, y, z)).T
    
    def plot(self, ax=None):
        """绘制麦克风阵列布局"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            
        positions = self.positions
        ax.scatter(positions[:, 0], positions[:, 1], marker='o', s=100, color='blue', label='Microphones')
        ax.add_patch(plt.Circle((0, 0), self.radius, fill=False, color='red', linestyle='--'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Uniform Circular Array (r={self.radius}m, {self.n_mics} mics)')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        
        return ax


class RoomSimulator:
    """基于pyroomacoustics的房间声学模拟器"""
    
    def __init__(self, room_dim=(9.7, 7.05, 3.0), rt60=0.6):
        """
        初始化房间模拟器
        
        参数:
            room_dim: 房间尺寸(长,宽,高)，单位为米
            rt60: 混响时间(秒)，默认为0.6s
        """
        self.room_dim = np.array(room_dim)
        self.rt60 = rt60
        self.fs = 16000  # 采样率 (Hz)
        self.room = None
        
    def setup_room(self):
        """设置房间的声学特性"""
        # 使用pyroomacoustics的方法根据RT60计算房间的吸声系数
        e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        
        # 创建房间
        self.room = pra.ShoeBox(
            self.room_dim,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            ray_tracing=True,  # 启用射线追踪以提高混响模拟的准确性
            air_absorption=True
        )
        
        return self.room
    
    def add_sources(self, source_positions, audio_files, output_dir="output/audio"):
        """
        添加多个声源到房间，并保存原始声源文件的副本
        
        参数:
            source_positions: 声源位置列表，每个元素是[x, y, z]坐标
            audio_files: 音频文件路径列表
            output_dir: 输出目录，用于保存原始声源文件
        """
        if self.room is None:
            self.setup_room()
            
        if len(source_positions) != len(audio_files):
            raise ValueError("声源位置数量必须等于音频文件数量")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        source_signals = []
        
        # 加载音频文件并添加到房间
        for i, (pos, file_path) in enumerate(zip(source_positions, audio_files)):
            # 加载音频文件
            if file_path.endswith('.flac'):
                audio_data, sample_rate = sf.read(file_path)
                # 保存原始声源的副本为WAV格式
                source_output_path = os.path.join(output_dir, f"source_{i+1}_original.wav")
                sf.write(source_output_path, audio_data, sample_rate)
            else:
                sample_rate, audio_data = wavfile.read(file_path)
                # 保存原始声源的副本
                source_output_path = os.path.join(output_dir, f"source_{i+1}_original.wav")
                # 如果是整数格式，需要转换为浮点数进行存储
                if audio_data.dtype == np.int16:
                    tmp_data = audio_data.astype(np.float32) / 32767.0
                    sf.write(source_output_path, tmp_data, sample_rate)
                else:
                    sf.write(source_output_path, audio_data, sample_rate)
                
            # 转换为浮点数
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            
            # 如果采样率不匹配，需要重采样(这里简化处理，实际应使用专业的重采样库)
            if sample_rate != self.fs:
                print(f"Warning: Audio sample rate ({sample_rate}Hz) does not match simulation sample rate ({self.fs}Hz)")
            
            # 添加声源
            self.room.add_source(pos, signal=audio_data)
            source_signals.append(audio_data)
            
            print(f"Original source {i+1} saved to: {source_output_path}")
            
        return self.room, source_signals
    
    def add_microphone_array(self, mic_array, center_pos=(4.5, 3.5, 1.7)):
        """
        将麦克风阵列添加到房间
        
        参数:
            mic_array: UniformCircularArray对象
            center_pos: 阵列中心位置，默认在房间中央附近
        """
        if self.room is None:
            self.setup_room()
        
        # 计算相对于中心的麦克风位置
        mic_positions = mic_array.positions
        
        # 转换为全局坐标
        global_positions = np.zeros_like(mic_positions)
        for i in range(mic_positions.shape[0]):
            global_positions[i] = mic_positions[i] + np.array(center_pos)
        
        # 创建麦克风阵列对象
        self.room.add_microphone_array(pra.MicrophoneArray(global_positions.T, self.fs))
        
        return self.room
    
    def simulate(self):
        """
        执行模拟
        
        返回:
            mic_signals: 麦克风信号，形状为[n_mics, signal_length]
        """
        if self.room is None:
            raise ValueError("Please set up the room and add sources and microphones first")
        
        # 运行模拟
        self.room.simulate()
        
        # 返回麦克风录制的信号
        return self.room.mic_array.signals
    
    def save_audio(self, signals, filename, normalize=True):
        """
        保存音频信号到WAV文件
        
        参数:
            signals: 音频信号
            filename: 文件名
            normalize: 是否归一化
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # 归一化
        if normalize:
            max_val = np.max(np.abs(signals))
            if max_val > 0:
                signals = signals / max_val * 0.9  # 留一些余量
        
        # 保存为WAV文件
        self.room.mic_array.to_wav(
            filename,
            norm=normalize,
            bitdepth=np.int16,
        )
        print(f"Mixed audio saved to: {filename}")
        
        # 保存每个麦克风的单独信号
        base_dir = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        for i in range(signals.shape[0]):
            mic_file = os.path.join(base_dir, f"{base_name}_mic_{i+1}.wav")
            wavfile.write(mic_file, self.fs, (signals[i] * 32767).astype(np.int16))
            print(f"Microphone {i+1} signal saved to: {mic_file}")
    
    def plot_room(self, fig_path=None):
        """
        绘制房间、声源和麦克风位置
        
        参数:
            fig_path: 图片保存路径，如果提供则保存图片
        """
        if self.room is None:
            raise ValueError("Please set up the room and add sources and microphones first")
        
        # 绘制房间
        fig, ax = self.room.plot()
        ax.set_title(f"Room Acoustics Simulation (RT60 = {self.rt60} s)")
        
        # 确保保存图片
        if fig_path:
            try:
                plt.figure(fig.number)  # 确保当前图形是我们想要保存的图形
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Room layout saved to: {fig_path}")
            except Exception as e:
                print(f"Failed to save room layout: {e}")
            
        return fig, ax
    
    def plot_rir(self, mic_idx=0, src_idx=0, fig_path=None):
        """
        绘制房间脉冲响应
        
        参数:
            mic_idx: 麦克风索引
            src_idx: 声源索引
            fig_path: 图片保存路径，如果提供则保存图片
        """
        if self.room is None:
            raise ValueError("Please set up the room and add sources and microphones first")
        
        # 绘制单个RIR
        fig, ax = plt.subplots(figsize=(10, 4))
        rir = self.room.rir[mic_idx][src_idx]
        time = np.arange(len(rir)) / self.fs * 1000  # 转换为毫秒
        ax.plot(time, rir)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Room Impulse Response (RT60 = {self.rt60} s)')
        ax.grid(True)
        
        # 确保保存图片
        if fig_path:
            try:
                plt.figure(fig.number)  # 确保当前图形是我们想要保存的图形
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"RIR plot saved to: {fig_path}")
            except Exception as e:
                print(f"Failed to save RIR plot: {e}")
            
        return fig, ax


def demo():
    """使用pyroomacoustics演示UCA和房间声学模拟"""
    print("Starting demo function...")
    
    # 创建输出目录
    audio_dir = "output/audio"
    fig_dir = "output/figures"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print("Output directories created.")
    
    # 创建UCA
    print("Creating Uniform Circular Array...")
    uca = UniformCircularArray(radius=0.04, n_mics=8)
    print("UCA created successfully.")
    
    # 创建房间模拟器
    print("Creating Room Simulator...")
    room_sim = RoomSimulator(room_dim=(9.7, 7.05, 3.0), rt60=0.6)
    print("Room Simulator created successfully.")
    
    # 设置房间
    print("Setting up room...")
    room_sim.setup_room()
    print("Room setup completed.")
    
    # 加载音频文件
    print("Loading audio files...")
    target_file = "data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac"  # 目标语音
    noise_file = "data/NoiseData/siren_noise.wav"  # 噪声源
    print(f"Target file: {target_file}")
    print(f"Noise file: {noise_file}")
    
    # 添加声源
    print("Adding sources...")
    source_positions = [
        [2.0, 1.5, 1.7],  # Target source position: front left
        [8.0, 6.0, 1.7]   # Noise source position: back right
    ]
    try:
        room_sim.add_sources(source_positions, [target_file, noise_file], output_dir=audio_dir)
        print("Sources added successfully.")
    except Exception as e:
        print(f"Error adding sources: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 添加麦克风阵列
    print("Adding microphone array...")
    try:
        room_sim.add_microphone_array(uca, center_pos=(4.5, 3.5, 1.7))
        print("Microphone array added successfully.")
    except Exception as e:
        print(f"Error adding microphone array: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行模拟
    print("Simulating room acoustics...")
    try:
        mic_signals = room_sim.simulate()
        print("Simulation completed successfully.")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存混合信号
    print("Saving mixed signals...")
    try:
        room_sim.save_audio(mic_signals, f"{audio_dir}/mixed_signals.wav")
        print("Mixed signals saved successfully.")
    except Exception as e:
        print(f"Error saving mixed signals: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nRoom acoustics simulation with pyroomacoustics completed!")
    print(f"Reverberation time: {room_sim.rt60} seconds")
    print(f"Original source files saved in: {audio_dir}/")
    print(f"Mixed signals saved in: {audio_dir}/")
    
    # 单独处理每个图形绘制，防止一个失败影响其他图形
    # 1. 首先保存麦克风阵列图
    try:
        print("Attempting to save microphone array plot...")
        plt.figure(figsize=(8, 8))
        uca.plot()
        plt.savefig(f"{fig_dir}/microphone_array.png", dpi=300, bbox_inches='tight')
        print(f"Microphone array plot saved to: {fig_dir}/microphone_array.png")
        plt.close()
    except Exception as e:
        print(f"Error saving microphone array plot: {e}")
    
    # 2. 保存房间布局图 - 使用最简单的方式
    try:
        print("Attempting to save room layout plot...")
        # 创建一个新的示意图，不使用pyroomacoustics的复杂绘图功能
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制房间边界
        room_width, room_length = room_sim.room_dim[0], room_sim.room_dim[1]
        rect = plt.Rectangle((0, 0), room_width, room_length, fill=False, color='black')
        ax.add_patch(rect)
        
        # 绘制麦克风位置
        mic_positions = room_sim.room.mic_array.R.T  # 转置以获取所需格式
        ax.scatter(mic_positions[:, 0], mic_positions[:, 1], c='blue', marker='o', s=50, label='Microphones')
        
        # 绘制声源位置
        for i, source in enumerate(room_sim.room.sources):
            src_pos = source.position
            ax.scatter(src_pos[0], src_pos[1], c='red', marker='x', s=100, label=f'Source {i+1}')
        
        # 设置图像属性
        ax.set_xlim(-0.1, room_width + 0.1)
        ax.set_ylim(-0.1, room_length + 0.1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Room Layout (RT60 = {room_sim.rt60} s)')
        ax.legend()
        ax.grid(True)
        
        # 保存图像
        fig.savefig(f"{fig_dir}/room_layout.png", dpi=300, bbox_inches='tight')
        print(f"Room layout (Figure 1) saved to: {fig_dir}/room_layout.png")
        plt.close(fig)
    except Exception as e:
        print(f"Error saving room layout plot: {e}")
        import traceback
        traceback.print_exc()  # 打印完整堆栈跟踪，以便更好地理解错误
    
    print("\nThis module provides room acoustics simulation and uniform circular array implementation based on pyroomacoustics")


if __name__ == "__main__":
    demo()
    
    print("\nThis module provides room acoustics simulation and uniform circular array implementation based on pyroomacoustics")
    print("Main features:")
    print("1. Uniform Circular Array (UCA) generation")
    print("2. Use pyroomacoustics for precise room acoustics simulation")
    print("3. Multi-source recording simulation, supporting different audio file formats")
    print("4. Visualize room layout, source positions, and microphone array") 