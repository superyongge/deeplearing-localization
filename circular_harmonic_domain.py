import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.signal import stft, istft
import librosa
import os
import pyroomacoustics as pra
import soundfile as sf

class CircularHarmonicBeamformer:
    """圆谐域波束形成器，用于实现TF-CHB-S-R-CNN算法的特征提取"""
    
    def __init__(self, radius=0.02, n_mics=4, fs=16000, max_order=None):
        """
        初始化圆谐域波束形成器
        
        参数:
            radius: 阵列半径 (m)
            n_mics: 麦克风数量
            fs: 采样率 (Hz)
            max_order: 最大圆谐阶数，默认为n_mics/2-1或n_mics/2
        """
        self.radius = radius
        self.n_mics = n_mics
        self.fs = fs
        self.c = 343.0  # 声速 (m/s)
        
        # 计算最大阶数
        if max_order is None:
            if n_mics % 2 == 0:
                self.max_order = n_mics // 2 - 1
            else:
                self.max_order = n_mics // 2
        else:
            self.max_order = max_order
        
        # 计算麦克风位置 (角度，弧度)
        self.mic_angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    
    def _compute_circular_harmonics(self, stft_signals, order):
        """
        计算STFT信号的第n阶圆谐分量
        
        参数:
            stft_signals: 麦克风信号的STFT，形状为[n_mics, n_freqs, n_frames]
            order: 圆谐阶数
            
        返回:
            ch: 圆谐分量，形状为[n_freqs, n_frames]
        """
        n_mics, n_freqs, n_frames = stft_signals.shape
        ch = np.zeros((n_freqs, n_frames), dtype=complex)
        
        # 打印调试信息
        # print(f"Computing circular harmonics for order {order}")
        # print(f"Input STFT shape: {stft_signals.shape}")
        
        for m in range(n_mics):
            angle_factor = np.exp(-1j * order * self.mic_angles[m])
            ch += stft_signals[m] * angle_factor
        
        ch /= n_mics  # 归一化
        return ch
    
    def _compute_mode_strength(self, stft_signals, k):
        """
        计算给定波数k下的模强度
        
        参数:
            stft_signals: 麦克风信号的STFT，形状为[n_mics, n_freqs, n_frames]
            k: 波数 (rad/m)
            
        返回:
            mode_strengths: 各阶模强度，形状为[2*max_order+1, n_frames]
        """
        _, n_freqs, n_frames = stft_signals.shape
        mode_strengths = np.zeros((2*self.max_order+1, n_frames), dtype=complex)
        
        for n_idx, n in enumerate(range(-self.max_order, self.max_order+1)):
            # 计算第n阶圆谐分量
            ch = self._compute_circular_harmonics(stft_signals, n)
            
            # 应用均衡因子
            if k == 0:  # 处理零频率点以避免除零
                equalization = 1.0
            else:
                # Bessel函数计算均衡因子
                jn_kr = special.jn(n, k * self.radius)
                if abs(jn_kr) > 1e-10:  # 避免除以非常小的值
                    equalization = 1.0 / (1j**n * jn_kr)
                else:
                    equalization = 0.0
            
            # 频率独立相位因子
            phase_factor = np.exp(1j * n * 0)  # 对应于论文中的0°方向
            
            # 计算并存储模强度
            mode_strengths[n_idx] = ch[0] * equalization * phase_factor
        
        return mode_strengths
    
    def compute_zero_mode_power(self, mic_signals, nfft=1024, hop_length=None):
        """
        计算0阶模强度功率，用于选择性处理中的TF点选择
        
        参数:
            mic_signals: 麦克风信号，形状为[n_mics, signal_length]
            nfft: FFT点数
            hop_length: 帧移，默认为nfft//2
            
        返回:
            E0: 0阶模强度功率，形状为[n_freqs, n_frames]
        """
        if hop_length is None:
            hop_length = nfft // 2
        
        # 计算STFT
        n_mics, _ = mic_signals.shape
        stft_signals = []
        
        for m in range(n_mics):
            _, _, Sm = stft(mic_signals[m], fs=self.fs, window='hann', 
                            nperseg=nfft, noverlap=nfft-hop_length,
                            return_onesided=True)
            stft_signals.append(Sm)
        
        # 将列表转换为numpy数组
        stft_signals = np.array(stft_signals)
        
        # 计算0阶圆谐分量
        ch0 = self._compute_circular_harmonics(stft_signals, 0)
        
        # 计算功率
        E0 = np.abs(ch0)**2
        
        return E0
    
    def selected_processing(self, mic_signals, selection_ratio=0.9, nfft=1024, hop_length=None):
        """
        实现论文中的选择性处理(Selected Processing)，选择高功率TF点
        
        参数:
            mic_signals: 麦克风信号，形状为[n_mics, signal_length]
            selection_ratio: 选择的高功率TF点比例，默认为0.9 (90%)
            nfft: FFT点数
            hop_length: 帧移，默认为nfft//2
            
        返回:
            Ef: 选择标志，形状为[n_freqs, n_frames]，1表示选择，0表示不选择
        """
        if hop_length is None:
            hop_length = nfft // 2
        
        # 计算0阶模强度功率
        E0 = self.compute_zero_mode_power(mic_signals, nfft, hop_length)
        
        # 将功率按降序排序
        E0_sorted = np.sort(E0.flatten())[::-1]
        
        # 确定功率阈值
        threshold_idx = int(len(E0_sorted) * selection_ratio)
        threshold = E0_sorted[threshold_idx]
        
        # 生成选择标志
        Ef = np.zeros_like(E0)
        Ef[E0 >= threshold] = 1
        
        return Ef
    
    def randomized_processing(self, Ef_list, n_sources):
        """
        实现论文中的随机化处理(Randomized Processing)，随机分配TF点到不同声源
        
        参数:
            Ef_list: 多个方向的选择标志列表，每个元素形状为[n_freqs, n_frames]
            n_sources: 声源数量
            
        返回:
            Ef_rand: 随机化后的选择标志，形状为[n_sources, n_freqs, n_frames]
        """
        if len(Ef_list) < n_sources:
            raise ValueError(f"需要至少{n_sources}个方向的选择标志")
        
        n_freqs, n_frames = Ef_list[0].shape
        Ef_rand = np.zeros((n_sources, n_freqs, n_frames))
        
        # 对每个频带和时间帧进行随机分配
        for f in range(n_freqs):
            for t in range(n_frames):
                # 获取当前TF点在所有方向上的选择标志
                tf_flags = [Ef[f, t] for Ef in Ef_list[:n_sources]]
                
                # 如果多个声源在此TF点都被选中，随机分配给一个声源
                if sum(tf_flags) > 1:
                    active_sources = np.where(tf_flags)[0]
                    chosen_source = np.random.choice(active_sources)
                    
                    # 只保留被随机选中的声源
                    for s in range(n_sources):
                        Ef_rand[s, f, t] = 1 if s == chosen_source else 0
                else:
                    # 如果只有一个声源被选中或没有声源被选中，保持原样
                    for s in range(n_sources):
                        Ef_rand[s, f, t] = tf_flags[s]
        
        return Ef_rand
    
    def extract_tf_chb_s_r_features(self, mic_signals, doas, nfft=1024, hop_length=None, selection_ratio=0.9):
        """
        提取TF-CHB-S-R特征，用于声源定位
        
        参数:
            mic_signals: 麦克风信号，形状为[n_mics, signal_length]
            doas: 待评估的DOA列表 (角度，弧度)
            nfft: FFT点数
            hop_length: 帧移，默认为nfft//2
            selection_ratio: 选择的高功率TF点比例
            
        返回:
            features: TF-CHB-S-R特征，形状为[n_doas, n_freqs]
        """
        if hop_length is None:
            hop_length = nfft // 2
        
        n_mics, _ = mic_signals.shape
        n_doas = len(doas)
        
        print(f"Computing STFT for {n_mics} microphones...")
        # 计算STFT
        stft_signals = []
        for m in range(n_mics):
            _, _, Sm = stft(mic_signals[m], fs=self.fs, window='hann', 
                           nperseg=nfft, noverlap=nfft-hop_length,
                           return_onesided=True)
            stft_signals.append(Sm)
        
        # 将所有麦克风的STFT整合成一个数组
        stft_signals = np.array(stft_signals)
        print(f"STFT shape: {stft_signals.shape}")
        
        # 应用选择性处理
        print("Applying selective processing...")
        Ef = self.selected_processing(mic_signals, selection_ratio, nfft, hop_length)
        print(f"Selection flag shape: {Ef.shape}")
        
        # 初始化TF-CHB-S特征
        n_freqs, n_frames = stft_signals.shape[1], stft_signals.shape[2]
        features = np.zeros((n_doas, n_freqs))
        
        print(f"Computing beam energy for {n_doas} DOAs...")
        # 计算各个DOA方向的波束能量
        for d_idx, doa in enumerate(doas):
            beam_energy = np.zeros((n_freqs, n_frames))
            
            for f in range(n_freqs):
                # 计算波数
                freq = f * self.fs / nfft
                k = 2 * np.pi * freq / self.c
                
                # 对每个阶数计算的结果累加到beam_energy
                beam_energy_f = np.zeros(n_frames, dtype=complex)
                
                for n in range(-self.max_order, self.max_order+1):
                    # 计算圆谐分量
                    ch = self._compute_circular_harmonics(stft_signals[:, :, :], n)
                    
                    # 计算均衡因子
                    if k == 0:  # 处理零频率点以避免除零
                        equalization = 1.0
                    else:
                        jn_kr = special.jn(n, k * self.radius)
                        if abs(jn_kr) > 1e-10:
                            equalization = 1.0 / (1j**n * jn_kr)
                        else:
                            equalization = 0.0
                    
                    # 计算相位因子
                    phase_factor = np.exp(1j * n * doa)
                    
                    # 累加到波束能量 (确保乘以相应频点的选择标志)
                    beam_energy_f += ch[f] * equalization * phase_factor
                
                # 计算能量并应用选择标志
                beam_energy[f] = np.abs(beam_energy_f)**2 * Ef[f]
            
            # 对各频段求平均，得到特征
            features[d_idx] = np.mean(beam_energy, axis=1)
            
            # 显示进度
            if (d_idx + 1) % 10 == 0 or d_idx == n_doas - 1:
                print(f"Processed {d_idx + 1}/{n_doas} DOAs")
        
        # 归一化特征
        print("Normalizing features...")
        features = features / np.max(features) if np.max(features) > 0 else features
        
        return features
    
    def visualize_features(self, features, doas_deg):
        """
        可视化TF-CHB-S-R特征
        
        参数:
            features: TF-CHB-S-R特征，形状为[n_doas, n_freqs]
            doas_deg: DOA列表 (角度，度)
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制特征热图
        plt.subplot(2, 1, 1)
        plt.imshow(features, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(label='Feature Magnitude')
        plt.xlabel('Frequency Bin')
        plt.ylabel('DOA (index)')
        plt.title('TF-CHB-S-R Features')
        
        # 绘制DOA响应图
        plt.subplot(2, 1, 2)
        doa_response = np.mean(features, axis=1)
        plt.plot(doas_deg, doa_response)
        plt.grid(True)
        plt.xlabel('DOA (degrees)')
        plt.ylabel('Response')
        plt.title('DOA Response')
        
        # 在DOA响应图上标记峰值
        peaks = find_peaks(doa_response, doas_deg)
        for peak_doa, peak_val in peaks:
            plt.plot(peak_doa, peak_val, 'ro')
            plt.text(peak_doa, peak_val, f'{peak_doa:.1f}°', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def find_peaks(doa_response, doas_deg, threshold_ratio=0.7, min_distance_deg=20):
    """
    在DOA响应中寻找峰值
    
    参数:
        doa_response: DOA响应，形状为[n_doas]
        doas_deg: DOA列表 (角度，度)
        threshold_ratio: 阈值比例，相对于最大响应
        min_distance_deg: 峰值间的最小角度差 (度)
        
    返回:
        peaks: 峰值列表，每个元素为(doa, value)
    """
    # 计算阈值
    threshold = threshold_ratio * np.max(doa_response)
    
    # 初始化峰值列表
    peaks = []
    
    # 寻找局部最大值
    for i in range(1, len(doa_response)-1):
        if (doa_response[i] > doa_response[i-1] and 
            doa_response[i] > doa_response[i+1] and 
            doa_response[i] > threshold):
            
            # 检查是否与已有峰值距离足够远
            is_far_enough = True
            for peak_doa, _ in peaks:
                if abs(doas_deg[i] - peak_doa) < min_distance_deg:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                peaks.append((doas_deg[i], doa_response[i]))
    
    # 按响应值降序排序
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks


def demo():
    """演示圆谐域处理的使用"""
    # 创建输出目录
    audio_dir = "output/audio"
    fig_dir = "output/figures"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # 设置参数
    fs = 16000
    duration = 3  # 3秒录音
    rt60 = 0.5  # 混响时间为0.5秒
    room_dim = [9.0, 7.0, 3.0]  # 房间尺寸 (m)
    mic_radius = 0.04  # 麦克风阵列半径 (m)
    n_mics = 8  # 麦克风数量
    
    # 创建房间
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )
    
    # 创建UCA麦克风阵列
    center_pos = [4.5, 3.5, 1.7]  # 麦克风阵列中心位置
    R = pra.beamforming.circular_2D_array(
        center=[center_pos[0], center_pos[1]], M=n_mics, phi0=0, radius=mic_radius
    )
    R = np.vstack((R, np.ones(n_mics) * center_pos[2]))  # 添加z坐标
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    # 创建3秒的信号
    t = np.arange(0, duration, 1/fs)
    
    # 声源1 - 语音 (使用data中的语音文件或生成模拟语音)
    try:
        target_file = "data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac"
        source1, sr = librosa.load(target_file, sr=fs, duration=duration)
        # 确保长度为3秒
        if len(source1) < duration * fs:
            source1 = np.pad(source1, (0, duration * fs - len(source1)), 'constant')
        else:
            source1 = source1[:int(duration * fs)]
    except:
        print("无法加载语音文件，使用模拟语音信号")
        np.random.seed(0)
        source1 = np.random.randn(len(t))
        source1 = librosa.effects.preemphasis(source1)
    
    # 声源2 - 噪声
    try:
        noise_file = "data/NoiseData/siren_noise.wav"
        source2, sr = librosa.load(noise_file, sr=fs, duration=duration)
        # 确保长度为3秒
        if len(source2) < duration * fs:
            source2 = np.pad(source2, (0, duration * fs - len(source2)), 'constant')
        else:
            source2 = source2[:int(duration * fs)]
    except:
        print("无法加载噪声文件，使用模拟噪声信号")
        source2 = 0.5 * np.sin(2 * np.pi * 500 * t)
    
    # 声源位置
    source_pos1 = [2.0, 1.5, 1.7]  # 约45度方向
    source_pos2 = [8.0, 6.0, 1.7]  # 约-45度方向
    
    # 保存原始音频
    os.makedirs(audio_dir, exist_ok=True)
    sf.write(f"{audio_dir}/source_1_original.wav", source1, fs)
    sf.write(f"{audio_dir}/source_2_original.wav", source2, fs)
    print(f"Original source 1 saved to: {audio_dir}/source_1_original.wav")
    print(f"Original source 2 saved to: {audio_dir}/source_2_original.wav")
    
    # 添加声源到房间
    room.add_source(source_pos1, signal=source1)
    room.add_source(source_pos2, signal=source2)
    
    # 运行仿真
    print("运行房间声学仿真...")
    room.simulate()
    print(f"仿真完成! 混响时间: {rt60} 秒")
    
    # 保存混合信号
    mixed_signals = room.mic_array.signals
    sf.write(f"{audio_dir}/mixed_signals.wav", mixed_signals.T, fs)
    print(f"Mixed signals saved to: {audio_dir}/mixed_signals.wav")
    
    # 单独保存每个麦克风的信号
    for m in range(n_mics):
        sf.write(
            f"{audio_dir}/mixed_signals_mic_{m+1}.wav", 
            mixed_signals[m, :], fs
        )
        print(f"Microphone {m+1} signal saved to: {audio_dir}/mixed_signals_mic_{m+1}.wav")
    
    # 创建圆谐域波束形成器
    chb = CircularHarmonicBeamformer(radius=mic_radius, n_mics=n_mics, fs=fs)
    
    # 使用麦克风信号提取特征
    doas_deg = np.linspace(-180, 170, 36)  # 10度间隔
    doas_rad = np.deg2rad(doas_deg)
    
    print("提取圆谐域特征...")
    # 确保麦克风信号格式正确 [n_mics, signal_length]
    print(f"Microphone signals shape: {mixed_signals.shape}")
    features = chb.extract_tf_chb_s_r_features(
        mixed_signals, doas_rad, nfft=1024, hop_length=512, selection_ratio=0.9
    )
    
    # 可视化并保存特征图
    plt.figure(figsize=(10, 8))
    # 绘制特征热图
    plt.subplot(2, 1, 1)
    plt.imshow(features, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Feature Magnitude')
    plt.xlabel('Frequency Bin')
    plt.ylabel('DOA (degrees)')
    plt.title('TF-CHB-S-R Features')
    
    # 绘制DOA响应图
    plt.subplot(2, 1, 2)
    doa_response = np.mean(features, axis=1)
    plt.plot(doas_deg, doa_response)
    plt.grid(True)
    plt.xlabel('DOA (degrees)')
    plt.ylabel('Response')
    plt.title('DOA Response')
    
    # 在DOA响应图上标记峰值
    peaks = find_peaks(doa_response, doas_deg)
    for peak_doa, peak_val in peaks:
        plt.plot(peak_doa, peak_val, 'ro')
        plt.text(peak_doa, peak_val, f'{peak_doa:.1f}°', 
                ha='center', va='bottom')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/doa_features.png", dpi=300, bbox_inches='tight')
    print(f"DOA features plot saved to: {fig_dir}/doa_features.png")
    
    # 绘制房间布局图
    plt.figure(figsize=(10, 8))
    
    # 创建自定义的2D房间布局图而不是使用室内声学库的3D绘图
    room_fig, room_ax = plt.subplots(figsize=(10, 8))
    
    # 绘制房间边界
    room_ax.add_patch(plt.Rectangle((0, 0), room_dim[0], room_dim[1], fill=False, color='black', linewidth=2))
    
    # 绘制麦克风阵列
    mic_x = R[0, :]
    mic_y = R[1, :]
    room_ax.scatter(mic_x, mic_y, color='blue', s=50, marker='o', label='Microphones')
    
    # 绘制麦克风阵列中心
    room_ax.scatter(center_pos[0], center_pos[1], color='cyan', s=80, marker='x', label='Array Center')
    
    # 绘制麦克风阵列圆
    circle = plt.Circle((center_pos[0], center_pos[1]), mic_radius, fill=False, color='blue', linestyle='--')
    room_ax.add_patch(circle)
    
    # 绘制声源
    room_ax.scatter(source_pos1[0], source_pos1[1], color='red', s=100, marker='*', label='Speech Source')
    room_ax.scatter(source_pos2[0], source_pos2[1], color='green', s=100, marker='*', label='Noise Source')
    
    # 添加声源方向线
    # 从麦克风阵列中心到声源1绘制线
    room_ax.plot([center_pos[0], source_pos1[0]], [center_pos[1], source_pos1[1]], 'r--', alpha=0.6)
    # 从麦克风阵列中心到声源2绘制线
    room_ax.plot([center_pos[0], source_pos2[0]], [center_pos[1], source_pos2[1]], 'g--', alpha=0.6)
    
    # 计算并显示DOA角度
    dx1 = source_pos1[0] - center_pos[0]
    dy1 = source_pos1[1] - center_pos[1]
    angle1 = np.degrees(np.arctan2(dy1, dx1))
    
    dx2 = source_pos2[0] - center_pos[0]
    dy2 = source_pos2[1] - center_pos[1]
    angle2 = np.degrees(np.arctan2(dy2, dx2))
    
    # 添加角度标注
    room_ax.annotate(f"DOA: {angle1:.1f}°", 
                     xy=((source_pos1[0] + center_pos[0])/2, (source_pos1[1] + center_pos[1])/2),
                     xytext=(10, 10), textcoords='offset points', color='red')
    room_ax.annotate(f"DOA: {angle2:.1f}°", 
                     xy=((source_pos2[0] + center_pos[0])/2, (source_pos2[1] + center_pos[1])/2),
                     xytext=(10, 10), textcoords='offset points', color='green')
    
    # 设置图形属性
    room_ax.set_xlim(-0.5, room_dim[0] + 0.5)
    room_ax.set_ylim(-0.5, room_dim[1] + 0.5)
    room_ax.set_xlabel('X (m)')
    room_ax.set_ylabel('Y (m)')
    room_ax.set_title(f"Room Acoustics Simulation (RT60 = {rt60} s)")
    room_ax.legend(loc='upper right')
    room_ax.grid(True)
    
    # 添加比例尺
    room_ax.set_aspect('equal')
    
    # 保存图像
    room_fig.tight_layout()
    room_fig.savefig(f"{fig_dir}/room_layout.png", dpi=300, bbox_inches='tight')
    print(f"Room layout saved to: {fig_dir}/room_layout.png")
    
    print("圆谐域处理示例完成！")


if __name__ == "__main__":
    demo()
    
    print("\n此模块提供了圆谐域波束形成器的实现，包括：")
    print("1. TF-CHB-S-R特征提取")
    print("2. 选择性处理和随机化处理")
    print("3. DOA响应的可视化和峰值检测")
    print("适用于实现TF-CHB-S-R-CNN声源定位算法的特征提取部分") 