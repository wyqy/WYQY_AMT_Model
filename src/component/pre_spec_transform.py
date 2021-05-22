import os, sys
import soundfile as sf
import librosa as rs
import numpy as np


# cqt_filter_fft = rs.constantq.__cqt_filter_fft
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial


class rs_spec_logMel():
    """
    功能: 输入音频, 输出对数梅尔谱变换时频表示
    https://blog.csdn.net/weixin_37598106/article/details/106104549
    
    Usage:
        src, _ = librosa.load(filename)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src)
    """

    # 必须将spec先变换完成为numpy, 为方便可以存在每个wav对应的文件夹下的npy
    # 暂时不清楚使用tf还是librosa变换
    # 如果要用librosa切记不要resampling!!!!!
    # 若要使用tensorflow变换, 可尝试使用多进程释放资源
    # https://blog.csdn.net/Forrest97/article/details/106895624
    # https://blog.csdn.net/jiangwei741/article/details/103802108

    def __init__(self, sr=16000, n_mel=229,
                 lower_edge_hertz=30.0,
                 upper_edge_hertz=8000.0,
                 hop_length=128,
                 frame_length=2048,
                 fft_length=2048,
                 window='hann'):
        # if fft_length not given
        # fft_length = 2**N for integer N such that 2**N >= frame_length.
        self.sr = sr
        self.n_mel = n_mel
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fft_length = fft_length  # 输出n_fft为1+fft_length//2
        self.n_fft = 1+fft_length//2
        self.window = window
        self.npdtype = np.float32
        self.sample_batch = initial.config['spec.logmel.sample.size']

        # 构建Mel滤波器
        self.linear_to_mel_weight_matrix = rs.filters.mel(
            sr=self.sr,
            n_fft=self.fft_length,  # ?
            n_mels=self.n_mel,
            fmin=self.lower_edge_hertz,
            fmax=self.upper_edge_hertz,
            htk=False,
            norm='slaney',
            dtype=self.npdtype)

    def __call__(self, path):
        # https://my.oschina.net/qinhui99/blog/899014?
        # 强制使用soundfile
        # 格式: (sample, channel)
        y, sr = sf.read(file=path, dtype='float32')
        # librosa重采样: scipy.signal.resample_poly
        # 格式: (channel, sample)
        y = np.transpose(y)
        y_pure_frames = y.shape[1]*self.sr//sr
        y = rs.core.resample(
            y, orig_sr=sr, target_sr=self.sr, res_type='linear', fix=False)
        y_frames = y.shape[1]
        
        # 最大变换阈值
        if y_frames > self.sample_batch:
            audio_offset = 0
            slice_offset = 0
            slice_samples = 0
            slice_frames = 0
            # 创建目标大小的numpy数组(frames, n_mels, channels)
            ret = np.zeros(
                ((1+(y_pure_frames-self.frame_length)//self.hop_length), self.n_mel, 2), dtype=self.npdtype)
            while audio_offset < y_frames-self.frame_length:
                # 计算输入长度
                slice_samples = min(
                    self.sample_batch, y_frames-audio_offset)
                # 计算输出长度
                slice_frames = 1 + \
                    (slice_samples-self.frame_length)//self.hop_length
                # 输入格式(channels, samples), 输出格式(frames, n_mels, channels)
                ret_slice = self.transform(
                    y[:, audio_offset: audio_offset+slice_samples], slice_frames)
                # 直接写入
                ret[slice_offset:slice_offset+slice_frames, :, :] = ret_slice
                # 更新偏移
                audio_offset = audio_offset+slice_samples-self.frame_length+self.hop_length
                slice_offset += slice_frames
            return ret
        else:
            # 输出格式(frames, n_mels, channels)
            # 计算输出长度
            return self.transform(y,
                                  1 + (y_pure_frames-self.frame_length)//self.hop_length)

    def transform(self, y, length) -> np.ndarray:
        # 输入格式(channels, samples), 输出格式(frames, n_mels, channels)

        # STFT变换并且直接得到magnitude (1 + n_fft//2, n_frames), 两个通道分别变换
        spectrograms = np.zeros((2, self.n_fft, length), dtype=self.npdtype)
        stfts = np.abs(rs.core.stft(y[0, :], n_fft=self.fft_length,
                                    hop_length=self.hop_length, win_length=self.frame_length,
                                    window=self.window, center=False, dtype=np.complex64),
                       dtype=self.npdtype)
        spectrograms[0, :, 0:stfts.shape[1]] = stfts
        stfts = np.abs(rs.core.stft(y[1, :], n_fft=self.fft_length,
                                    hop_length=self.hop_length, win_length=self.frame_length,
                                    window=self.window, center=False, dtype=np.complex64),
                       dtype=self.npdtype)
        spectrograms[1, :, 0:stfts.shape[1]] = stfts

        # 得到梅尔谱Mel Spectrum (channels, n_mels, n_frames)
        mel_spectrograms = np.matmul(
            self.linear_to_mel_weight_matrix, spectrograms)

        # 对数/分贝(任取其一):
        # spectrograms = np.log(spectrograms + 1e-12)
        log_mel_spectrograms = rs.core.amplitude_to_db(
            mel_spectrograms, ref=np.max, top_db=120.0)
        # 两通道分别归一化到[0, 1]
        s_max = np.max(log_mel_spectrograms[0, :, :])
        s_min = np.min(log_mel_spectrograms[0, :, :])
        log_mel_spectrograms[0, :, :] = (log_mel_spectrograms[0, :, :]-s_min)/(s_max-s_min)
        s_max = np.max(log_mel_spectrograms[1, :, :])
        s_min = np.min(log_mel_spectrograms[1, :, :])
        log_mel_spectrograms[1, :, :] = (log_mel_spectrograms[1, :, :]-s_min)/(s_max-s_min)

        # 还原格式(channels, n_mels, n_frames) -> (n_frames, n_mels, channels)
        logmel = np.transpose(log_mel_spectrograms, axes=[2, 1, 0])

        return logmel


class rs_spec_cqt():
    """
    功能: 输入音频, 输出对数梅尔谱变换时频表示
    https://blog.csdn.net/weixin_37598106/article/details/106104549
    
    Usage:
        src, _ = librosa.load(filename)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src)
    """

    # 必须将spec先变换完成为numpy, 为方便可以存在每个wav对应的文件夹下的npy
    # 暂时不清楚使用tf还是librosa变换
    # 如果要用librosa切记不要resampling!!!!!
    # 若要使用tensorflow变换, 可尝试使用多进程释放资源
    # https://blog.csdn.net/Forrest97/article/details/106895624
    # https://blog.csdn.net/jiangwei741/article/details/103802108

    def __init__(self, n_bins=356,
                 bins_per_octave=48,
                 fmin=27.5,
                 frame_length=2048,
                 hop_length=512,
                 window='hann'):
        # if fft_length not given
        # fft_length = 2**N for integer N such that 2**N >= frame_length.
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.hop_length = hop_length
        self.window = window
        self.npdtype = np.float32
        self.frame_length = frame_length
        self.sample_batch = initial.config['spec.cqt.sample.size']
        # 其它固定参数
        self.sr = 44100
        self.filter_scale = 1
        self.norm = 1
        self.sparsity = 0.01
        self.cqt_minlength = 128*self.frame_length
        
    def __call__(self, path):
        # 格式: (channel, sample)
        y, _ = rs.core.load(path, sr=self.sr, mono=False, dtype=self.npdtype)
        y_frames = y.shape[1]

        # 最大变换阈值

        # TODO: 直接完整变换!!!
        # TODO: 测试取不同的hop_length!!!
        '''
        if y_frames > self.sample_batch:
            audio_offset = 0
            slice_offset = 0
            slice_samples = 0
            slice_frames = 0
            # 创建目标大小的numpy数组(frames, n_bins, channels)
            ret = np.zeros(
                ((1+(y_frames-self.frame_length)//self.hop_length), self.n_bins, 2), dtype=self.npdtype)
            while not audio_offset > y_frames-self.cqt_minlength:
                # 计算输入长度
                slice_samples = min(
                    self.sample_batch, y_frames-audio_offset)
                # 因为CQT最少需要frame_length*128的samples, 长度不可忽略
                if y_frames-1-audio_offset-self.sample_batch < self.cqt_minlength:
                    slice_samples = y_frames-audio_offset
                # 计算输出长度
                slice_frames = 1 + \
                    (slice_samples-self.frame_length)//self.hop_length
                # 输入格式(channels, samples), 输出格式(frames, n_bins, channels)
                ret_slice = self.transform(
                    y[:, audio_offset: audio_offset+slice_samples], slice_frames)
                # 直接写入
                ret[slice_offset:slice_offset+slice_frames, :, :] = ret_slice
                # 更新偏移
                audio_offset = audio_offset+slice_samples-self.frame_length+self.hop_length
                slice_offset += slice_frames
            return ret
        else:
            # 输出格式(frames, n_bins, channels)
            # 计算输出长度
            return self.transform(y, 1 + (y_frames-self.frame_length)//self.hop_length)
        '''
        return self.transform(y, 1 + (y_frames-self.frame_length)//self.hop_length)

    def transform(self, y, length) -> np.ndarray:
        # 输入格式(channels, samples), 输出格式(frames, n_bins, channels)
        spectrograms = np.zeros((2, self.n_bins, length), dtype=self.npdtype)

        # 直接使用librosa变换
        # CQT变换
        cqt = rs.core.hybrid_cqt(y[0, :], sr=self.sr, hop_length=self.hop_length,
                                 fmin=self.fmin, n_bins=self.n_bins,
                                 bins_per_octave=self.bins_per_octave, tuning=0.0,
                                 filter_scale=self.filter_scale,
                                 norm=self.norm, sparsity=self.sparsity,
                                 window=self.window)
        spectrograms[0, :, :] = np.abs(cqt[:, 2:length+2])
        cqt = rs.core.hybrid_cqt(y[1, :], sr=self.sr, hop_length=self.hop_length,
                                 fmin=self.fmin, n_bins=self.n_bins,
                                 bins_per_octave=self.bins_per_octave, tuning=0.0,
                                 filter_scale=self.filter_scale,
                                 norm=self.norm, sparsity=self.sparsity,
                                 window=self.window)
        spectrograms[1, :, :] = np.abs(cqt[:, 2:length+2])
        # 对数/分贝(任取其一):
        # spectrograms = np.log(spectrograms + 1e-12)
        spectrograms = rs.core.amplitude_to_db(spectrograms, ref=np.max, top_db=120.0)
        # 两通道分别归一化到[0, 1]
        s_max = np.max(spectrograms[0, :, :])
        s_min = np.min(spectrograms[0, :, :])
        spectrograms[0, :, :] = (spectrograms[0, :, :]-s_min)/(s_max-s_min)
        s_max = np.max(spectrograms[1, :, :])
        s_min = np.min(spectrograms[1, :, :])
        spectrograms[1, :, :] = (spectrograms[1, :, :]-s_min)/(s_max-s_min)

        # 还原格式(channels, n_bins, n_frames) -> (n_frames, n_bins, channels)
        spec = np.transpose(spectrograms, axes=[2, 1, 0])

        return spec


def main():
    '''
    trans_logmel = rs_spec_logMel(sr=initial.config['spec.logmel.sampling'],
                                  lower_edge_hertz=initial.config['spec.logmel.lower.edge'],
                                  upper_edge_hertz=initial.config['spec.logmel.upper.edge'],
                                  hop_length=initial.config['spec.logmel.hop_length'],
                                  frame_length=initial.config['spec.logmel.frame_length'],
                                  n_mel=initial.config['spec.logmel.n_mel'],
                                  fft_length=initial.config['spec.logmel.fft_length'],
                                  window=initial.config['spec.logmel.window'])
    out = trans_logmel(
        'data\\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
    out.shape
    '''

    spec_cqt = rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                           bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                           fmin=initial.config['spec.cqt.fmin'],
                           frame_length=initial.config['spec.cqt.frame_length'],
                           hop_length=initial.config['spec.cqt.hop_length'],
                           window=initial.config['spec.cqt.window'])

    out = spec_cqt(
        'data\\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
    print(out.shape)
    return 0


if __name__ == '__main__':
  main()
