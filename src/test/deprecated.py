import os
import wave
import librosa as rs
import numpy as np
import pandas as pd
import tensorflow as tf

import src.preprocessing.data_import as data_import
import src.preprocessing.spec_transform as spec_transform


"""
init_spec_cqt函数
功能: 对数据集进行CQT预处理, 保存到指定目录
输入: config_content配置信息, trainsets原始csv的信息!!!
输出: 
"""
def init_spec_cqt(config_content: dict, trainsets: pd.core.frame.DataFrame):
    # 读取配置信息
    audio_common_path = config_content['dataset.audio.path']
    spec_common_path = config_content['spec.batch.path']
    batch_size = config_content['spec.batch.size'] - 1

    # 按原始csv的index依次处理, 满batch_size就保存
    iter_batch = 0
    iter_offset = 0
    this_length = 0
    this_size = 0
    spec_offset = 0

    for index in range(len(trainsets)):
        path = trainsets.iloc[index].at['audio_filename']
        path = os.path.join(audio_common_path, path)

        # 若是第一个则创建矩阵
        if iter_offset == 0:
            # 计算batch长度, 准备矩阵(一左一右)
            this_length, this_size = data_import.batch_whole_length(
                trainsets, audio_common_path, config_content['spec.cqt.hop_length'], config_content['spec.batch.size'], index)
            spec_left_matrix = np.empty(
                (config_content['spec.cqt.n_bins'], this_length))
            spec_right_matrix = np.empty(
                (config_content['spec.cqt.n_bins'], this_length))

        # 变换并保存
        spec = spec_transform.transform_spec_cqt(path, config_content)
        spec_offset = data_import.batch_base_address(
            trainsets, audio_common_path, config_content['spec.cqt.hop_length'], iter_batch*config_content['spec.batch.size'], iter_offset)
        spec_left_matrix[:, spec_offset:(
            spec_offset+spec.shape[1])] = spec[:, :, 0]
        spec_right_matrix[:, spec_offset:(
            spec_offset+spec.shape[1])] = spec[:, :, 1]

        # 若是最后一个则保存
        if iter_offset == (this_size-1):
            path_left_string = str(iter_batch) + '_cqt_left.npy'
            path_right_string = str(iter_batch) + '_cqt_right.npy'
            path = os.path.join(spec_common_path, path_left_string)
            np.save(path, spec_left_matrix)
            path = os.path.join(spec_common_path, path_right_string)
            np.save(path, spec_right_matrix)
            iter_batch += 1
            iter_offset = 0
        else:
            iter_offset += 1


"""
load_spec_cqt函数
功能: 按batch标号载入预先处理好的CQT变换
输入: 
输出: 
"""
def load_spec_cqt(config_content: dict, trainsets: pd.core.frame.DataFrame, batch_number: int) -> np.ndarray:
    # 读取配置信息
    spec_batch_size = config_content['spec.batch.size']
    spec_common_path = config_content['spec.batch.path']
    train_batch_size = config_content['spec.batch.size']

    # 计算第一个位置(注意没有溢出检查)
    batch_base = train_batch_size*(batch_number-1)

    # 计算batch大小, 防止最后一个不能整除
    load_size = min(train_batch_size, trainsets.shape[0] - batch_base)

    # 逐个读入
    for iter in range(0, load_size):
        index = trainsets.iloc[batch_base+iter].at['original_index']

        # 计算在哪个文件的哪个位置
        cqt_batch_loc = index // spec_batch_size
        cqt_offset_loc = (index % spec_batch_size)-1

        path_left_string = str(cqt_batch_loc) + '_cqt_left.npy'
        path_right_string = str(cqt_batch_loc) + '_cqt_right.npy'
        cqt_batch_left_path = os.path.join(spec_common_path, path_left_string)
        cqt_batch_right_path = os.path.join(
            spec_common_path, path_right_string)

        # 取数组
        spec_batch_left = np.load(cqt_batch_left_path)
        spec_batch_right = np.load(cqt_batch_right_path)

## 以下代码没有修改, 有问题!!!

        batch_left_offset = data_import.batch_base_address(
            trainsets, audio_common_path, config_content['spec.cqt.hop_length'], iter_batch*config_content['spec.batch.size'], iter_offset)

        spec_element = spec_batch_left[:, cqt_offset_loc]

        # 分情况讨论
        if iter == 0:
            if load_size == 0:
                train_batch = spec_element[:, np.newaxis]
            if load_size != 0:
                train_batch = spec_element
        if iter != 0:
            train_batch = np.dstack(train_batch, spec_element)

    return train_batch


def batch_whole_length(sets: pd.core.frame.DataFrame, common_path: str, hop_length: int, batch_size: int, batch_offset: int) -> int:
    """
    batch_whole_length函数
    功能: 计算batch总长度
    输入: sets标注集, hop_length每秒采样数, batch_size元素个数, batch_offset偏移地址(从0起始)
    输出: 总长度和个数
    """

    # offset 从0开始!
    # 计算batch中element个数
    batch_base = batch_offset*batch_size
    real_size = min(sets.shape[0] - batch_base, batch_size)
    ret = 0

    # 逐个计算element长度
    for iter in range(real_size):
        file_path = sets.iloc[batch_base+iter].at['audio_filename']
        path_string = os.path.join(common_path, file_path)
        audio_file = wave.open(path_string, mode='rb')
        audio_length = (audio_file.getnframes() // hop_length) + 1
        audio_file.close()
        ret += audio_length

    return ret, real_size


def batch_base_address(sets: pd.core.frame.DataFrame, common_path: str, hop_length: int, element_base: int, element_offset: int) -> int:
    """
    batch_base_address函数
    功能: 计算某个element在batch中的基地址
    输入: sets标注集, hop_length每秒采样数, element_base整个batch起始地址, element_offset元素在batch的偏移地址(从0起始)
    输出: 元素的基地址
    """

    # offset 从0开始!
    ret = 0

    # 逐个计算element长度
    for iter in range(element_offset+1):
        file_path = sets.iloc[element_base+iter].at['audio_filename']
        path_string = os.path.join(common_path, file_path)
        audio_file = wave.open(path_string, mode='rb')
        audio_length = (audio_file.getnframes() // hop_length) + 1
        audio_file.close()
        ret += audio_length

    return ret


def train_spec_cqt(offset: int) -> np.ndarray:
    """
    train_spec_cqt函数
    功能: 对指定文件进行CQT预处理和归一化, 并返回
    输入: offset第几个文件(从0开始!)
    输出: CQT变换矩阵356*length*2的ndarray
    """

    # 读取配置信息
    audio_common_path = initial.config['dataset.audio.path']

    # 错误验证? 可能需要抛出异常!
    if offset >= initial.trainsets.shape[0]:
        return (np.zeros((initial.config['spec.cqt.n_bins'], 1, 2), dtype=np.float32),
                np.zeros((initial.config['spec.cqt.n_bins'], 1, 2), dtype=np.float32))

    # 计算路径
    path = initial.trainsets.iloc[offset].at['audio_filename']
    path = os.path.join(audio_common_path, path)

    # 变换
    spec = spec_transform.transform_spec_cqt(path)

    # 左右两声道分别归一化
    spec_min = np.min(spec[:, :, 0])
    spec_max = np.max(spec[:, :, 0])
    spec_range = spec_max - spec_min
    spec[:, :, 0] = (spec[:, :, 0] - spec_min) / spec_range
    spec_min = np.min(spec[:, :, 1])
    spec_max = np.max(spec[:, :, 1])
    spec_range = spec_max - spec_min
    spec[:, :, 1] = (spec[:, :, 1] - spec_min) / spec_range

    return spec


def train_spec_mfcc(offset: int) -> np.ndarray:
    """
    train_spec_mfcc函数
    功能: 对指定文件进行MFCC预处理和归一化, 并返回
    输入: offset第几个文件(从0开始!)
    输出: MFCC变换矩阵180*length*2的ndarray
    """

    # 读取配置信息
    audio_common_path = initial.config['dataset.audio.path']

    # 错误验证? 可能需要抛出异常!
    if offset >= initial.trainsets.shape[0]:
        return (np.zeros((initial.config['spec.mfcc.n_mfcc'], 1, 2), dtype=np.float32),
                np.zeros((initial.config['spec.mfcc.n_mfcc'], 1, 2), dtype=np.float32))

    # 计算路径
    path = initial.trainsets.iloc[offset].at['audio_filename']
    path = os.path.join(audio_common_path, path)

    # 变换
    spec = spec_transform.transform_spec_mfcc(path)

    # 左右两声道分别归一化
    spec_min = np.min(spec[:, :, 0])
    spec_max = np.max(spec[:, :, 0])
    spec_range = spec_max - spec_min
    spec[:, :, 0] = (spec[:, :, 0] - spec_min) / spec_range
    spec_min = np.min(spec[:, :, 1])
    spec_max = np.max(spec[:, :, 1])
    spec_range = spec_max - spec_min
    spec[:, :, 1] = (spec[:, :, 1] - spec_min) / spec_range

    return spec


def train_sheet(offset: int, spec: np.ndarray) -> np.ndarray:
    """
    train_sheet函数
    功能: 对指定文件读取对应MIDI, 并返回
    输入: offset第几个文件(从0开始!), spec时频的元组
    输出: CQT变换矩阵356*length的ndarray
    """

    # 读取配置信息
    audio_common_path = initial.config['dataset.audio.path']
    midi_common_path = initial.config['dataset.midi.path']

    # 错误验证? 可能需要抛出异常!
    if offset >= initial.trainsets.shape[0]:
        return (np.zeros((initial.config['spec.cqt.n_bins'], 1), dtype=np.float32),
                np.zeros((initial.config['spec.cqt.n_bins'], 1), dtype=np.float32))

    # 计算路径
    audio_path = initial.trainsets.iloc[offset].at['audio_filename']
    audio_path = os.path.join(audio_common_path, audio_path)
    midi_path = initial.trainsets.iloc[offset].at['midi_filename']
    midi_path = os.path.join(midi_common_path, midi_path)

    # 变换?!
    sheet = sheet_transform.midi_trans(
        midi_path, audio_path)

    return sheet


cqt_filter_fft = rs.constantq.__cqt_filter_fft

class tf_PseudoCqt():
    """
    功能: 输入音频, 输出PseudoCQT变换时频表示, 使用tf快速变换
    https://gist.github.com/voodoohop/2089c61218605f758289cada102c1b9e
    A class to compute pseudo-CQT with Tensorflow.
    Written by Keunwoo Choi and adapted to tensorflow by Thomas Haferlach
    API (+implementations) follows librosa (https://librosa.github.io/librosa/generated/librosa.core.pseudo_cqt.html)
    
    Usage:
        src, _ = librosa.load(filename)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src)
    """

    def __init__(self, sr=22050, mono=False,
                 hop_length=512, fmin=None, n_bins=84,
                 bins_per_octave=12, filter_scale=1,
                 norm=1, sparsity=0.01, window='hann', scale=True):

        assert scale
        assert window == "hann"
        if fmin is None:
            fmin = 2 * 32.703195  # note_to_hz('C2') because C1 is too low

        fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
                                             filter_scale, norm, sparsity,
                                             hop_length=hop_length, window=window)

        # because it was sparse. (n_bins, n_fft)
        fft_basis = np.abs(fft_basis.astype(dtype=np.float32)).todense()
        self.fft_basis = tf.expand_dims(
            tf.convert_to_tensor(fft_basis), 0)  # (n_freq, n_bins)
        self.sr = sr
        self.mono = mono
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scale = scale
        self.window = tf.signal.hann_window
        self.npdtype = np.float32

    def __call__(self, path):
        # 计算总采样数
        audio = wave.open(path, mode='rb')
        samples = (audio.getnframes() * self.sr) / audio.getframerate()
        audio.close()

        if not self.mono:
            file = tf.io.read_file(path)
            y, _ = tf.audio.decode_wav(
                file, desired_channels=2, desired_samples=samples)
            return self.fstereo(y)
        else:
            file = tf.io.read_file(path)
            y, _ = tf.audio.decode_wav(
                file, desired_channels=1, desired_samples=samples)
            y = tf.reshape(y, [-1])
            return self.fmono(y)

    # XLA加速
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)],
        experimental_compile=True)
    def fstereo(self, y):
        # 注意y的格式是(duration, channel)
        # 变换格式
        y = tf.transpose(y)
        stft_magnitudes = tf.transpose(tf.math.real(tf.signal.stft(y, fft_length=self.n_fft,
                                                                   frame_length=self.hop_length*4,
                                                                   frame_step=self.hop_length,
                                                                   window_fn=self.window,
                                                                   pad_end=True)), perm=[0, 2, 1])

        D = tf.math.pow(stft_magnitudes, 2)   # n_freq, time
        # without EPS, backpropagating through CQT can yield NaN.
        D = tf.math.sqrt(D + 0.0001)
        # Project onto the pseudo-cqt basis
        C = tf.matmul(self.fft_basis, D)  # n_bins, time
        C /= tf.math.sqrt(float(self.n_fft))  # because `scale` is always True

        return C

    # XLA加速
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None), dtype=tf.float32)],
        experimental_compile=True)
    def fmono(self, y):
        stft_magnitudes = tf.transpose(tf.math.real(tf.signal.stft(y, fft_length=self.n_fft,
                                                                   frame_length=self.hop_length*4,
                                                                   frame_step=self.hop_length,
                                                                   window_fn=self.window,
                                                                   pad_end=True)))

        D = tf.math.pow(stft_magnitudes, 2)   # n_freq, time
        # without EPS, backpropagating through CQT can yield NaN.
        D = tf.math.sqrt(D + 0.0001)
        # Project onto the pseudo-cqt basis
        C = tf.matmul(self.fft_basis, D)  # n_bins, time
        C /= tf.math.sqrt(float(self.n_fft))  # because `scale` is always True

        return C


class tf_MFCC():
    """
    功能: 输入音频, 输出MFCC变换时频表示, 使用tf快速变换
    https://blog.csdn.net/weixin_37598106/article/details/106104549
    
    Usage:
        src, _ = librosa.load(filename)
        cqt_calculator = PseudoCqt()
        cqt_calculator(src)
    """

    def __init__(self, sr=22050, mono=False,
                 lower_edge_hertz=125.0, upper_edge_hertz=3800.0,
                 hop_length=512, n_mfcc=20, n_fft=2048,
                 window='hann'):
        assert window == "hann"

        # if fft_length not given
        # fft_length = 2**N for integer N such that 2**N >= frame_length.
        # shape (25, 513)
        self.sr = sr
        self.mono = mono
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.window = tf.signal.hann_window
        self.npdtype = np.float32

    def __call__(self, path):
        # 计算总采样数
        audio = wave.open(path, mode='rb')
        samples = (audio.getnframes() * self.sr) / audio.getframerate()
        audio.close()

        if not self.mono:
            file = tf.io.read_file(path)
            y, _ = tf.audio.decode_wav(
                file, desired_channels=2, desired_samples=samples)
            return self.fstereo(y)
        else:
            file = tf.io.read_file(path)
            y, _ = tf.audio.decode_wav(
                file, desired_channels=1, desired_samples=samples)
            y = tf.reshape(y, [-1])
            return self.fmono(y)

    # XLA加速
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)],
        experimental_compile=True)
    def fstereo(self, y):
        # 注意y的格式是(duration, channel)
        # 变换格式
        y = tf.transpose(y)

        # STFT变换
        stfts = tf.signal.stft(y, frame_length=self.hop_length,
                               frame_step=self.hop_length, fft_length=self.n_fft)
        spectrograms = tf.abs(stfts)
        # Warp the linear scale spectrograms into the mel-scale.
        stfts = tf.ensure_shape(stfts, shape=(2, None, self.n_fft // 2 + 1))
        num_spectrogram_bins = stfts.shape.as_list()[-1]  # 513
        # Mel滤波器
        # 构建Mel滤波器
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mfcc,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sr,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz)
        mel_spectrograms = tf.matmul(spectrograms,
                                     linear_to_mel_weight_matrix)
        # shape (25, 40)
        mel_spectrograms.set_shape(spectrograms.shape[:-1]
                                   .concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # log变换
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-12)
        # DCT变换
        # shape (1 + (wav-win_length)/win_step, dct)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)
        # 取低频维度上的部分值输出, 语音能量大多集中在低频域, 数值一般取13?
        mfcc = mfccs[..., :self.n_mfcc]

        # 注意mfcc的格式是(batch, duration, n_mfcc)
        # 还原格式
        mfcc = tf.transpose(mfcc, perm=[0, 2, 1])

        return mfcc

    # XLA加速
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None), dtype=tf.float32)],
        experimental_compile=True)
    def fmono(self, y):
        # STFT变换
        stfts = tf.signal.stft(y, frame_length=self.hop_length,
                               frame_step=self.hop_length, fft_length=self.n_fft)
        spectrograms = tf.abs(stfts)
        # Warp the linear scale spectrograms into the mel-scale.
        stfts = tf.ensure_shape(stfts, shape=(None, self.n_fft // 2 + 1))
        num_spectrogram_bins = stfts.shape.as_list()[-1]  # 513
        # Mel滤波器
        # 构建Mel滤波器
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mfcc,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sr,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz)
        mel_spectrograms = tf.matmul(spectrograms,
                                     linear_to_mel_weight_matrix)
        # shape (25, 40)
        mel_spectrograms.set_shape(spectrograms.shape[:-1]
                                   .concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # log变换
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-12)
        # DCT变换
        # shape (1 + (wav-win_length)/win_step, dct)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms)
        # 取低频维度上的部分值输出, 语音能量大多集中在低频域, 数值一般取13?
        mfcc = mfccs[..., :self.n_mfcc]

        # 还原格式
        y = tf.transpose(y)

        return mfcc
