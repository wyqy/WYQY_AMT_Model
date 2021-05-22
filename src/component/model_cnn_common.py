import os, sys
import random
from typing import Generator
import numpy as np
import numba as nb
from numba.typed import List
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.pre_spec_transform as spec_transform
import src.component.pre_sheet_transform as sheet_transform


class FocalLoss(keras.losses.Loss):
    '''
    FocalLoss类
    继承keras.losses.Loss类, 用于计算Focal Loss, 自带去除不确定样本功能
    使用该类不需要在最后做sigmoid
    '''

    def __init__(self, threshold=0.5, alpha=0.25, gamma=2.0,
                 reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name='focal_loss', **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.threshold = threshold
        self.alpha = alpha
        self.gamma = gamma

    #def __call__(self, y_true, y_pred, sample_weight=None):
    #    self.call(y_true, y_pred)

    def call(self, y_true, y_pred):
        y_true = tf.where(y_true > self.threshold, x=1.0, y=0.0)
        return tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred,
                    alpha=self.alpha, gamma=self.gamma, from_logits=True)
    

class WeightBinaryCrossentropy(keras.losses.Loss):
    '''
    WeightBinaryCrossentropy类
    继承keras.losses.Loss类, 用于计算正负样本权重不等的Binary交叉熵, 自带去除不确定样本功能
    使用该类不需要在最后做sigmoid
    '''

    def __init__(self, threshold=0.5, pos_weight=1,
                 reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name='wight_binary_cross_entropy', **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.threshold = threshold
        self.pos_weight = pos_weight

    #def __call__(self, y_true, y_pred, sample_weight=None):
    #    self.call(y_true, y_pred)

    def call(self, y_true, y_pred):
        y_true = tf.where(y_true > self.threshold, x=1.0, y=0.0)
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight)

    '''
    # XLA
    #@tf.function(experimental_compile=True)
    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        # y_true = tf.cast(y_true, y_pred.dtype)
        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        crossentropy = tf.multiply((1.0-y_true), y_pred) + \
                       tf.multiply((1.0+(self.pos_weight-1)*y_true),
                                   (tf.math.log(1.0+tf.exp(-1.0*tf.abs(y_pred))) + \
                                    tf.maximum(-1.0*y_pred, 0)))
        return crossentropy
    '''


class Macro_Binary_Fb_Evaluation(keras.metrics.Metric):
    '''
    Macro_Binary_Fb_Evaluation类
    继承keras.metrics.Metric类, 用于计算各种情况下的metrics(每个epoch运行完输出!)
    自带去除不确定样本功能!
    https://blog.csdn.net/qq_40836518/article/details/105295369
    https://zjmmf.com/2019/08/13/F1-Score%E8%AE%A1%E7%AE%97/
    https://blog.csdn.net/quiet_girl/article/details/70830796
    https://zhuanlan.zhihu.com/p/147663370
    '''

    def __init__(self, size=88, thresholds=0.5, beta=2, name='macro_binary_fb_evaluation', **kwargs):
        # beta=2, 强化recall, 尽可能多识别, 牺牲误报
        super(Macro_Binary_Fb_Evaluation, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', shape=(size,), initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', shape=(size,), initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', shape=(size,), initializer='zeros', dtype=tf.float32)
        self.thresholds = thresholds
        self.size = size
        self.beta = beta*beta
        self.min_delta = 1e-6

    # XLA
    @tf.function(experimental_compile=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 注意x, y的含义: 条件true=x...
        # y的形状(batch, size: 88/1)
        y_true = tf.cast(tf.where(y_true > self.thresholds, x=1, y=0), tf.int8)
        # 将y_pred应用sigmoid
        y_pred = tf.math.sigmoid(y_pred)
        y_pred = tf.cast(tf.where(y_pred > self.thresholds, x=1, y=0), tf.int8)

        tp = tf.math.count_nonzero(y_pred*y_true, axis=0, dtype=tf.float32)
        fp = tf.math.count_nonzero(y_pred*(1-y_true), axis=0, dtype=tf.float32)
        fn = tf.math.count_nonzero((1-y_pred)*y_true, axis=0, dtype=tf.float32)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    # XLA
    @tf.function(experimental_compile=True)
    def result(self):
        scalar = tf.divide(
            (1+self.beta)*self.tp,
            (1+self.beta)*self.tp+self.beta*self.fn+self.fp+self.min_delta)
        return tf.math.reduce_mean(scalar)

    def reset_states(self):
        self.tp.assign(tf.zeros(shape=(self.size,)))
        self.fp.assign(tf.zeros(shape=(self.size,)))
        self.fn.assign(tf.zeros(shape=(self.size,)))


class Train_CNN_DataGenerator():
    """
    Train_DataGenerator类:
    用于从原数据生成各种数据集
    """

    def __init__(self, io='train', output='multitone_start'):
        self.n_sheets = 0
        self.sheet_type = ''
        if output == 'multitone_start':
            self.n_sheets = 88
            self.sheet_type = 'start'
            self.spec_slice_size = initial.config['detect.tone.slice']
        if output == 'common_start':
            self.n_sheets = 1
            self.sheet_type = 'start'
            self.spec_slice_size = initial.config['detect.start.common.slice']
        if output == 'multitone_duration':
            self.n_sheets = 88
            self.sheet_type = 'duration'
            self.spec_slice_size = initial.config['detect.tone.slice']
        self.io = io
        # 不同执行周期公用
        if self.io == 'train':
            self.file_num = initial.trainsets.shape[0]
            self.batch_size = initial.config['detect.train.batch.size']
        if self.io == 'test':
            self.file_num = initial.testsets.shape[0]
            self.batch_size = initial.config['detect.predict.batch.size']
        # 初始化常量
        self.batch_threshold = self.batch_size // 2

        # 初始化变量
        self.audio_path = ''
        self.aunumpy_path = ''
        self.midi_path = ''
        self.mdnumpy_path = ''
        # spec变换初始化
        self.trans_cqt = spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                                    bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                                    fmin=initial.config['spec.cqt.fmin'],
                                                    frame_length=initial.config['spec.cqt.frame_length'],
                                                    hop_length=initial.config['spec.cqt.hop_length'],
                                                    window=initial.config['spec.cqt.window'])

    def __call__(self) -> Generator:
        """
        功能: 生成数据集, 其中spec优先使用numpy, 否则自行变换并保存
        输入: 
        输出: (input, label)组成的元组
        """

        # 生成器初始化, 注意: 一次一个音频文件/MIDI的一部分
        file_offset = 0

        # 生成对象
        while file_offset < self.file_num:
            # 读取下一个文件
            # 计算路径
            self.data_path(file_offset)

            # CQT变换
            # 格式(duration, n_bins, 2)
            # 优先使用numpy, 否则自行变换并保存
            if os.path.isfile(self.aunumpy_path):
                spec = np.load(self.aunumpy_path, allow_pickle=False)
            else:
                spec = self.trans_cqt(self.audio_path)
                np.save(self.aunumpy_path, spec, allow_pickle=False)

            # MIDI变换
            # 格式(duration, 88/1)
            sheet = self.sheet_transform()

            # 计算滑窗中心长度信息
            #spec_length = tf.shape(spec).numpy()[0]
            spec_length = spec.shape[0]
            spec_length = 1 + (spec_length - self.spec_slice_size)

            # 生成随机打乱的切片集合
            data_range = list(range(spec_length))

            while len(data_range) > 0:
                # 计算抽取长度, 多取一些
                if len(data_range) < self.batch_size+self.batch_threshold:
                    slice_size = len(data_range)
                else:
                    slice_size = min(self.batch_size, len(data_range))

                # 生成随机抽样序列
                # slice_list = random.sample(data_range, slice_size)
                # 生成顺序抽样序列(测试用)
                slice_list = list(range(min(data_range), min(data_range)+slice_size))
                # 去除已抽样数据
                data_range = list(set(data_range).difference(set(slice_list)))

                # 切片
                spec_slice, sheet_slice = self.data_sample(
                    spec, sheet, List(slice_list),
                    initial.config['spec.cqt.n_bins'], self.n_sheets,
                    self.spec_slice_size)

                # 变换数据格式?
                # spec_slice = tf.convert_to_tensor(spec_slice, dtype=tf.float32)
                # sheet_slice = tf.convert_to_tensor(sheet_slice, dtype=tf.float32)

                # 输出
                yield (spec_slice, sheet_slice)

            # 更新file_offset
            file_offset += 1

        # 结束返回(抛出异常)
        return

    def data_path(self, file_offset):
        if self.io == 'train':
            self.audio_path = initial.trainsets.iloc[file_offset].at['audio_filename']
            self.aunumpy_path = self.audio_path[:-3] + 'npy'
            self.midi_path = initial.trainsets.iloc[file_offset].at['midi_filename']
            if self.sheet_type == 'start':
                if self.n_sheets == 88:
                    self.mdnumpy_path = self.midi_path[:-4] + '_msmidi.npy'
                if self.n_sheets == 1:
                    self.mdnumpy_path = self.midi_path[:-4] + '_csmidi.npy'
            if self.sheet_type == 'duration':
                self.mdnumpy_path = self.midi_path[:-4] + '_mdmidi.npy'

        if self.io == 'test':
            self.audio_path = initial.testsets.iloc[file_offset].at['audio_filename']
            self.aunumpy_path = self.audio_path[:-3] + 'npy'
            self.midi_path = initial.testsets.iloc[file_offset].at['midi_filename']
            if self.sheet_type == 'start':
                if self.n_sheets == 88:
                    self.mdnumpy_path = self.midi_path[:-4] + '_msmidi.npy'
                if self.n_sheets == 1:
                    self.mdnumpy_path = self.midi_path[:-4] + '_csmidi.npy'
            if self.sheet_type == 'duration':
                self.mdnumpy_path = self.midi_path[:-4] + '_mdmidi.npy'

    def sheet_transform(self):
        if os.path.isfile(self.mdnumpy_path):
            sheet = np.load(self.mdnumpy_path, allow_pickle=False)
        else:
            # 加入缓存功能! 下同!
            if self.sheet_type == 'start':
                if self.n_sheets == 88:
                    sheet = sheet_transform.midi_trans_start(
                        self.midi_path, self.audio_path)
                    np.save(self.mdnumpy_path, sheet, allow_pickle=False)
                if self.n_sheets == 1:
                    sheet = sheet_transform.midi_trans_common(
                        self.midi_path, self.audio_path)
                    np.save(self.mdnumpy_path, sheet, allow_pickle=False)
            if self.sheet_type == 'duration':
                sheet = sheet_transform.midi_trans_duration(
                    self.midi_path, self.audio_path)
                np.save(self.mdnumpy_path, sheet, allow_pickle=False)
            
        return sheet

    @staticmethod
    @nb.jit(nopython=True)
    def data_sample(spec, sheet, slice_list, n_bins, n_sheets, spec_slice_size):
        # 新建数组
        spec_slice = np.zeros(
            (len(slice_list), spec_slice_size, n_bins, 2), dtype=np.float32)
        sheet_slice = np.zeros((len(slice_list), n_sheets), dtype=np.float32)
        # 新建变量
        iter = 0
        spec_mid_offset = spec_slice_size // 2

        # spec_slice, sheet_slice
        for index in slice_list:
            spec_slice[iter, :, :, :] = spec[index:(index+spec_slice_size), :, :]
            sheet_slice[iter, :] = sheet[index+spec_mid_offset, :]
            iter += 1

        return (spec_slice, sheet_slice)


def model_compile(type, model: keras.Model):
    """
    功能: 模型配置
    """
    if type == 'multitone_start':
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-2,
                #beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                name='Adam',
                clipnorm=10.0),  # l2 norm of the gradients is capped at the specified value
            # loss=model_cnn_common.WeightBinaryCrossentropy(
            #     threshold=0.4, # 使用不确定样本用于训练
            #     pos_weight=initial.config['detect.weight.loss.posweigh'],
            #     reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss=FocalLoss(
                threshold=0.4, # 使用不确定样本用于训练
                alpha=initial.config['detect.focal.loss.multistart.at'],
                gamma=initial.config['detect.focal.loss.gamma'],
                reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            metrics=Macro_Binary_Fb_Evaluation(
                size=88,
                thresholds=initial.config['detect.train.fn.threshold'],
                beta=initial.config['detect.train.fn.beta']))
    if type == 'common_start':
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-2,
                # beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                name='Adam',
                clipnorm=10.0),  # l2 norm of the gradients is capped at the specified value
            loss=WeightBinaryCrossentropy(
                threshold=0.6, # 去除不确定样本
                pos_weight=initial.config['detect.weight.loss.posweigh'],
                reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss=FocalLoss(
            #     threshold=0.6, # 去除不确定样本
            #     alpha=initial.config['detect.focal.loss.common.at'],
            #     gamma=initial.config['detect.focal.loss.gamma'],
            #     reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            metrics=Macro_Binary_Fb_Evaluation(
                size=1,
                thresholds=initial.config['detect.train.fn.threshold'],
                beta=initial.config['detect.train.fn.beta']))
    if type == 'multitone_duration':
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-2,
                #beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                name='Adam',
                clipnorm=10.0),  # l2 norm of the gradients is capped at the specified value
            # loss=model_cnn_common.WeightBinaryCrossentropy(
            #     threshold=0.5,
            #     pos_weight=initial.config['detect.weight.loss.posweigh'],
            #     reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss=FocalLoss(
                threshold=0.5,
                alpha=initial.config['detect.focal.loss.multiduration.at'],
                gamma=initial.config['detect.focal.loss.gamma'],
                reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            metrics=Macro_Binary_Fb_Evaluation(
                size=88,
                thresholds=initial.config['detect.train.fn.threshold'],
                beta=initial.config['detect.train.fn.beta']))
    
    return model


def main():
    g = Train_CNN_DataGenerator(io='test', output='common_start')
    g = g()
    t = next(g)
    t = next(g)
    t = next(g)
    t = next(g)
    return 0


if __name__ == '__main__':
  main()
