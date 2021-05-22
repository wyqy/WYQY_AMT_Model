import os, sys
import gc
from typing import Generator
import multiprocessing
from multiprocessing import connection
import mido
# import pandas as pd
import numpy as np
import numba as nb
import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.pre_spec_transform as spec_transform
import src.component.model_cnn_build as model_cnn_build
import src.component.model_cnn_common as model_cnn_common


# 定义一些MIDI变换用常数
# 感谢: https://www.gamedev.net/forums/topic/535653-convert-midi-deltatime-to-milliseconds/
MID_BPM = 75
MID_DIVISION = mido.midifiles.midifiles.DEFAULT_TICKS_PER_BEAT


class Transcription_Model():
    '''
    Transcription_Model类
    用于根据模型预测转谱! 不生成MIDI!
    '''

    def __init__(self, multitone_start_path, common_start_path, multitone_duration_path):
        # 参数导入
        self.batch_size = initial.config['detect.predict.batch.size']
        self.slice_threshold = self.batch_size // 2

        # 路径导入
        self.multitone_start_path = multitone_start_path
        self.common_start_path = common_start_path
        self.multitone_duration_path = multitone_duration_path

        # 限制内存增长
        # 参考: https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
        # 参考: https://github.com/keras-team/keras/issues/13118
        # 这里使用多进程: https://github.com/tensorflow/tensorflow/issues/36465

        # 生成trans_spec对象
        self.trans_cqt = spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                                    bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                                    fmin=initial.config['spec.cqt.fmin'],
                                                    frame_length=initial.config['spec.cqt.frame_length'],
                                                    hop_length=initial.config['spec.cqt.hop_length'],
                                                    window=initial.config['spec.cqt.window'])

    def notes_plot(self, audio_path):
        """
        功能: 实战!
        """
        # audio -> spec
        self.spec = self.trans_cqt(audio_path)

        # spec -> pred
        # 分批predict
        # multitone_start -> (m_frames, 88)
        # common_start -> (c_frames, 1)
        # multone_duration -> (m_frames, 88)
        multone_start, common_start, multone_duration = self.predict_proc_wrapper()

        # pred -> list-like notes ndarray
        # 生成list-like notes ndarray
        piano_roll, notes = self.alignment(multone_start, common_start, multone_duration,
                                           initial.config['detect.predict.mulstart.threshold'],
                                           initial.config['detect.predict.comstart.threshold'],
                                           initial.config['detect.predict.mulduration.threshold'],
                                           initial.config['detect.tone.slice'],
                                           initial.config['detect.start.common.slice'],
                                           initial.config['spec.cqt.sampling'],
                                           initial.config['spec.cqt.frame_length'],
                                           initial.config['spec.cqt.hop_length'])

        return (piano_roll, notes)

    def sheet_plot(self, audio_path):
        '''
        功能: 输出模型生成的原始预测数据!
        '''
        # audio -> spec
        self.spec = self.trans_cqt(audio_path)

        # spec -> pred
        # 分批predict
        # multitone_start -> (m_frames, 88)
        # common_start -> (c_frames, 1)
        # multone_duration -> (m_frames, 88)
        multone_start, common_start, multone_duration = self.predict_proc_wrapper()

        # 输出
        return (multone_start, common_start, multone_duration)

    def predict_proc_wrapper(self):
        """
        功能: 模型预测的多进程包装函数
        """
        # 创建管道
        recvPipe, sendPipe = multiprocessing.Pipe(duplex=False)

        proc = multiprocessing.Process(target=self.predict,
                                       kwargs={'sendPipe': sendPipe})
        # 启动进程
        proc.start()
        # 阻塞接收方法, 保证进程完成
        multone_start = recvPipe.recv()
        common_start = recvPipe.recv()
        multone_duration = recvPipe.recv()
        proc.join()  # 等待子进程
        # 关闭进程
        proc.terminate()
        
        # 返回结果
        return (multone_start, common_start, multone_duration)

    def predict(self, sendPipe: connection.Connection, *args, **kwargs):
        """
        功能: 分批导入模型预测!
        """
        # 限制内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Multitone Start
        self.slice_size = initial.config['detect.tone.slice']
        model = self.initModel('multitone_start')
        # 输出格式: (samples, slice_size, n_bins, 2)
        predictsets = tf.data.Dataset.from_generator(
            generator=self.predict_data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(None,
                           self.slice_size,
                           initial.config['spec.cqt.n_bins'],
                           2),
                    dtype=tf.float32)))
        # 送入模型预测
        pred = model.predict(predictsets, verbose=1)
        # 转为sigmoid
        pred = 1/(1+np.exp(-pred))
        # 返回预测的ndarray
        sendPipe.send(pred)

        # Common Start
        self.slice_size = initial.config['detect.start.common.slice']
        model = self.initModel('common_start')
        # 输出格式: (samples, slice_size, n_bins, 2)
        predictsets = tf.data.Dataset.from_generator(
            generator=self.predict_data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(None,
                           self.slice_size,
                           initial.config['spec.cqt.n_bins'],
                           2),
                    dtype=tf.float32)))
        # 回收内存
        gc.collect()
        # 送入模型预测
        pred = model.predict(predictsets, verbose=1)
        # 转为sigmoid
        pred = 1/(1+np.exp(-pred))
        # 返回预测的ndarray
        sendPipe.send(pred)


        # Multitone Duration
        self.slice_size = initial.config['detect.tone.slice']
        model = self.initModel('multitone_duration')
        # 输出格式: (samples, slice_size, n_bins, 2)
        predictsets = tf.data.Dataset.from_generator(
            generator=self.predict_data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(None,
                           self.slice_size,
                           initial.config['spec.cqt.n_bins'],
                           2),
                    dtype=tf.float32)))
        # 回收内存
        gc.collect()
        # 送入模型预测
        pred = model.predict(predictsets, verbose=1)
        # 转为sigmoid
        pred = 1/(1+np.exp(-pred))
        # 返回预测的ndarray
        sendPipe.send(pred)

        # 对应主进程, 确保管道关闭
        # 参考: https://www.cnblogs.com/pyxiaomangshe/p/7797359.html
        sendPipe.close()

    def initModel(self, model_type) -> keras.Model:
        '''
        功能: 初始化模型
        '''
        if model_type == 'multitone_start':
            model = model_cnn_build.detect_multitone_model_cnn_build()
            model = model_cnn_common.model_compile('multitone_start', model)
            model.load_weights(self.multitone_start_path, by_name=False)

        if model_type == 'common_start':
            model = model_cnn_build.detect_start_common_model_cnn_build()
            model = model_cnn_common.model_compile('common_start', model)
            model.load_weights(self.common_start_path, by_name=False)

        if model_type == 'multitone_duration':
            model = model_cnn_build.detect_multitone_model_cnn_build()
            model = model_cnn_common.model_compile('multitone_duration', model)
            model.load_weights(self.multitone_duration_path, by_name=False)

        return model

    def predict_data_generator(self) -> Generator:
        # 音频长度信息
        spec_length = self.spec.shape[0]

        # 初始化变量
        slice_offset = 0
        pred_offset = 0

        # 生成切片
        while not slice_offset > spec_length-self.slice_threshold:
            # 计算切片长度
            slice_end = min(
                slice_offset+self.batch_size, spec_length)
            # 多取一些
            if spec_length-1-slice_offset-self.batch_size < self.slice_threshold:
                slice_end = spec_length

            # spec切片
            spec_slice = self.spec[slice_offset:slice_end, :, :]

            # spec滑窗切片
            # 输出格式: (samples, slice_size, n_bins, 2)
            spec_slice = self.data_sample(
                spec_slice, initial.config['spec.cqt.n_bins'],
                self.slice_size)
            # 滑窗长度
            slice_frame = 1 + ((slice_end-slice_offset) - self.slice_size)

            # 输出
            yield spec_slice

            # 更新slice_offset
            slice_offset = slice_end-self.slice_size+1
            pred_offset += slice_frame

        # 结束返回(抛出异常)
        return

    @staticmethod
    @nb.jit(nopython=True)
    def alignment(mul_start, com_start, mul_duration,
                  mulstart_th, comstart_th, mulduration_th,
                  mul_slicelength, com_slicelength,
                  sample_rate, spec_frame_length, spec_hop_length):
        '''
        功能: 生成最终dict!
        考虑到多音调和公共起始的时间片不等长! (要么补零要么)保存difference!
        '''
        # 计算长度
        # 要求两个slicelength必须为奇数!
        # com_start比multitone前后各多difference帧, 共2*difference帧!
        comstart_framesize = com_start.shape[0]
        multitione_framesize = mul_start.shape[0]
        framesize_difference = (mul_slicelength-com_slicelength) // 2
        # 计算模型检测用滑窗的一半(去尾), 用于还原spec frame位置
        # multitone_pred比spec前后各多multitione_half帧, 共2*multitone_halframe帧!
        multitone_halframe = mul_slicelength // 2
        # 计算声谱变换用滑窗的一半(去尾), 用于还原audio frame位置
        # spec比audio前后多spectrans_halframe?帧, 因为为偶数, 故有一帧误差!
        spectrans_halframe = spec_frame_length // 2

        # 初始化变量
        # 注意com的frames多, 为了对齐, 该index范围为[difference, comstart_slicesize-difference)
        comstart_start = framesize_difference
        comstart_max = 0
        comstart_end = 0
        # 初始化中间变量
        # 中间变量的2代表开始, 1代表持续!
        # 按短的计算, 也就是multitone!
        piano_roll = np.zeros(
            (multitione_framesize, 88), dtype=np.int8)

        # 依次按公共起始时间对齐多音调起始时间
        # mul_index = com_index + difference
        while comstart_start < comstart_framesize-framesize_difference:
            # 检测公共起始点, 记录阈值起始点
            if com_start[comstart_start] > comstart_th:
                # 检测阈值最大点和终止点
                # 初始化max参数
                comstart_max = comstart_start
                for comstart_index in range(comstart_start+1, comstart_framesize-framesize_difference):
                    # 检测阈值终止点
                    if com_start[comstart_index] > comstart_th:
                        # 检测阈值最大点
                        if com_start[comstart_index] >= com_start[comstart_max]:
                            comstart_max = comstart_index
                    else:
                        # 写入阈值终止点, 保证了左开右闭
                        comstart_end = comstart_index
                        break
                # 遍历[comstart_start, comstart_end)之间的多音调起始点
                # 检测在此期间内多音调是否有起始
                for pitch_index in range(88):
                    if np.max(mul_start[
                        # !!!!ATTENTION!!!!: 注意是减不是加! 下同!
                        comstart_start-framesize_difference:comstart_end-framesize_difference,
                        pitch_index] > mulstart_th):
                        # 若有则记录起始点为公共最大值处
                        # 初始化参数
                        duration_end = comstart_max - framesize_difference
                        # 从mulduration的[comstart_max, comstart_end)找起始点, 检测持续时间
                        for duration_start in range(comstart_max-framesize_difference, comstart_end-framesize_difference):
                            if mul_duration[duration_start, pitch_index] > mulduration_th:
                                # 若检测到持续时间则一直检测下去
                                duration_offset = duration_start
                                while duration_offset < multitione_framesize and \
                                      mul_duration[duration_offset, pitch_index] > mulduration_th:
                                    duration_offset += 1
                                # 检测完毕, break, 最终区间: [duration_start(或称comstart_max), duration_end)
                                duration_end = duration_offset
                                break
                        
                        # 如果有持续时间才记录音符, 单起始时间不记录
                        if duration_end > comstart_max-framesize_difference:
                            # 记录note起始
                            piano_roll[comstart_max-framesize_difference, pitch_index] = 2
                            # 记录note持续
                            piano_roll[comstart_max-framesize_difference+1:duration_end, pitch_index] = 1

                # 直接跳过检测过的公共起始时间
                comstart_start = comstart_end
            else:
                # 否则没有检测到common中的起始, iter自然增加
                comstart_start += 1

        # piano-roll转notes numpy
        # 初始化输出变量
        # shape = (multitione_framesize*88, channel/3), channel的含义: onset, offset, pitch
        # for songs last 40min, it takes memory for 208M
        # 最后再切片
        notes = np.zeros((multitione_framesize*88, 3), dtype=np.float32)
        notes_index = 0

        # 依次计入, 先计pitch再计frames
        for pitch_index in range(88):
            # 初始化index变量
            roll_start = 0
            while roll_start < multitione_framesize:
                # 检测是否有音符起始信息!
                if piano_roll[roll_start, pitch_index] == 2:
                    # 如果有音符其实信息则继续检测持续时间
                    # 初始化index变量
                    roll_end = roll_start + 1
                    # 检测终止时间
                    while roll_end < multitione_framesize and \
                        piano_roll[roll_end, pitch_index] == 1:
                        # 此处保证了必然是左开右闭
                        roll_end += 1

                    # 将音符起始结束时间转换为真实时间!
                    # 还原滑窗损失
                    frame_start = roll_start + multitone_halframe
                    frame_end = roll_end + multitone_halframe
                    # 还原spec变换损失(一帧误差!)
                    frame_start = frame_start * spec_hop_length + spectrans_halframe
                    frame_end = frame_end * spec_hop_length + spectrans_halframe
                    # 计算真实时间(舍入误差!)
                    time_start = frame_start / sample_rate
                    time_end = frame_end / sample_rate

                    # 将音符添加到notes中!
                    notes[notes_index, :] = [time_start, time_end, pitch_index]
                    notes_index += 1

                    # 直接跳过检测过的帧
                    roll_start = roll_end
                else:
                    # 否则没有检测到piano-roll中的起始, iter自然增加
                    roll_start += 1

        # 输出piano-roll和list-like notes ndarray!
        return (piano_roll, notes[0:notes_index, :])

    @staticmethod
    @nb.jit(nopython=True)
    def data_sample(spec, n_bins, slice_size):
        # 新建数组
        ret_size = 1 + (spec.shape[0] - slice_size)
        ret = np.zeros((ret_size, slice_size, n_bins, 2), dtype=np.float32)

        # 顺序切片
        for index in range(ret_size):
            ret[index, :, :, :] = spec[index:index+slice_size, :, :]

        return ret


def MIDI_Maker(notes: np.ndarray, midi_path=''):
    '''
    MIDI_Maker函数
    功能: 生成MIDI文件
    参见: https://www.jianshu.com/p/6c495b51a40c
    '''
    # 生成MIDI文件对象
    # type_1: 同步多音轨, DEFAULT_TICKS_PER_BEAT=480
    mid = mido.MidiFile(type=1, ticks_per_beat=MID_DIVISION)
    # 创建音轨并加入到MIDI文件
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # 创建控制器事件并加入到MIDI文件
    # program=?代表钢琴
    track.append(mido.Message('program_change', program=0, time=0))
    # 创建节拍事件
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    # 创建节奏事件
    tempo = mido.bpm2tempo(MID_BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # 创建调式事件
    # meta_tone = mido.MetaMessage('key_signature', key='C')

    # 在开头全写note_off
    for index in range(88):
        track.append(mido.Message('note_off', note=index+21,
                                  velocity=64, time=0))
    # 将notes排序
    # 建立数组(messages, 3) => message_type => 0 for off, 1 for on; pitch; time(for lexsort)
    # 测试时pitch不用加21!
    sorted_notes = np.zeros((notes.shape[0]*2, 3), dtype=np.float32)
    sorted_notes[0:notes.shape[0], 0] = 1
    sorted_notes[0:notes.shape[0], 1:] = notes[:, [2, 0]]
    sorted_notes[notes.shape[0]:2*notes.shape[0], 1:] = notes[:, [2, 1]]
    # 数组排序, 按最后一行时间轴排序
    sorted_notes = sorted_notes[np.lexsort(sorted_notes.T)]
    
    # 根据array写音符
    # 记录上一个音符结束时间
    last_msg_time = 0
    for index in range(sorted_notes.shape[0]):
        # 计算MIDI的pitch(测试时不用+21)
        pitch = int(sorted_notes[index, 1]+21)
        # pitch = int(sorted_notes[index, 1])
        # 将真实事件变换为delta_time
        if sorted_notes[index, 0] == 1:
            time = sorted_notes[index, 2] - last_msg_time
            time = int(round(time*MID_BPM*MID_DIVISION/(60.0)))
            track.append(mido.Message('note_on', note=pitch,
                                      velocity=64, time=time))
        if sorted_notes[index, 0] == 0:
            time = sorted_notes[index, 2] - last_msg_time
            time = int(round(time*MID_BPM*MID_DIVISION/(60.0)))
            track.append(mido.Message('note_off', note=pitch,
                                      velocity=64, time=time))
        last_msg_time = sorted_notes[index, 2]

    # 保存文件, 同时返回mid对象
    if not midi_path == '':
        mid.save(midi_path)
    return mid


def main():
    # sheets = pd.read_table('data\MAPS_MUS-bk_xmas1_ENSTDkAm.txt', header=0, encoding='utf-8')
    # sheets = np.array(sheets)
    # mid = MIDI_Maker(sheets, 'data\MAPS_MUS-bk_xmas1_ENSTDkAm_make.mid')

    predict_test = Transcription_Model(
        multitone_start_path=initial.config['detect.model.multistart.weight'],
        common_start_path=initial.config['detect.model.commonstart.weight'],
        multitone_duration_path=initial.config['detect.model.multiduration.weight'])
    piano_roll, notes = predict_test.notes_plot(
        'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')

    return 0


if __name__ == '__main__':
  main()
