import os,sys
import wave
import math
import pretty_midi as pm
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
# import src.component.pre_spec_transform as spec_transform
# import src.component.util as util


# fix problem, may increase the RAM load: https://github.com/craffel/pretty-midi/issues/112
pm.pretty_midi.MAX_TICK = 1e10


def midi_trans_start(midi_path: str, audio_path: str) -> np.ndarray:
    """
    midi_trans_start函数
    功能: 将midi文件转为矩阵, 和spectrum同步
    输入: midi_path曲谱(MIDI ver1)文件路径, audio_path对应音频文件路径
    输出: duration*88的ndarray
        说明: 1代表starting
    """

    # 计算samples数量
    audio = wave.open(audio_path, mode='rb')
    audio_length = audio.getnframes() / audio.getframerate()
    # 计算frames数量
    # 优先考虑已计算的spec
    if os.path.isfile(audio_path[:-3]+ 'npy'):
        spec_length = np.load(audio_path[:-3]+ 'npy', allow_pickle=False)
        spec_length = spec_length.shape[0]
    else:
        spec_length = 1 + \
            ((math.ceil(audio.getnframes()*initial.config['spec.cqt.sampling']/audio.getframerate()) -
             initial.config['spec.cqt.frame_length'])//initial.config['spec.cqt.hop_length'])
    # 关闭文件
    audio.close()

    # 使用txt文件
    # 准备矩阵
    notation = np.zeros((spec_length, 88), dtype=np.float32)
    # 使用midi路径得到txt路径
    txt_path = midi_path[:-3] + 'txt'
    # 读入矩阵
    sheets = pd.read_table(txt_path, header=0, encoding='utf-8')
    sheets = np.array(sheets)
    # 按音频文件的时间片记录
    for index in range(sheets.shape[0]):
        # 起始时间记录, remap到琴键
        start_time = int(round(sheets[index, 0]*spec_length/audio_length))
        start_time = min(start_time, spec_length-1)
        pitch = int(sheets[index, 2] - 21)
        notation[start_time, pitch] = 1
    
    '''
    # 使用MIDI文件
    # 准备矩阵
    notation = np.zeros((spec_length, 88), dtype=np.float32)
    # 读入MIDI
    midi = pm.PrettyMIDI(midi_path)
    # 检查音轨
    if len(midi.instruments) < 1:
        return notation
    # 按音频文件的时间片记录
    for instrument in midi.instruments:
        for note in instrument.notes:
            # remap到琴键
            pitch = note.pitch - 21

            # 起始时间记录
            start_time = int(round(note.start*spec_length/audio_length))
            start_time = min(start_time, spec_length-1)
            notation[start_time, pitch] = 1
    '''
    
    # 添加ADSR
    for index in range(spec_length):
        for pitch in range(88):
            if notation[index, pitch] == 1:
                if index > 0 and notation[index-1, pitch] == 0:
                    notation[index-1, pitch] = 0.5
                if index < spec_length-1 and notation[index+1, pitch] == 0:
                    notation[index+1, pitch] = 0.5

    return notation


def midi_trans_common(midi_path: str, audio_path: str) -> np.ndarray:
    """
    midi_trans_common函数
    功能: 将midi文件转为矩阵, 和spectrum同步
        注意会将相隔一帧的起始时间合并!
    输入: midi_path曲谱(MIDI ver1)文件路径, audio_path对应音频文件路径
    输出: duration*1的ndarray
        说明: 1代表starting
    """

    # 计算samples数量
    audio = wave.open(audio_path, mode='rb')
    audio_length = audio.getnframes() / audio.getframerate()
    # 计算frames数量
    # 优先考虑已计算的spec
    if os.path.isfile(audio_path[:-3]+ 'npy'):
        spec_length = np.load(audio_path[:-3]+ 'npy', allow_pickle=False)
        spec_length = spec_length.shape[0]
    else:
        spec_length = 1 + \
            ((math.ceil(audio.getnframes()*initial.config['spec.cqt.sampling']/audio.getframerate()) -
             initial.config['spec.cqt.frame_length'])//initial.config['spec.cqt.hop_length'])
    # 关闭文件
    audio.close()

    # 使用txt文件
    # 准备矩阵
    notation = np.zeros((spec_length, 1), dtype=np.float32)
    # 使用midi路径得到txt路径
    txt_path = midi_path[:-3] + 'txt'
    # 读入矩阵
    sheets = pd.read_table(txt_path, header=0, encoding='utf-8')
    sheets = np.array(sheets)
    # 按音频文件的时间片记录
    for index in range(sheets.shape[0]):
        # 起始时间记录, remap到琴键
        start_time = int(round(sheets[index, 0]*spec_length/audio_length))
        start_time = min(start_time, spec_length-1)
        notation[start_time, 0] = 1

    '''
    # 使用MIDI文件
    # 准备矩阵
    notation = np.zeros((spec_length, 1), dtype=np.float32)
    # 读入MIDI
    midi = pm.PrettyMIDI(midi_path)
    # 检查音轨
    if len(midi.instruments) < 1:
        return notation
    # 按音频文件的时间片记录
    for instrument in midi.instruments:
        for note in instrument.notes:
            # 起始时间记录
            start_time = int(round(note.start*spec_length/audio_length))
            start_time = min(start_time, spec_length-1)
            notation[start_time, 0] = 1
    '''
    
    # 合并相邻起始事件
    index = 0
    onsets_tolerance = 1 # 最大合并距离参数!
    while index < spec_length:
        # 若出现起始时间
        if notation[index, 0] == 1:
            # 设置第一个起始时间标记
            onsets_start = index
            onsets_end = index
            # 继续查找起始时间, 此时设置忽略的间隔
            while onsets_end < spec_length:
                onsets_continue = False
                onsets_end_offset = 0
                while onsets_end_offset < onsets_tolerance:
                    # 注意此处相当于判断不大于(下面无条件+1)
                    onsets_end_offset += 1
                    if onsets_end+onsets_end_offset < spec_length and \
                       notation[onsets_end+onsets_end_offset, 0] == 1:
                        onsets_continue = True
                        break
                # 检查相邻起始事件
                if onsets_continue:
                    # 若发现了相邻起始事件, 可以继续查找
                    onsets_end += onsets_end_offset
                    continue
                # 否则没有进一步的相邻起始事件, 停止查找
                else:
                    break
            # 查找结束, 按照查找结果写入原数组中
            # 注意end要+1, 因为是左闭右闭
            notation[onsets_start:onsets_end+1, 0] = 0
            onsets_mid = (onsets_start+onsets_end)//2 # 去尾法, 偏向起始
            notation[onsets_mid, 0] = 1
            # 递增index
            index = onsets_end+onsets_tolerance+1
        else:
            # 没有出现起始时间, 正常递增
            index += 1
    
    # 添加ADSR(没有用)
    # for index in range(spec_length):
    #     if notation[index, 0] == 1:
    #         if index > 0 and notation[index-1, 0] == 0:
    #             notation[index-1, 0] = 0.5
    #         if index < spec_length-1 and notation[index+1, 0] == 0:
    #             notation[index+1, 0] = 0.5

    return notation


def midi_trans_duration(midi_path: str, audio_path: str) -> np.ndarray:
    """
    midi_trans_duration函数
    功能: 将midi文件转为矩阵, 和spectrum同步
    输入: midi_path曲谱(MIDI ver1)文件路径, audio_path对应音频文件路径
    输出: duration*88的ndarray
        说明: 1代表note duration
    """

    # 计算samples数量
    audio = wave.open(audio_path, mode='rb')
    audio_length = audio.getnframes() / audio.getframerate()
    # 计算frames数量
    # 优先考虑已计算的spec
    if os.path.isfile(audio_path[:-3]+ 'npy'):
        spec_length = np.load(audio_path[:-3]+ 'npy', allow_pickle=False)
        spec_length = spec_length.shape[0]
    else:
        spec_length = 1 + \
            ((math.ceil(audio.getnframes()*initial.config['spec.cqt.sampling']/audio.getframerate()) -
             initial.config['spec.cqt.frame_length'])//initial.config['spec.cqt.hop_length'])
    # 关闭文件
    audio.close()

    # 使用txt文件
    # 准备矩阵
    notation = np.zeros((spec_length, 88), dtype=np.float32)
    # 使用midi路径得到txt路径
    txt_path = midi_path[:-3] + 'txt'
    # 读入矩阵
    sheets = pd.read_table(txt_path, header=0, encoding='utf-8')
    sheets = np.array(sheets)
    # 按音频文件的时间片记录
    for index in range(sheets.shape[0]):
        # 持续时间记录, remap到琴键
        start_time = int(round(sheets[index, 0]*spec_length/audio_length))
        start_time = min(start_time, spec_length-1)
        ending_time = int(round(sheets[index, 1]*spec_length/audio_length))
        ending_time = min(ending_time, spec_length-1)
        pitch = int(sheets[index, 2] - 21)
        notation[start_time:ending_time, pitch] = 1

    '''
    # 使用MIDI文件
    # 准备矩阵
    notation = np.zeros((spec_length, 88), dtype=np.float32)
    # 读入MIDI
    midi = pm.PrettyMIDI(midi_path)
    # 检查音轨
    if len(midi.instruments) < 1:
        return notation
    # 按音频文件的时间片记录
    for instrument in midi.instruments:
        for note in instrument.notes:
            pitch = note.pitch - 21

            # 持续时间记录
            start_time = int(round(note.start*spec_length/audio_length))
            start_time = min(start_time, spec_length-1)
            ending_time = int(round(note.end*spec_length/audio_length))
            ending_time = min(ending_time, spec_length-1)
            notation[start_time:ending_time, pitch] = 1
    '''

    return notation


def midi_trans(midi_path: str, audio_path: str) -> np.ndarray:
    """
    midi_trans函数
    功能: 将midi文件转为矩阵, 和spectrum同步
    输入: midi_path曲谱(MIDI ver1)文件路径, audio_path对应音频文件路径
    输出: duration*88的ndarray
        说明: 1代表start, -1代表end, 0代表不变化(持续/休止)
    """

    # 计算samples数量
    audio = wave.open(audio_path, mode='rb')
    audio_length = audio.getnframes() / audio.getframerate()
    # 计算frames数量
    # 优先考虑已计算的spec
    if os.path.isfile(audio_path[:-3]+ 'npy'):
        spec_length = np.load(audio_path[:-3]+ 'npy', allow_pickle=False)
        spec_length = spec_length.shape[0]
    else:
        spec_length = 1 + \
            ((math.ceil(audio.getnframes()*initial.config['spec.cqt.sampling']/audio.getframerate()) -
             initial.config['spec.cqt.frame_length'])//initial.config['spec.cqt.hop_length'])
    # 关闭文件
    audio.close()

    # 准备矩阵
    notation = np.zeros((spec_length, 88), dtype=np.flaot32)

    # 读入MIDI
    midi = pm.PrettyMIDI(midi_path)

    # 检查音轨
    if len(midi.instruments) < 1:
        return notation

    # 按音频文件的时间片记录
    for instrument in midi.instruments:
        for note in instrument.notes:
            pitch = note.pitch - 21

            # 起始时间记录
            start_time = int(round(note.start*spec_length/audio_length))
            start_time = min(start_time, spec_length-1)
            notation[start_time, pitch] = 1

            # 终止时间记录
            ending_time = int(round(note.end*spec_length/audio_length))
            ending_time = min(ending_time, spec_length-1)
            notation[ending_time, pitch] = 1

    return notation


def main():
    '''
    func = spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                      bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                      fmin=initial.config['spec.cqt.fmin'],
                                      frame_length=initial.config['spec.cqt.frame_length'],
                                      hop_length=initial.config['spec.cqt.hop_length'],
                                      window=initial.config['spec.cqt.window'])
    spec = func('data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')

    sheet = midi_trans_duration(
        'data\MAPS_MUS-bk_xmas1_ENSTDkAm.mid', 'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
    util.plot_dual_transform(spec[0:1024, :, 0], sheet[0:1024, :])
    '''
    return 0


if __name__ == '__main__':
  main()
