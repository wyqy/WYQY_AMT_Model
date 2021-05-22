import os, sys
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.pre_spec_transform as spec_transform
import src.component.pre_sheet_transform as sheet_transform


def spec_preprocess():
    '''
    spec_preprocess函数
    功能: 将预定义的数据集, 测试集在原位置生成npy格式的频谱文件, 可自动跳过已有文件(注意!)
    输入:
    输出:
    '''
    # 初始化对象
    trans_cqt = spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                           bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                           fmin=initial.config['spec.cqt.fmin'],
                                           frame_length=initial.config['spec.cqt.frame_length'],
                                           hop_length=initial.config['spec.cqt.hop_length'],
                                           window=initial.config['spec.cqt.window'])

    # 训练集
    data_length = initial.trainsets.shape[0]
    data_offset = 0
    for row in initial.trainsets.iterrows():
        # 音频
        audio_path = row[1]['audio_filename']
        numpy_path = audio_path[:-3] + 'npy'

        # 检查文件是否已存在
        if not os.path.isfile(numpy_path):
            spec = trans_cqt(audio_path)
            np.save(numpy_path, spec, allow_pickle=False)
        
        # 输出进度
        data_offset += 1
        print('trainsets percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')

    # 测试集
    data_length = initial.testsets.shape[0]
    data_offset = 0
    for row in initial.testsets.iterrows():
        audio_path = row[1]['audio_filename']
        numpy_path = audio_path[:-3] + 'npy'

        # 检查文件是否已存在
        if not os.path.isfile(numpy_path):
            spec = trans_cqt(audio_path)
            np.save(numpy_path, spec, allow_pickle=False)
        # 输出进度
        data_offset += 1
        print('testests percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')


def sheet_preprocess():
    '''
    sheet_preprocess函数
    功能: 将预定义的数据集, 测试集在原位置生成npy格式的曲谱文件, 可自动跳过已有文件(注意!)
    输入:
    输出:
    '''
    # 训练集
    data_length = initial.trainsets.shape[0]
    data_offset = 0
    for row in initial.trainsets.iterrows():
        audio_path = row[1]['audio_filename']
        midi_path = row[1]['midi_filename']

        # multitone_start
        numpy_path = midi_path[:-4] + '_msmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_start(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # common_start
        numpy_path = midi_path[:-4] + '_csmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_common(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # multitone_duration
        numpy_path = midi_path[:-4] + '_mdmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_duration(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # 输出进度
        data_offset += 1
        print('trainsets percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')

    # 测试集
    data_length = initial.testsets.shape[0]
    data_offset = 0
    for row in initial.testsets.iterrows():
        audio_path = row[1]['audio_filename']
        midi_path = row[1]['midi_filename']

        # multitone_start
        numpy_path = midi_path[:-4] + '_msmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_start(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # common_start
        numpy_path = midi_path[:-4] + '_csmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_common(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # multitone_duration
        numpy_path = midi_path[:-4] + '_mdmidi.npy'
        if not os.path.isfile(numpy_path):
            sheet = sheet_transform.midi_trans_duration(midi_path, audio_path)
            np.save(numpy_path, sheet, allow_pickle=False)
        # 输出进度
        data_offset += 1
        print('testests percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')


def delete_spec_preprocess():
    '''
    delete_spec_preprocess函数
    功能: 清除npy曲谱文件
    输入:
    输出:
    '''
    # 训练集
    data_length = initial.trainsets.shape[0]
    data_offset = 0
    for row in initial.trainsets.iterrows():
        # 音频
        audio_path = row[1]['audio_filename']
        numpy_path = audio_path[:-3] + 'npy'
        
        # 检查文件是否已存在
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        
        # 输出进度
        data_offset += 1
        print('trainsets percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')

    # 测试集
    data_length = initial.testsets.shape[0]
    data_offset = 0
    for row in initial.testsets.iterrows():
        audio_path = row[1]['audio_filename']
        numpy_path = audio_path[:-3] + 'npy'

       # 检查文件是否已存在
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        # 输出进度
        data_offset += 1
        print('testests percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')


def delete_sheet_preprocess():
    '''
    delete_sheet_preprocess函数
    功能: 清除npy频谱文件
    输入:
    输出:
    '''
    # 训练集
    data_length = initial.trainsets.shape[0]
    data_offset = 0
    for row in initial.trainsets.iterrows():
        midi_path = row[1]['midi_filename']

        # multitone_start
        numpy_path = midi_path[:-4] + '_msmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        # common_start
        numpy_path = midi_path[:-4] + '_csmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        # multitone_duration
        numpy_path = midi_path[:-4] + '_mdmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        
        # 输出进度
        data_offset += 1
        print('trainsets percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')

    # 测试集
    data_length = initial.testsets.shape[0]
    data_offset = 0
    for row in initial.testsets.iterrows():
        midi_path = row[1]['midi_filename']

        # multitone_start
        numpy_path = midi_path[:-4] + '_msmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        # common_start
        numpy_path = midi_path[:-4] + '_csmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        # multitone_duration
        numpy_path = midi_path[:-4] + '_mdmidi.npy'
        if os.path.isfile(numpy_path):
            os.remove(numpy_path)
        
        # 输出进度
        data_offset += 1
        print('testests percentage: %s [%d/%d]' %
              (str(round(1.0*data_offset/data_length*100, 2)),
               data_offset, data_length), end='\r')

    # 还原windows下默认print
    print('', end='\n')


def main():
    spec_preprocess()
    return 0


if __name__ == '__main__':
  main()
