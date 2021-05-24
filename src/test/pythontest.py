import time
import wave
import pandas as pd
import numpy as np
import librosa as rs
import pretty_midi as pm
import mido
import importlib
import mir_eval

import tensorflow as tf

import src.component.initial as initial
import src.component.util as util
import src.component.pre_spec_transform as spec_transform
import src.component.pre_sheet_transform as sheet_transform

import src.component.model_cnn_common as model_cnn_common
import src.component.model_cnn_inference as model_cnn_inference


pm.pretty_midi.MAX_TICK = 1e10


# 重新加载模块
importlib.reload(initial)
importlib.reload(util)
importlib.reload(spec_transform)
importlib.reload(sheet_transform)
importlib.reload(model_cnn_common)
importlib.reload(model_cnn_inference)

# numpy实验
a = np.array(([1, 2], [3, 4], [5, 6]))
b = np.array(([7, 8], [9, 10]))
c = np.array(([11], [12]))
zzz = np.dstack((a, b, c))
zzz[:, :, 0]

# CQT变换长度实验
cqt_filter_fft = rs.constantq.__cqt_filter_fft
_, frame_length, _ = cqt_filter_fft(44100, 27.5, 356, 48, 1, 1, 0.01, 512, 'hann')

a = wave.open('data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
length = 1+(a.getnframes()-512*4)//512
print(length)
a.close()
# librosa实验
time_start = time.time()
y, sr = rs.core.load('data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav',
                     sr=44100, mono=False)
spec = np.zeros((2, 356, 33195), dtype=np.float32)

time_start = time.time()
x = rs.core.cqt(
    y[0, :], sr=44100,
    hop_length=512*4,
    fmin=27.5, n_bins=356,
    bins_per_octave=48, tuning=0.0,
    filter_scale=1,
    norm=1, sparsity=0.01,
    window='hann')
time_end = time.time()
print(time_end - time_start)

# CQT最短长度实验
# 实验证明最短长度和hop_length不成线性关系?
x = rs.core.hybrid_cqt(
    y[0, 500:512*4*4*16+500], sr=44100,
    hop_length=512*4,
    fmin=27.5, n_bins=356,
    bins_per_octave=48, tuning=0.0,
    filter_scale=1,
    norm=1, sparsity=0.01,
    window='hann')
z = np.zeros((256, 64))
z[:, 2:61+2] = rs.feature.melspectrogram(
    y[0, 500:512*4*4*16+500], sr=44100,
    n_fft=512*4*4,
    hop_length=512*4,
    win_length=512*4*4,
    center=False,
    n_mels=256, fmin=30.0, fmax=8000.0)
x = rs.core.amplitude_to_db(x, ref=np.max, top_db=120.0)
Min = np.min(x)
Max = np.max(x)
x = np.transpose((x - Min) / (Max - Min))
z = rs.core.amplitude_to_db(z, ref=np.max, top_db=120.0)
Min = np.min(z)
Max = np.max(z)
z = np.transpose((z - Min) / (Max - Min))
util.plot_dual_transform(x, z)

spec[0, :, :] = rs.core.hybrid_cqt(
    y[0, :], sr=44100,
    hop_length=512,
    fmin=27.5, n_bins=356,
    bins_per_octave=48, tuning=0.0,
    filter_scale=1,
    norm=1, sparsity=0.01,
    window='hann')[:, 2:33197]
spec[0, :, :] = rs.core.hybrid_cqt(
    y[0, :], sr=44100,
    hop_length=512,
    fmin=27.5, n_bins=356,
    bins_per_octave=48, tuning=0.0,
    filter_scale=1,
    norm=1, sparsity=0.01,
    window='hann')[:, 2:33197]
time_end = time.time()
print(spec.shape)
print(spec.dtype)
print(time_end - time_start)

# 音频对齐实验
spec_cqt = spec_transform.rs_spec_cqt(n_bins=356,
                                      bins_per_octave=48,
                                      fmin=27.5,
                                      frame_length=2048,
                                      hop_length=512,
                                      window='hann')
spec = spec_cqt(
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
print(spec.shape)
sheet = sheet_transform.midi_trans_common(
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.mid',
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
sheet = np.where(sheet > 0.9, 1, 0)
# 画图
util.plot_dual_transform(spec[-5000:-4000, :, 0], sheet[-5000:-4000, :])
util.plot_single_transform(spec[0, :, :])
# 切片测试
# 滑窗长度
slice_frame = 1 + \
    (2000 - 9 // 1)
# CQT滑窗
# 内存不安全!
spec_slice = np.lib.stride_tricks.sliding_window_view(
    spec[0:2000, :, :],
    window_shape=(9, 356),
    axis=(0, 1),
    writeable=False)
print(spec_slice.shape)
tenspec = tf.convert_to_tensor(spec_slice[:, 0, :, :, :], dtype=tf.float32)
util.plot_single_transform(tenspec[256, 0, :, :])
util.plot_dual_transform(tenspec[:, 0, 0, :], spec[0:1992, :, 0])


# sheet测试
sheet = sheet_transform.midi_trans_common(
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.mid', 'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
# 画图
util.plot_single_transform(sheet[-5000:-4000, :])

# 切片实验
g = model_cnn_common.Train_CNN_DataGenerator(io='train', output='multitone_duration')
g = g()
y = next(g)
print(y[0].shape, y[1].shape)
# (1016, 2, 9, 356) (1016, 1)
util.plot_single_transform(y[0][:, 0, 0, :])
util.plot_dual_transform(y[0][:, 0, 0, :], y[1])


# pretty_midi实验
time_start = time.time()
out_midi = sheet_transform.midi_trans_duration(
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.mid',
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
time_end = time.time()
print(time_end - time_start)
util.plot_single_transform((out_midi))
# 测试
a = pm.PrettyMIDI('data\MAPS_MUS-bk_xmas1_ENSTDkAm.mid')
a.get_piano_roll(fs=125).shape
a.instruments
a.instruments[0].notes[0]
ret = np.zeros((len(a.instruments[0].notes), 3))
for index in range(len(a.instruments[0].notes)):
    ret[index, 0] = a.instruments[0].notes[index].start
    ret[index, 1] = a.instruments[0].notes[index].end
    ret[index, 2] = a.instruments[0].notes[index].pitch

df = pd.DataFrame(ret)
df.columns = [
    'onset',
    'offset',
    'pitch']
# 保存为csv
csv_path = 'models\\result\\midi_notes.csv'
df.to_csv(csv_path)
a.key_signature_changes # 调式改变
a.time_signature_changes # 节拍改变

# wave实验
a = wave.open('data\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
print(a.getnframes())

# LayerNormalization实验
spec_cqt = spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                      bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                      fmin=initial.config['spec.cqt.fmin'],
                                      hop_length=initial.config['spec.cqt.hop_length'],
                                      window=initial.config['spec.cqt.window'])
spec = spec_cqt(
    'data\\MAPS_MUS-bk_xmas1_ENSTDkAm.wav')
print(spec.shape)
layer = tf.keras.layers.BatchNormalization(axis=-1)
output = layer(tenspec)
output_np = output.numpy()
print(np.max(output_np), np.mean(output_np), np.std(output_np))
print(np.max(output_np[0, :, :, 0]), np.mean(
    output_np[:, :, :, 0]), np.std(output_np[:, :, :, 0]))

# eval
ref_time = np.array(np.arange(0, 2, 0.02))
ref_freq = np.transpose(np.array([np.arange(1000, 2000, 10), np.arange(1000, 2000, 10)]))
est_time = np.array(np.arange(0, 2, 0.02))
est_freq = np.transpose(np.array([np.arange(1000, 2000, 10), np.arange(1001, 2001, 10)]))
scores = mir_eval.multipitch.evaluate(ref_time, ref_freq, est_time, est_freq, window=0.)
scores

sheets = pd.read_table('data\MAPS_MUS-bk_xmas1_ENSTDkAm.txt', header=0, encoding='utf-8')
sheets = np.array(sheets)
ref_intervals = sheets[:, :-1]
ref_pitches = sheets[:, -1] - 21
est_intervals = sheets[:, :-1]
est_pitches = sheets[:, -1] - 21
mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)


# MIDI测试
sheets = pd.read_table(
    'data\MAPS_MUS-bk_xmas1_ENSTDkAm.txt', header=0, encoding='utf-8')
sheets = np.array(sheets)
mid = model_cnn_inference.MIDI_Maker(
    sheets, 'data\MAPS_MUS-bk_xmas1_ENSTDkAm_make.mid')

mido.second2tick(1.0, 120, 800000)


'''
多线程测试
from multiprocessing import Process, Pipe
def f(index, conn):
  conn.send([index, None, 'hello'])
  conn.close()

def main():
    parent_conn, child_conn = Pipe(duplex=False)
    index = 0
    while index < 10:
        p = Process(target=f, args=(index, child_conn))
        p.start()
        print(parent_conn.recv())
        p.join()
        p.terminate()
        index += 1

if __name__ == '__main__':
    main()

'''
