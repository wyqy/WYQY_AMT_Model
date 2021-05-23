import os, sys
import pickle
import multiprocessing
from multiprocessing import connection
import numpy as np
import pandas as pd
import mir_eval
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.model_cnn_inference as model_cnn_inference
import src.component.model_cnn_common as model_cnn_common
import src.component.model_cnn_build as model_cnn_build


# 可能不需要用以下函数(因为较为麻烦)
# multipitch 需要将源文件变为关于频率的pred和true
    # 可以考虑直接变换成关于pitch的pred和true
# 好处: 内置了window这一参数!


class test_model():
    def __init__(self, type='multitone_start', weight_path_start=''):
        # 其它参数
        self.type = type
        self.weight_path_start = weight_path_start

    def __call__(self):
        '''
        功能: 测试模型的整体acc
        '''
        loss_list = []
        for index in range(1, initial.config['detect.train.epoch']+1):
            weight_path = self.weight_path_start[:-7] + ('%02d' % index) + '.hdf5'
            loss = self.evaluate_proc_wrapper(weight_path)
            loss_list.append(loss)

        # loss_list序列化保存
        # https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras/51303340
        _, f_path = os.path.split(self.weight_path_start)
        f_path = f_path[:-5]
        f_path = os.path.join(
            initial.config['detect.model.loss.path'],
            self.type,
            'loss_' + f_path)
        with open(f_path, 'wb') as file_pi:
            pickle.dump(loss_list, file_pi)
        return loss_list

    def evaluate_proc_wrapper(self, weight_path):
        '''
        功能: 进程包装器
        '''
        # 创建管道
        recvPipe, sendPipe = multiprocessing.Pipe(duplex=False)

        proc = multiprocessing.Process(target=self.evaluate,
                                       kwargs={
                                           'weight_path': weight_path,
                                           'sendPipe': sendPipe})
        # 启动进程
        proc.start()
        # 阻塞接收方法, 保证进程完成
        loss = recvPipe.recv()
        proc.join()  # 等待子进程
        # 关闭进程
        proc.terminate()

        # 返回结果
        return loss

    def evaluate(self, weight_path: str, sendPipe: connection.Connection, *args, **kwargs):
        '''
        功能: 单个模型测试
        '''
        # 限制内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        # 模型配置
        model = self.model_preparation()
        model = model_cnn_common.model_compile(self.type, model)
        # 导入权重
        model.load_weights(weight_path, by_name=False)
        # 生成器准备
        self.generator_preparation()
        # 数据准备
        self.data_preparation()
        # 数据预取
        self.testsets = self.testsets.prefetch(
            buffer_size=tf.data.AUTOTUNE)
        # 模型评估
        loss = model.evaluate(
            self.testsets,
            verbose=1,
            return_dict=True)
        # 返回
        sendPipe.send(loss)
        # 对应主进程, 确保管道关闭
        sendPipe.close()

    def model_preparation(self):
        """
        功能: 模型准备
        """
        if self.type == 'multitone_start' or self.type == 'multitone_duration':
            model = model_cnn_build.detect_multitone_model_cnn_build()
        if self.type == 'common_start':
            model = model_cnn_build.detect_start_common_model_cnn_build()()
        return model

    def generator_preparation(self):
        """
        功能: 生成器准备
        """
        self.test_generator = model_cnn_common.Train_CNN_DataGenerator(
            io='test', output=self.type)
    
    def data_preparation(self):
        """
        功能: 数据准备
        """
        # 数据准备
        if self.type == 'multitone_start' or self.type == 'multitone_duration':
            self.testsets = tf.data.Dataset.from_generator(
                generator=self.test_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(None,
                               initial.config['detect.tone.slice'],
                               initial.config['spec.cqt.n_bins'],
                               2),
                        dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 88), dtype=tf.float32)))
        if self.type == 'common_start':
            self.testsets = tf.data.Dataset.from_generator(
                generator=self.test_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(None,
                               initial.config['detect.start.common.slice'],
                               initial.config['spec.cqt.n_bins'],
                               2),
                        dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))


def eval_preprocess(multitone_start_path, common_start_path, multitone_duration_path):
    '''
    eval_preprocess函数
    功能: 预处理, 保留缓存!
    '''
    # 生成对象
    preder = model_cnn_inference.Transcription_Model(
        multitone_start_path, common_start_path, multitone_duration_path)
    # 遍历计算
    for file_offset in range(initial.testsets.shape[0]):
        # 取路径
        audio_path = initial.testsets.iloc[file_offset].at['audio_filename']
        # 生成缓存
        numpy_path = audio_path[:-4] + '_ret.npy'
        if not os.path.isfile(numpy_path):
            # 计算预测值
            _, notes = preder.notes_plot(audio_path)
            # 暂存预测值
            np.save(numpy_path, notes, allow_pickle=False)

        # 输出进度
        print('evaluation percentage: %s [%d/%d]' %
              (str(round(1.0*(file_offset+1)/initial.testsets.shape[0]*100, 2)),
               (file_offset+1), initial.testsets.shape[0]), end='\r')

    # 还原windows下默认print
    print('', end='\n')


def eval_all_pitch(multitone_start_path, common_start_path, multitone_duration_path):
    '''
    eval_all_pitch函数
    功能: 评价所有音高!
    '''
    # 生成向量
    results = np.zeros((initial.testsets.shape[0], 14), dtype=np.float32)
    # 预处理
    eval_preprocess(multitone_start_path, common_start_path,
                    multitone_duration_path)
    # 遍历计算
    for file_offset in range(results.shape[0]):
        # 取路径
        audio_path = initial.testsets.iloc[file_offset].at['audio_filename']
        txt_path = audio_path[:-3] + 'txt'
        # 取缓存
        numpy_path = audio_path[:-4] + '_ret.npy'
        notes = np.load(numpy_path, allow_pickle=False)
        # 切片时间
        est_intervals = notes[:, :-1]
        # 切片音高
        # 不能给ref_pitches加21, 否则会报non-positive错误!
        est_pitches = notes[:, -1] + 21
        # 得到标准值
        labels = pd.read_table(txt_path, header=0, encoding='utf-8')
        labels = np.array(labels)
        ref_intervals = labels[:, :-1]
        ref_pitches = labels[:, -1]
        # 使用mir_eval计算
        eval = mir_eval.transcription.evaluate(
            ref_intervals, ref_pitches, est_intervals, est_pitches)
        # 将值记录在最终数组中
        # 记录顺序: Precision, Recall, F-measure, Average_Overlap_Ratio,
        #   Precision_no_offset, Recall_no_offset, F-measure_no_offset, Average_Overlap_Ratio_no_offset,
        #   Onset_Precision, Onset_Recall, Onset_F-measure,
        #   Offset_Precision, Offset_Recall, Offset_F-measure
        # 共14个数据
        results[file_offset, 0] = eval['Precision']
        results[file_offset, 1] = eval['Recall']
        results[file_offset, 2] = eval['F-measure']
        results[file_offset, 3] = eval['Average_Overlap_Ratio']
        results[file_offset, 4] = eval['Precision_no_offset']
        results[file_offset, 5] = eval['Recall_no_offset']
        results[file_offset, 6] = eval['F-measure_no_offset']
        results[file_offset, 7] = eval['Average_Overlap_Ratio_no_offset']
        results[file_offset, 8] = eval['Onset_Precision']
        results[file_offset, 9] = eval['Onset_Recall']
        results[file_offset, 10] = eval['Onset_F-measure']
        results[file_offset, 11] = eval['Offset_Precision']
        results[file_offset, 12] = eval['Offset_Recall']
        results[file_offset, 13] = eval['Offset_F-measure']

    # 加权平均
    # return np.mean(results, axis=0)
    # 作为DataFrame返回
    df = pd.DataFrame(results)
    df.columns = [
        'Precision',
        'Recall',
        'F-measure',
        'Average_Overlap_Ratio',
        'Precision_no_offset',
        'Recall_no_offset',
        'F-measure_no_offset',
        'Average_Overlap_Ratio_no_offset',
        'Onset_Precision',
        'Onset_Recall',
        'Onset_F-measure',
        'Offset_Precision',
        'Offset_Recall',
        'Offset_F-measure']
    # 保存为csv
    csv_path = os.path.join(initial.config['eval.model.result.path'], 'all_pitch.csv')
    df.to_csv(csv_path)
    return df


def est_each_pitch():
    '''
eval_each_pitch函数
    功能: 使用测试集进行模型验证, 基于音符
    返回: 数据组成的数组
    '''
    # 生成向量
    results = np.zeros((88, 1), dtype=np.float32)
    labels_overall = np.zeros((0, 3))
    # 遍历计算
    for file_offset in range(initial.testsets.shape[0]):
        # 取路径
        audio_path = initial.testsets.iloc[file_offset].at['audio_filename']
        txt_path = audio_path[:-3] + 'txt'
        # 得到标准值
        labels = pd.read_table(txt_path, header=0, encoding='utf-8')
        labels = np.array(labels)
        labels_overall = np.append(labels_overall, labels, axis=0)

    # 按不同音高计算
    for pitch_offset in range(0, 88):
        # 抽取对应音高数据
        pitch_index = pitch_offset + 21
        results[pitch_offset, 0] = sum(labels_overall[:, -1] == pitch_index)

    # 加权平均
    # return np.mean(results, axis=0)
    # 作为DataFrame返回
    df = pd.DataFrame(results)
    df.columns = ['count']
    # 保存为csv
    csv_path = os.path.join(
        initial.config['eval.model.result.path'], 'est_each_pitch.csv')
    df.to_csv(csv_path)
    return df


def eval_each_pitch(multitone_start_path, common_start_path, multitone_duration_path):
    '''
    eval_each_pitch函数
    功能: 使用测试集进行模型验证, 基于音符
    返回: 14个数据组成的数组
    '''
    # 生成向量
    results = np.zeros((88, 14), dtype=np.float32)
    # 预处理
    eval_preprocess(multitone_start_path, common_start_path,
                    multitone_duration_path)
    notes_overall = np.zeros((0, 3))
    labels_overall = np.zeros((0, 3))
    # 遍历计算
    for file_offset in range(initial.testsets.shape[0]):
        # 取路径
        audio_path = initial.testsets.iloc[file_offset].at['audio_filename']
        txt_path = audio_path[:-3] + 'txt'
        # 取缓存
        numpy_path = audio_path[:-4] + '_ret.npy'
        notes = np.load(numpy_path, allow_pickle=False)
        # 不能给ref_pitches加21, 否则会报non-positive错误!
        notes[:, -1] += 21
        notes_overall = np.append(notes_overall, notes, axis=0)
        # 得到标准值
        labels = pd.read_table(txt_path, header=0, encoding='utf-8')
        labels = np.array(labels)
        labels_overall = np.append(labels_overall, labels, axis=0)

    # 按不同音高计算
    for pitch_offset in range(0, 88):
        # 抽取对应音高数据
        pitch_index = pitch_offset + 21
        pitched_est_intervals = notes_overall[notes_overall[:, -1] == pitch_index, :-1]
        pitched_est_pitches = notes_overall[notes_overall[:, -1] == pitch_index, -1]
        pitched_ref_intervals = labels_overall[labels_overall[:, -1] == pitch_index, :-1]
        pitched_ref_pitches = labels_overall[labels_overall[:, -1] == pitch_index, -1]
        # 若音符数为0则不计算
        if pitched_est_intervals.shape[0] != 0 and pitched_ref_intervals.shape[0] != 0:
            # 使用mir_eval计算
            eval = mir_eval.transcription.evaluate(
                pitched_ref_intervals, pitched_ref_pitches,
                pitched_est_intervals, pitched_est_pitches)
            # 将值记录在最终数组中
            # 记录顺序: Precision, Recall, F-measure, Average_Overlap_Ratio,
            #   Precision_no_offset, Recall_no_offset, F-measure_no_offset, Average_Overlap_Ratio_no_offset,
            #   Onset_Precision, Onset_Recall, Onset_F-measure,
            #   Offset_Precision, Offset_Recall, Offset_F-measure
            # 共14个数据
            results[pitch_offset, 0] = eval['Precision']
            results[pitch_offset, 1] = eval['Recall']
            results[pitch_offset, 2] = eval['F-measure']
            results[pitch_offset, 3] = eval['Average_Overlap_Ratio']
            results[pitch_offset, 4] = eval['Precision_no_offset']
            results[pitch_offset, 5] = eval['Recall_no_offset']
            results[pitch_offset, 6] = eval['F-measure_no_offset']
            results[pitch_offset, 7] = eval['Average_Overlap_Ratio_no_offset']
            results[pitch_offset, 8] = eval['Onset_Precision']
            results[pitch_offset, 9] = eval['Onset_Recall']
            results[pitch_offset, 10] = eval['Onset_F-measure']
            results[pitch_offset, 11] = eval['Offset_Precision']
            results[pitch_offset, 12] = eval['Offset_Recall']
            results[pitch_offset, 13] = eval['Offset_F-measure']

    # 加权平均
    # return np.mean(results, axis=0)
    # 作为DataFrame返回
    df = pd.DataFrame(results)
    df.columns = [
        'Precision',
        'Recall',
        'F-measure',
        'Average_Overlap_Ratio',
        'Precision_no_offset',
        'Recall_no_offset',
        'F-measure_no_offset',
        'Average_Overlap_Ratio_no_offset',
        'Onset_Precision',
        'Onset_Recall',
        'Onset_F-measure',
        'Offset_Precision',
        'Offset_Recall',
        'Offset_F-measure']
    # 保存为csv
    csv_path = os.path.join(initial.config['eval.model.result.path'], 'each_pitch.csv')
    df.to_csv(csv_path)
    return df


def main():
    eval_result = eval_all_pitch(
        multitone_start_path = initial.config['detect.model.multistart.weight'],
        common_start_path = initial.config['detect.model.commonstart.weight'],
        multitone_duration_path = initial.config['detect.model.multiduration.weight']
    )
    return 0


if __name__ == '__main__':
  main()
