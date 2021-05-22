import os
import re
import wave
import json
import pandas as pd


def config_initalize() -> dict:
    """
    config_initalize函数
    功能: 初始化配置文件
    输入: 无
    输出: config_content
    """

    # 读取json配置
    config_file = open('config\\config.json', 'r', encoding='UTF-8')
    config_content = config_file.read()
    config_content = json.loads(config_content)
    config_file.close()

    config_file = open('config\\cuda.json', 'r', encoding='UTF-8')
    cuda_contnet = config_file.read()
    cuda_contnet = json.loads(cuda_contnet)
    config_file.close()

    return (config_content, cuda_contnet)


# 读取配置文件
config, cuda = config_initalize()
# 启用XLA加速
cuda_path = cuda['xla.cuda.path']
cuda_path = '--xla_gpu_cuda_data_dir=\'' + cuda_path + '\''
os.environ['XLA_FLAGS'] = cuda_path
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# 修复
# https://github.com/tensorflow/tensorflow/issues/44697


def dataset_maps_init() -> tuple:
    """
    dataset_maps_init函数
    功能: 初始化数据集标记
    输入: 无
    输出: 二个pd.DataFrame
    """

    # 读取csv文件为数组
    datacsv = pd.read_csv(config['dataset.csv.path'])

    # 打乱并划分train, test, (validation)
    # 按照环境独立(context-independent)原则, 使用合成音频训练, 真钢琴录音测试
    train_files = pd.DataFrame(columns=('midi_filename', 'audio_filename', 'duration'))
    test_files = pd.DataFrame(columns=('midi_filename', 'audio_filename', 'duration'))
    for index in range(datacsv.shape[0]):
        if re.search(pattern=r'(ENSTDkAm|ENSTDkCl)', string=datacsv.iloc[index].at['midi_filename']) == None:
            train_files = train_files.append(datacsv.iloc[index], ignore_index=True)
        else:
            test_files = test_files.append(datacsv.iloc[index], ignore_index=True)
    train_files = train_files.sample(frac=1.0, replace=False)
    test_files = test_files.sample(frac=1.0, replace=False)

    #train_files = datacsv.sample(frac=0.8, replace=False)
    #outcome_files = datacsv[~datacsv.index.isin(train_files.index)]
    #test_files = outcome_files.sample(frac=1.0, replace=False)
    # validation_files = outcome_files[~outcome_files.index.isin(test_files.index)]

    # 重置索引
    train_files = train_files.reset_index(drop=True)
    test_files = test_files.reset_index(drop=True)
    # validation_files = validation_files.reset_index(drop=True)

    # 保存三->二个数组
    train_files.to_csv('config\\maps\\train.csv')
    test_files.to_csv('config\\maps\\test.csv')
    # validation_files.to_csv('config\\maps\\validation.csv')

    # 返回三->二个数组
    return (train_files, test_files)


def dataset_musicnet_init() -> tuple:
    """
    dataset_musicnet_init函数
    功能: 初始化数据集标记
    输入: 无
    输出: 二个pd.DataFrame
    """
    pass


def dataset_fusion_init() -> tuple:
    """
    dataset_musicnet_init函数
    功能: 初始化数据集标记
    输入: 无
    输出: 二个pd.DataFrame
    """
    pass


def dataest_maestro_init() -> tuple:
    """
    dataest_maestro_init函数
    功能: 初始化数据集标记
    输入: 无
    输出: 三个pd.DataFrame
    """

    # 读取csv文件为数组
    datacsv = pd.read_csv(config['dataset.csv.path'])

    # 随机打乱
    datacsv = datacsv.sample(frac=1.0, replace=False)

    # 按split列分train, test, validation
    datacsv_grouped = datacsv.groupby('split')
    train_files = datacsv_grouped.get_group('train')
    test_files = datacsv_grouped.get_group('test')
    validation_files = datacsv_grouped.get_group('validation')

    # 保存test和validation
    test_files.to_csv(config['test.file.path'])
    validation_files.to_csv(config['validation.file.path'])

    # 随机打乱train分组, 保存信息
    # 将原索引命名并变为列
    train_files.index = train_files.index.set_names(['original_index'])
    train_files = train_files.reset_index()

    # 打乱分组数据, 丢弃无用的index, 命名新的index
    train_files = train_files.sample(frac=1.0, replace=False)
    train_files = train_files.reset_index(drop=True)
    train_files.index = train_files.index.set_names(['now_index'])

    # 保存分组数据
    train_files.to_csv('config\\maestro\\train.csv')
    test_files.to_csv('config\\maestro\\test.csv')
    validation_files.to_csv('config\\maestro\\validation.csv')

    # 返回三个数组
    return (train_files, test_files, validation_files)


def load_sample_datasets(path: str) -> pd.core.frame.DataFrame:
    """
    func_data_import函数
    功能: 导入数据集
    输入:
    输出: 数据集
    """

    return pd.read_csv(path)


# 读取数据集
# MAPS
trainsets = load_sample_datasets('config\\maps\\train.csv')
testsets = load_sample_datasets('config\\maps\\test.csv')


def dataset_walker():
    # 爬取文件
    filelist = []
    dataframe = pd.DataFrame(columns=('midi_filename', 'audio_filename', 'duration'))
    for root, _, files in os.walk(top='D:\\Model\\Datasets\\Maps', topdown=False):
        for name in files:
            filelist.append(os.path.join(root, name))
    # MIDI列表
    iter = 0
    for index in range(len(filelist)):
        if re.match(pattern=r'.*mid$', string=filelist[index]) != None:
            dataframe.loc[iter] = {'midi_filename': filelist[index]}
            iter += 1
    # 附加WAV
    for index in range(dataframe.shape[0]):
        audio_path = dataframe.iloc[index].at['midi_filename'][:-3] + 'wav'
        dataframe.loc[index, 'audio_filename'] = audio_path
    # 附加时长
    for index in range(dataframe.shape[0]):
        audio_file = wave.open(dataframe.iloc[index].at['audio_filename'])
        duration = audio_file.getnframes() / audio_file.getframerate()
        audio_file.close()
        dataframe.loc[index, 'duration'] = duration
    # 存为csv
    dataframe.to_csv('config\\maps\\dataset.csv')


def main():
    # 初始化metadata
    # MAPS
    # dataset_walker()

    # 数据集分组
    # MAPS
    dataset_maps_init()

if __name__ == '__main__':
  main()
