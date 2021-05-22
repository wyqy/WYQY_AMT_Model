import os, sys
import time
import pickle
import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.model_cnn_common as model_cnn_common


class train_model():
    def __init__(self, model: keras.Model, type='multitone_start', resume=False, ckpt_path=''):
        self.model = model
        self.type = type
        self.resume = resume
        self.ckpt_path = ckpt_path

        self.tn_steps = initial.config['detect.tone.epoch.steps']
        self.cm_steps = initial.config['detect.start.common.epoch.steps']

    def __call__(self):
        """
        功能: 训练模型
        """
        # 模型配置
        self.model = model_cnn_common.model_compile(self.type, self.model)

        # 如果需要恢复则恢复模型变量, 注意此时epoch, learning rate需要手动调整!
        if self.resume:
            self.model.load_weights(self.ckpt_path, by_name=False)

        # 生成器准备
        self.generator_preparation()

        # 生成器恢复
        # 不需要, 因为checkpoint是按epoch保存的, 此时两offset都为0
        # 如果要接续训练, 请手动调整epoch次数和学习率!

        # 数据准备
        self.data_preparation()

        # 数据预取
        self.trainsets = self.trainsets.prefetch(
            buffer_size=tf.data.AUTOTUNE)
        self.testsets = self.testsets.prefetch(
            buffer_size=tf.data.AUTOTUNE)

        # 训练回调
        ckpt_path = os.path.join(
            initial.config['detect.model.checkpoint.path'],
            self.type,
            'chckpt_' +
            time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) +
            '{epoch:02d}.hdf5')
        tsbd_path = os.path.join(
            initial.config['detect.model.tensorboar.path'],
            self.type,
            time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())))
        os.makedirs(tsbd_path)
        # early_stop = keras.callbacks.EarlyStopping(
        #     monitor='macro_binary_fb_evaluation',
        #     min_delta=0.002,
        #     patience=8,
        #     verbose=1)

        self.train_callbacks = [
            # 定期保存
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor='macro_binary_fb_evaluation',
                save_best_only=False,
                save_weights_only=True,
                save_freq='epoch'),
            # 打开监视面板
            # 在 notebooks, 使用 %tensorboard 命令. 在命令行中, 运行不带“％”的相同命令.
            # %tensorboard --logdir "models\tensorboard"
            # https://www.tensorflow.org/tensorboard/get_started?hl=zh-cn
            keras.callbacks.TensorBoard(
                log_dir=tsbd_path,
                write_images=True,
                update_freq='epoch',
                profile_batch=2),
            # 动态调整学习率
            keras.callbacks.ReduceLROnPlateau(
                monitor='macro_binary_fb_evaluation',
                factor=0.1,
                patience=4,
                verbose=1,
                min_delta=0.005,
                cooldown=3,
                min_lr=1e-6)]
            # 提前结束训练
            #early_stop
        
        # 模型训练
        history = self.model.fit(
            self.trainsets,
            epochs=initial.config['detect.train.epoch'],
            verbose=1,
            callbacks=self.train_callbacks)
        # 模型保存
        save_path = os.path.join(
            initial.config['detect.model.save.path'],
            self.type,
            'saved_' + time.strftime("%Y-%m-%d-%H_%M_%S",
                                      time.localtime(time.time())))
        self.model.save(
            filepath=save_path,
            overwrite=False,
            include_optimizer=True,
            save_format='tf',
            save_traces=True)
        # history序列化保存
        # https://www.liaoxuefeng.com/wiki/1016959663602400/1017624706151424
        f_path = os.path.join(
            initial.config['detect.model.history.path'],
            self.type,
            'history_' + time.strftime("%Y-%m-%d-%H_%M_%S",
                                       time.localtime(time.time())))
        with open(f_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        # 模型评估
        loss = self.model.evaluate(
            self.testsets,
            verbose=1,
            return_dict=True)
        # loss序列化保存
        # https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras/51303340
        f_path = os.path.join(
            initial.config['detect.model.loss.path'],
            self.type,
            'loss_' + time.strftime("%Y-%m-%d-%H_%M_%S",
                                    time.localtime(time.time())))
        with open(f_path, 'wb') as file_pi:
            pickle.dump(loss, file_pi)

        # 返回
        return (history, loss)

    def generator_preparation(self):
        """
        功能: 生成器准备
        """
        self.train_generator = model_cnn_common.Train_CNN_DataGenerator(
            io='train', output=self.type)
        self.test_generator = model_cnn_common.Train_CNN_DataGenerator(
            io='test', output=self.type)
    
    def data_preparation(self):
        """
        功能: 数据准备
        """
        # 数据准备
        if self.type == 'multitone_start' or self.type == 'multitone_duration':
            self.trainsets = tf.data.Dataset.from_generator(
                generator=self.train_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(None,
                               initial.config['detect.tone.slice'],
                               initial.config['spec.cqt.n_bins'],
                               2),
                        dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 88), dtype=tf.float32)))
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
            self.trainsets = tf.data.Dataset.from_generator(
                generator=self.train_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(None,
                               initial.config['detect.start.common.slice'],
                               initial.config['spec.cqt.n_bins'],
                               2),
                        dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)))
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


def main():
    return 0


if __name__ == '__main__':
  main()
