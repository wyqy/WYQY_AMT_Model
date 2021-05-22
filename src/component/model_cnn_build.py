import os, sys
import math
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial

dense_pi_init = -1 * \
    math.log((1-initial.config['detect.train.dense.pi.init']) /
             (initial.config['detect.train.dense.pi.init']))

# TODO: 残差网络https://zhuanlan.zhihu.com/p/31852747, https://www.yanxishe.com/TextTranslation/1643
# TODO: Inception块: https://book.51cto.com/art/201906/598648.htm

def detect_multitone_model_cnn_build(needPrint=False) -> keras.Model:
    """
    detect_multitone_model_cnn_build函数
    功能: 多音调起始/持续时间检测
    输出: 模型
    """
    # 模型搭建
    inputs = keras.Input(
        shape=(initial.config['detect.tone.slice'],
               initial.config['spec.cqt.n_bins'],
               2),
        name='spec_input')

    conv_1 = layers.Conv2D(
        filters=10, kernel_size=(2, 16), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_1')(inputs)

    bn_1 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_1')(conv_1)
    relu_1 = layers.LeakyReLU(alpha=0.1, name='relu_1')(bn_1)
    pool_1 = layers.MaxPool2D(
        pool_size=(2, 2), strides=(1, 2), padding='same', data_format='channels_last', name='pool_1')(relu_1)

    conv_2 = layers.Conv2D(
        filters=20, kernel_size=(3, 12), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_2')(pool_1)

    bn_2 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_2')(conv_2)
    relu_2 = layers.LeakyReLU(alpha=0.1, name='relu_2')(bn_2)
    pool_2 = layers.MaxPool2D(
        pool_size=(2, 2), strides=(1, 2), padding='same', data_format='channels_last', name='pool_2')(relu_2)

    conv_3 = layers.Conv2D(
        filters=40, kernel_size=(3, 9), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_3')(pool_2)

    bn_3 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_3')(conv_3)
    relu_3 = layers.LeakyReLU(alpha=0.1, name='relu_3')(bn_3)
    pool_3 = layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='pool_3')(relu_3)

    conv_4 = layers.Conv2D(
        filters=80, kernel_size=(2, 5), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_4')(pool_3)

    bn_4 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_4')(conv_4)
    relu_4 = layers.LeakyReLU(alpha=0.1, name='relu_4')(bn_4)

    reshape_5 = layers.Flatten(name='reshape_5')(relu_4)

    fc_5 = layers.Dense(
        units=256, activation=None, use_bias=True,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=4e-3),
        bias_regularizer=None,
        name='fc_5')(reshape_5)
    dropout_5 = layers.Dropout(
        rate=0.5, noise_shape=None, name='dropout_5')(fc_5)
    relu_5 = layers.LeakyReLU(alpha=0.1, name='relu_5')(dropout_5)

    fc_6 = layers.Dense(
        units=88, activation=None, use_bias=True,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Constant(dense_pi_init),
        kernel_regularizer=keras.regularizers.L2(l2=4e-3),
        bias_regularizer=None,
        name='fc_6')(relu_5)

    # 模型创建
    model = keras.Model(inputs=inputs, outputs=fc_6, name='multitone_model')

    # 模型展示
    if needPrint:
        model.summary()
        plot_path = os.path.join(
            initial.config['detect.model.png.path'], 'cnn multitone model.jpg')
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True)

    return model

def detect_start_common_model_cnn_build(needPrint=False) -> keras.Model:
    """
    detect_start_common_model_cnn_build函数
    功能: 公共起始时间检测
    输出: 模型
    """
    # 模型搭建
    inputs = keras.Input(
        shape=(initial.config['detect.start.common.slice'],
               initial.config['spec.cqt.n_bins'],
               2),
        name='spec_input')

    conv_1 = layers.Conv2D(
        filters=10, kernel_size=(5, 16), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_1')(inputs)

    bn_1 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_1')(conv_1)
    relu_1 = layers.LeakyReLU(alpha=0.1, name='relu_1')(bn_1)
    pool_1 = layers.MaxPool2D(
        pool_size=(2, 2), strides=(1, 2), padding='same', data_format='channels_last', name='pool_1')(relu_1)

    conv_2 = layers.Conv2D(
        filters=20, kernel_size=(3, 12), strides=(1, 1),
        padding='valid', data_format='channels_last', activation=None,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=5e-5),
        bias_regularizer=None,
        name='conv_2')(pool_1)

    bn_2 = layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=1e-6,
        center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=keras.regularizers.l2(l2=1e-5),
        gamma_regularizer=keras.regularizers.l2(l2=1e-5),
        renorm=False, trainable=True, name='bn_2')(conv_2)
    relu_2 = layers.LeakyReLU(alpha=0.1, name='relu_2')(bn_2)
    pool_2 = layers.MaxPool2D(
        pool_size=(2, 2), strides=(1, 2), padding='same', data_format='channels_last', name='pool_2')(relu_2)

    reshape_3 = layers.Flatten(name='reshape_3')(pool_2)

    fc_3 = layers.Dense(
        units=256, activation=None, use_bias=True,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=keras.regularizers.L2(l2=4e-3),
        bias_regularizer=None,
        name='fc_3')(reshape_3)
    dropout_3 = layers.Dropout(
        rate=0.5, noise_shape=None, name='dropout_3')(fc_3)
    relu_3 = layers.LeakyReLU(alpha=0.1, name='relu_3')(dropout_3)

    fc_4 = layers.Dense(
        units=1, activation=None, use_bias=True,
        kernel_initializer=keras.initializers.GlorotUniform(),
        bias_initializer=keras.initializers.Constant(dense_pi_init),
        kernel_regularizer=keras.regularizers.L2(l2=4e-3),
        bias_regularizer=None,
        name='fc_4')(relu_3)

    # 模型创建
    model = keras.Model(inputs=inputs, outputs=fc_4, name='start_common_model')

    # 模型展示
    if needPrint:
        model.summary()
        plot_path = os.path.join(
            initial.config['detect.model.png.path'], 'cnn start common model.jpg')
        keras.utils.plot_model(model, to_file=plot_path, show_shapes=True)

    return model


def main():
    return 0


if __name__ == '__main__':
  main()
