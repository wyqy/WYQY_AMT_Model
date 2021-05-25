import numpy as np
import matplotlib.pyplot as plt


def plot_single_transform(transform, cmap='jet'):
    """
    plot_single_transform函数
    功能: 输入二维Tensor表示, 输出图像
    输入: transform张量(m纵轴 x n横轴)
    输出:
    """

    # 准备画布
    plt.close()
    plt.figure(num=1, figsize=(10, 5), dpi=100)

    # 在画布上绘图
    # xyz: x行y列第几张子图
    plt.subplot(111)
    plt.imshow(np.transpose(transform),
               # https://matplotlib.org/2.0.2/users/colormaps.html
               cmap=plt.cm.get_cmap(cmap),
               vmin=0.0, vmax=1.0,
               aspect='auto',
               interpolation='none',
               origin='lower')
    plt.ylabel('Transform Channel(s)', fontsize=15)
    plt.xlabel('Frame(s)', fontsize=15)
    plt.title('Transform', fontsize=18)
    plt.axis('on')

    # 展示图
    plt.show()

    # 关闭图
    # plt.close('all')


def plot_single_label(transform, cmap='jet'):
    """
    plot_single_transform函数
    功能: 输入二维Tensor表示, 输出图像
    输入: transform张量(m纵轴 x n横轴)
    输出:
    """

    # 准备画布
    plt.close()
    plt.figure(num=1, figsize=(10, 5), dpi=100)

    # 在画布上绘图
    # xyz: x行y列第几张子图
    plt.subplot(111)
    plt.imshow(np.transpose(transform),
               # https://matplotlib.org/2.0.2/users/colormaps.html
               cmap=plt.cm.get_cmap(cmap),
               vmin=0.0, vmax=1.0,
               aspect='auto',
               interpolation='none',
               origin='lower')
    plt.ylabel('Label Channel(s)', fontsize=15)
    plt.xlabel('Frame(s)', fontsize=15)
    plt.title('Label', fontsize=18)
    plt.axis('on')

    # 展示图
    plt.show()


def plot_dual_transform(transform_1, transform_2, cmap_1='jet', cmap_2='gray_r'):
    '''
    plot_dual_transform函数:
    功能: 输入两个二维Tensor表示, 输出共x轴图像
    输入: transform张量(m纵轴 x n横轴)x2
    输出:
    '''
    plt.close()
    fig = plt.figure(num=1, figsize=(12, 8), dpi=100)
    ax1 = fig.add_subplot(211)
    ax1.imshow(np.transpose(transform_1),
               cmap=plt.cm.get_cmap(cmap_1),
               vmin=0.0, vmax=1.0,
               aspect='auto',
               interpolation='none',
               origin='lower')
    ax1.set_xlabel('Frame(s)', fontsize=15)
    ax1.set_ylabel('Transform Channel(s)', fontsize=15)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.imshow(np.transpose(transform_2),
               cmap=plt.cm.get_cmap(cmap_2),  # gray_r
               vmin=0.0, vmax=1.0,
               aspect='auto',
               interpolation='none',
               origin='lower')
    ax2.set_ylabel('Label Channel(s)', fontsize=15)
    plt.axis('on')
    plt.show()


def main():
    return 0


if __name__ == '__main__':
  main()
