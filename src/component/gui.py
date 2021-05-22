import os, sys
import math
import mido
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as matplotlib_qt5agg
from PyQt5 import QtCore, QtWidgets, QtGui


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import src.component.initial as initial
import src.component.model_cnn_inference as model_cnn_inference
import src.component.pre_spec_transform as pre_spec_transform


# 定义常量
FIGURE_FRAMES = 1024


# 参考资料...
# https://maicss.gitbook.io/pyqt5-chinese-tutoral/ 中文教程
# https://cniter.github.io/posts/5f54aa2c.html
# https://lovesoo.org/2020/03/14/pyqt-getting-started/  QtDesigner设计界面
# https://segmentfault.com/a/1190000022814242  信号和槽机制
# https://blog.csdn.net/weixin_42532587/article/details/105689637 实例
# 需要知识: 线程!!!!
#   https://blog.csdn.net/qq_39607437/article/details/79213717


class workerSpecTransform(QtCore.QThread):
    '''
    功能: Spectrogram Transform 工作线程!
    参考: https://zhuanlan.zhihu.com/p/62988456
    另一种实现方法: https://blog.csdn.net/lynfam/article/details/7081757
    '''
    # 信号必须为类属性!
    specResult = QtCore.pyqtSignal(np.ndarray, name='spec_result')

    def __init__(self, *args, **kw):
        super(workerSpecTransform, self).__init__()
        self.argsInit(*args, **kw)
    
    def argsInit(self, path):
        self.path = path

    def __del__(self):
        self.wait()

    def run(self):
        # 数据生成
        specer = pre_spec_transform.rs_spec_cqt(n_bins=initial.config['spec.cqt.n_bins'],
                                                bins_per_octave=initial.config['spec.cqt.bins_per_octave'],
                                                fmin=initial.config['spec.cqt.fmin'],
                                                frame_length=initial.config['spec.cqt.frame_length'],
                                                hop_length=initial.config['spec.cqt.hop_length'],
                                                window=initial.config['spec.cqt.window'])
        data = specer(self.path)[:, :, 0]
        self.specResult.emit(data)


class workerRollPrediction(QtCore.QThread):
    '''
    功能: Piano-roll Prediction 工作线程!
    '''
    # 信号必须为类属性!
    predRollResult = QtCore.pyqtSignal(np.ndarray, name='pred_roll_result')
    predNotesResult = QtCore.pyqtSignal(np.ndarray, name='pred_notes_result')

    def __init__(self, *args, **kw):
        super(workerRollPrediction, self).__init__()
        self.argsInit(*args, **kw)

    def argsInit(self, path):
        self.path = path

    def __del__(self):
        self.wait()

    def run(self):
        # 数据生成
        predict_test = model_cnn_inference.Transcription_Model(
            initial.config['detect.model.multistart.weight'],
            initial.config['detect.model.commonstart.weight'],
            initial.config['detect.model.multiduration.weight'])
        piano_roll, notes = predict_test.notes_plot(self.path)
        self.predRollResult.emit(piano_roll)
        self.predNotesResult.emit(notes)


class workerMIDIMaker(QtCore.QThread):
    '''
    功能: MIDI Maker 工作线程!
    '''
    # 信号必须为类属性!
    midiResult = QtCore.pyqtSignal(mido.MidiFile, name='midi_result')

    def __init__(self, *args, **kw):
        super(workerMIDIMaker, self).__init__()
        self.argsInit(*args, **kw)

    def argsInit(self, notes, path):
        self.notes = notes
        self.path = path

    def __del__(self):
        self.wait()

    def run(self):
        # 数据生成
        mid = model_cnn_inference.MIDI_Maker(self.notes, self.path)
        self.midiResult.emit(mid)


# 主窗口
class MainWindow(QtWidgets.QMainWindow):
    '''
    功能: 主界面线程!
    '''
    # 信号必须为类属性!
    buttonLock = QtCore.pyqtSignal(bool, name='button_lock')

    def __init__(self):
        super(MainWindow, self).__init__()
        # 初始化变量
        self.initVar()
        # 主界面绘制
        self.initUI()

    def initVar(self):
        # 初始化变量
        self.programStatus = 0
        self.inputfilePath = ''
        self.inputfileType = ''
        self.outputfilePath = ''
        self.sepcResult = np.zeros((FIGURE_FRAMES, initial.config['spec.cqt.n_bins']))
        self.predRollResult = np.zeros((FIGURE_FRAMES, 88))
        self.specCount = 0
        self.specOffset = 0
        self.predCount = 0
        self.predOffset = 0
        # 初始化信号连接
        self.buttonLock.connect(self.slot_buttonlock)
        # 初始化线程对象
        # 参考 https://blog.csdn.net/qq_39607437/article/details/79213717

    def initUI(self):
        # 控件初始化
        self.widgetsSetup()
        # 布局初始化
        self.layoutSetup()
        # resize()方法调整窗口的大小. x px宽y px高
        self.resize(1024, 768)
        # 移动窗口在屏幕上的位置到中心
        self.initUI_center()
        # 设置窗口的标题
        self.setWindowTitle('NeuralNetwork Based Piano Transcription by WYQY 1.16')
        # 设置窗口的图标, 引用当前目录下的图片
        self.setWindowIcon(QtGui.QIcon('src\\component\\gui_icon.png'))
        # 设置初始状态栏
        self.statusBar().showMessage('Ready!')
        # 显示在屏幕上
        self.show()

    def initUI_center(self):
        # 将程序显示到屏幕中心
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def widgetsSetup(self):
        # 左侧按钮
        self.FileInputButton = QtWidgets.QPushButton(self)
        self.FileInputButton.setObjectName('AudiInput')
        self.FileInputButton.setText('Input File')
        self.FileInputButton.clicked.connect(self.slot_button_fileinput)

        self.SpecTransformButton = QtWidgets.QPushButton(self)
        self.SpecTransformButton.setObjectName('SpecTransform')
        self.SpecTransformButton.setText('Spec Transformation')
        self.SpecTransformButton.clicked.connect(self.slot_button_spectransform)

        self.ModelPredictButton = QtWidgets.QPushButton(self)
        self.ModelPredictButton.setObjectName('ModelPredict')
        self.ModelPredictButton.setText('Model Predictor')
        self.ModelPredictButton.clicked.connect(self.slot_button_modelpredict)

        self.MidiPlayerButton = QtWidgets.QPushButton(self)
        self.MidiPlayerButton.setObjectName('MidiPlayer')
        self.MidiPlayerButton.setText('MIDI Player')
        self.MidiPlayerButton.clicked.connect(self.slot_button_midiplayer)

        self.MidiOutputButton = QtWidgets.QPushButton(self)
        self.MidiOutputButton.setObjectName('MidiOutput')
        self.MidiOutputButton.setText('Output MIDI')
        self.MidiOutputButton.clicked.connect(self.slot_button_midioutput)

        self.CloseFileButton = QtWidgets.QPushButton(self)
        self.CloseFileButton.setObjectName('CloseFile')
        self.CloseFileButton.setText('Close File')
        self.CloseFileButton.clicked.connect(self.slot_button_closefile)

        self.AboutButton = QtWidgets.QPushButton(self)
        self.AboutButton.setObjectName('About')
        self.AboutButton.setText('About')
        self.AboutButton.clicked.connect(self.slot_button_about)

        # 右侧图表
        # 参考: https://blog.csdn.net/qq_40587575/article/details/85171401,
        # 参考: https://www.geeksforgeeks.org/how-to-embed-matplotlib-graph-in-pyqt5/
        self.SpecDiagramFigure = plt.figure(figsize=(3, 2), dpi=100)
        self.SpecDiagramFigure.suptitle('Spectrogram Transformation')
        self.SpecDiagramCanvas = matplotlib_qt5agg.FigureCanvasQTAgg(
            self.SpecDiagramFigure)
        self.SpecDiagramToolbar = matplotlib_qt5agg.NavigationToolbar2QT(
            self.SpecDiagramCanvas, self)

        self.PredDiagramFigure = plt.figure(figsize=(3, 2), dpi=100)
        self.PredDiagramFigure.suptitle('Piano-roll Prediction')
        self.PredDiagramCanvas = matplotlib_qt5agg.FigureCanvasQTAgg(
            self.PredDiagramFigure)
        self.PredDiagramToolbar = matplotlib_qt5agg.NavigationToolbar2QT(
            self.PredDiagramCanvas, self)

        # 右侧按钮
        self.SpecLeftButton = QtWidgets.QPushButton(self)
        self.SpecLeftButton.setObjectName('SpecLeft')
        self.SpecLeftButton.setText('<')
        self.SpecLeftButton.clicked.connect(self.slot_button_specleft)

        self.SpecRightButton = QtWidgets.QPushButton(self)
        self.SpecRightButton.setObjectName('SpecRight')
        self.SpecRightButton.setText('>')
        self.SpecRightButton.clicked.connect(self.slot_button_specright)

        self.PredLeftButton = QtWidgets.QPushButton(self)
        self.PredLeftButton.setObjectName('PredLeft')
        self.PredLeftButton.setText('<')
        self.PredLeftButton.clicked.connect(self.slot_button_predleft)

        self.PredRightButton = QtWidgets.QPushButton(self)
        self.PredRightButton.setObjectName('PredRight')
        self.PredRightButton.setText('>')
        self.PredRightButton.clicked.connect(self.slot_button_predright)

        # 右侧文本
        self.SpecOffsetLabel = QtWidgets.QLabel(self)
        self.SpecOffsetLabel.setObjectName('SpecOffset')
        self.SpecOffsetLabel.setText('?/?')
        self.SpecOffsetLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.PredOffsetLabel = QtWidgets.QLabel(self)
        self.PredOffsetLabel.setObjectName('PredOffset')
        self.PredOffsetLabel.setText('?/?')
        self.PredOffsetLabel.setAlignment(QtCore.Qt.AlignCenter)

        # 利用信号和槽实现初始化
        self.buttonLock.emit(False)
        self.buttonLock.emit(True)

    def layoutSetup(self):
        # 参考: https://blog.csdn.net/jia666666/article/details/81702137
        # 布局初始化
        # 全局水平布局
        self.globalGrid = QtWidgets.QHBoxLayout() # 水平
        # 局部xx布局
        self.buttonGrid = QtWidgets.QGridLayout() # 网格
        self.diagramGrid = QtWidgets.QVBoxLayout() # 垂直
        self.specControlGrid = QtWidgets.QHBoxLayout() # 水平
        self.predControlGrid = QtWidgets.QHBoxLayout()  # 水平
        self.specGrid = QtWidgets.QVBoxLayout() # 垂直
        self.predGrid = QtWidgets.QVBoxLayout() # 垂直

        # 添加控件到布局
        self.buttonGrid.addWidget(self.FileInputButton, 0, 0)
        self.buttonGrid.addWidget(self.SpecTransformButton, 2, 0)
        self.buttonGrid.addWidget(self.ModelPredictButton, 4, 0)
        self.buttonGrid.addWidget(self.MidiPlayerButton, 6, 0)
        self.buttonGrid.addWidget(self.MidiOutputButton, 8, 0)
        self.buttonGrid.addWidget(self.CloseFileButton, 10, 0)
        self.buttonGrid.addWidget(self.AboutButton, 12, 0)
        self.specControlGrid.addStretch()
        self.specControlGrid.addWidget(self.SpecLeftButton)
        self.specControlGrid.addStretch()
        self.specControlGrid.addWidget(self.SpecOffsetLabel)
        self.specControlGrid.addStretch()
        self.specControlGrid.addWidget(self.SpecRightButton)
        self.specControlGrid.addStretch()
        self.predControlGrid.addStretch()
        self.predControlGrid.addWidget(self.PredLeftButton)
        self.predControlGrid.addStretch()
        self.predControlGrid.addWidget(self.PredOffsetLabel)
        self.predControlGrid.addStretch()
        self.predControlGrid.addWidget(self.PredRightButton)
        self.predControlGrid.addStretch()
        self.specGrid.addWidget(self.SpecDiagramToolbar)
        self.specGrid.addWidget(self.SpecDiagramCanvas)
        self.predGrid.addWidget(self.PredDiagramToolbar)
        self.predGrid.addWidget(self.PredDiagramCanvas)

        # 设置局部控件
        self.buttonWidget = QtWidgets.QWidget()
        self.diagramWidget = QtWidgets.QWidget()
        self.specControlWidget = QtWidgets.QWidget()
        self.predControlWidget = QtWidgets.QWidget()
        self.specWidget = QtWidgets.QWidget()
        self.predWidget = QtWidgets.QWidget()
        # 添加局部布局
        self.buttonWidget.setLayout(self.buttonGrid)
        self.diagramWidget.setLayout(self.diagramGrid)
        self.specControlWidget.setLayout(self.specControlGrid)
        self.predControlWidget.setLayout(self.predControlGrid)
        self.specWidget.setLayout(self.specGrid)
        self.predWidget.setLayout(self.predGrid)

        # 设置全局布局
        self.diagramGrid.setSpacing(0)
        self.diagramGrid.addWidget(self.specWidget)
        self.diagramGrid.addWidget(self.specControlWidget)
        self.diagramGrid.addWidget(self.predWidget)
        self.diagramGrid.addWidget(self.predControlWidget)
        self.diagramGrid.setStretch(0, 8)
        self.diagramGrid.setStretch(1, 1)
        self.diagramGrid.setStretch(2, 8)
        self.diagramGrid.setStretch(3, 1)
        self.globalGrid.addWidget(self.buttonWidget)
        self.globalGrid.addWidget(self.diagramWidget)
        # 添加全局布局
        self.globalWidget = QtWidgets.QWidget()
        self.globalWidget.setLayout(self.globalGrid)
        self.setCentralWidget(self.globalWidget)

    def closeEvent(self, event):
        '''
        功能: 关闭确认, 重写closeEvent()事件处理程序
        '''
        reply = QtWidgets.QMessageBox.question(
            self, 'Warning', 'Are you sure to quit?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def slot_buttonlock(self, sig_buttonLock):
        '''
        详细解读GUI程序状态:
        0: 程序初始化 -> 开启input, close
        100: 输入midi -> 开启play, close
        200: 输入wav -> 开启spec, pred, close
        201: pred后 -> 开启spec, pred, output, close
        202: output后 -> 开启spec, pred, output, play, close
        '''
        if sig_buttonLock:
            if self.programStatus == 0:
                self.FileInputButton.setEnabled(True)
            if self.programStatus == 100:
                self.MidiPlayerButton.setEnabled(True)
                self.CloseFileButton.setEnabled(True)
            if self.programStatus == 200:
                self.SpecTransformButton.setEnabled(True)
                self.ModelPredictButton.setEnabled(True)
                self.CloseFileButton.setEnabled(True)
            if self.programStatus == 201:
                self.SpecTransformButton.setEnabled(True)
                self.ModelPredictButton.setEnabled(True)
                self.MidiOutputButton.setEnabled(True)
                self.CloseFileButton.setEnabled(True)
            if self.programStatus == 202:
                self.SpecTransformButton.setEnabled(True)
                self.ModelPredictButton.setEnabled(True)
                self.MidiPlayerButton.setEnabled(True)
                self.MidiOutputButton.setEnabled(True)
                self.CloseFileButton.setEnabled(True)
        else:
            self.FileInputButton.setEnabled(False)
            self.SpecTransformButton.setEnabled(False)
            self.ModelPredictButton.setEnabled(False)
            self.MidiPlayerButton.setEnabled(False)
            self.MidiOutputButton.setEnabled(False)
            self.CloseFileButton.setEnabled(False)

    def slot_button_fileinput(self):
        '''
        功能: file input 槽函数
        参考: https://blog.csdn.net/HuangZhang_123/article/details/78144692
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)

        self.inputfilePath, self.inputfileType = QtWidgets.QFileDialog.getOpenFileName(
            self, "File Select", "./",
            "All Files (*);;Wave Files (*.wav);;Midi Files (*.mid)")
        self.chk_programstatus('Input File')

        # 恢复按钮
        self.statusBar().showMessage('Ready! Loaded file named: ' + self.inputfilePath)
        self.buttonLock.emit(True)

    def slot_button_spectransform(self):
        '''
        功能: spec transform 槽函数
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)
        self.chk_programstatus('Spec Transformation')

        self.threadSpec = workerSpecTransform(self.inputfilePath)
        self.threadSpec.specResult.connect(self.slot_thread_spectransform)
        self.threadSpec.finished.connect(self.slot_thread_spectransform_close)
        self.threadSpec.start()

    def slot_thread_spectransform(self, sig_specresult):
        '''
        功能: 线程结果处理函数
        参考: https://www.cnblogs.com/linyfeng/p/12239856.html
        参考: https://blog.csdn.net/zong596568821xp/article/details/78893360
        '''
        self.sepcResult = sig_specresult
        self.specCount = math.ceil(self.sepcResult.shape[0] / FIGURE_FRAMES)
        self.specOffset = 0
        self.matplotDraw(self.sepcResult, self.specOffset, 'jet',
                         self.SpecDiagramFigure, self.SpecDiagramCanvas)
        self.SpecOffsetLabel.setText(
            str(self.specOffset+1) + '/' + str(self.specCount))
    
    def slot_thread_spectransform_close(self):
        '''
        功能: 线程结束函数
        '''
        if self.threadSpec.isRunning():
            self.threadSpec.wait()
        self.threadSpec.specResult.disconnect(self.slot_thread_spectransform)
        self.threadSpec.finished.disconnect(self.slot_thread_spectransform_close)
        del self.threadSpec

        # 恢复按钮
        self.statusBar().showMessage('Ready! Transformed file named: ' + self.inputfilePath)
        self.buttonLock.emit(True)

    def slot_button_modelpredict(self):
        '''
        功能: model predict 槽函数
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)
        self.chk_programstatus('Model Predictor')

        self.threadPred = workerRollPrediction(self.inputfilePath)
        self.threadPred.predRollResult.connect(self.slot_thread_predrolltransform)
        self.threadPred.predNotesResult.connect(self.slot_thread_prednotestransform)
        self.threadPred.finished.connect(self.slot_thread_predtransform_close)
        self.threadPred.start()

    def slot_thread_predrolltransform(self, sig_predrollresult):
        '''
        功能: 线程结果处理函数
        '''
        self.predRollResult = sig_predrollresult
        self.predCount = math.ceil(self.predRollResult.shape[0] / FIGURE_FRAMES)
        self.predOffset = 0
        self.matplotDraw(self.predRollResult, self.predOffset, 'gray_r',
                         self.PredDiagramFigure, self.PredDiagramCanvas)
        self.PredOffsetLabel.setText(
            str(self.predOffset+1) + '/' + str(self.predCount))

    def slot_thread_prednotestransform(self, sig_prednotesresult):
        '''
        功能: 线程结果处理函数
        '''
        self.predNotesResult = sig_prednotesresult

    def slot_thread_predtransform_close(self):
        '''
        功能: 线程结束函数
        '''
        if self.threadPred.isRunning():
            self.threadPred.wait()
        self.threadPred.predRollResult.disconnect(self.slot_thread_predrolltransform)
        self.threadPred.predNotesResult.disconnect(self.slot_thread_prednotestransform)
        self.threadPred.finished.disconnect(self.slot_thread_predtransform_close)
        del self.threadPred

        # 恢复按钮
        self.statusBar().showMessage('Ready! Predicted file named: ' + self.inputfilePath)
        self.buttonLock.emit(True)

    def slot_button_midiplayer(self):
        '''
        功能: MIDI player 槽函数
        参考: https://www.cnblogs.com/xiyuan2016/p/7019686.html 只适用Windows?
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)
        self.chk_programstatus('MIDI Player')

        if self.inputfilePath[-3:] == 'mid' and os.path.isfile(self.inputfilePath):
            os.startfile(self.inputfilePath)
            final_path = self.inputfilePath
        elif self.outputfilePath[-3:] == 'mid' and os.path.isfile(self.outputfilePath):
            os.startfile(self.outputfilePath)
            final_path = self.outputfilePath

        # 恢复按钮
        self.statusBar().showMessage('Ready! Started player for file named: ' + final_path)
        self.buttonLock.emit(True)

    def slot_button_midioutput(self):
        '''
        功能: MIDI output 槽函数
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)
        self.chk_programstatus('Output MIDI')

        self.outputfilePath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "File Save", "./",
            "All Files (*);;Midi Files (*.mid)")
        self.threadMaker = workerMIDIMaker(self.predNotesResult, self.outputfilePath)
        self.threadMaker.midiResult.connect(self.slot_thread_midimaker)
        self.threadMaker.finished.connect(self.slot_thread_midimaker_close)
        self.threadMaker.start()

    def slot_thread_midimaker(self, sig_midiresult):
        '''
        功能: 线程结果处理函数
        '''
        self.midoObject = sig_midiresult

    def slot_thread_midimaker_close(self):
        '''
        功能: 线程结束函数
        '''
        if self.threadMaker.isRunning():
            self.threadMaker.wait()
        self.threadMaker.midiResult.disconnect(self.slot_thread_midimaker)
        self.threadMaker.finished.disconnect(self.slot_thread_midimaker_close)
        del self.threadMaker

        # 恢复按钮
        self.statusBar().showMessage('Ready! Saved for file named: ' + self.outputfilePath)
        self.buttonLock.emit(True)

    def slot_button_closefile(self):
        '''
        功能: close file 槽函数
        '''
        # 锁住按钮
        self.statusBar().showMessage('Busy...')
        self.buttonLock.emit(False)
        self.chk_programstatus('Close File')

        self.inputfileType = ''
        self.outputfilePath = ''

        # 恢复按钮
        self.statusBar().showMessage('Ready! Closed file named: ' + self.inputfilePath)
        self.inputfilePath = ''
        self.buttonLock.emit(True)

    def slot_button_about(self):
        '''
        功能: about 槽函数
        '''
        QtWidgets.QMessageBox.about(
            self, 'About Piano Transcription System',
            'Powered by TensorFlow, librosa...\
            \nMade by WYQY.\
            \nCopyright 2021© You Jingze.\
            \nDistributed under the terms of CC Public License.')

    def slot_button_specleft(self):
        '''
        功能: spec left 槽函数
        '''
        if self.specOffset > 0:
            self.specOffset -= 1
            self.matplotDraw(self.sepcResult, self.specOffset, 'jet',
                             self.SpecDiagramFigure, self.SpecDiagramCanvas)
            self.SpecOffsetLabel.setText(
                str(self.specOffset+1) + '/' + str(self.specCount))

    def slot_button_specright(self):
        '''
        功能: spec right 槽函数
        '''
        if self.specOffset < self.specCount - 1:
            self.specOffset += 1
            self.matplotDraw(self.sepcResult, self.specOffset, 'jet',
                             self.SpecDiagramFigure, self.SpecDiagramCanvas)
            self.SpecOffsetLabel.setText(
                str(self.specOffset+1) + '/' + str(self.specCount))

    def slot_button_predleft(self):
        '''
        功能: pred left 槽函数
        '''
        if self.predOffset > 0:
            self.predOffset -= 1
            self.matplotDraw(self.predRollResult, self.predOffset, 'gray_r',
                             self.PredDiagramFigure, self.PredDiagramCanvas)
            self.PredOffsetLabel.setText(
                str(self.predOffset+1) + '/' + str(self.predCount))

    def slot_button_predright(self):
        '''
        功能: pred right 槽函数
        '''
        if self.predOffset < self.predCount - 1:
            self.predOffset += 1
            self.matplotDraw(self.predRollResult, self.predOffset, 'gray_r',
                             self.PredDiagramFigure, self.PredDiagramCanvas)
            self.PredOffsetLabel.setText(
                str(self.predOffset+1) + '/' + str(self.predCount))

    def chk_programstatus(self, sender_text):
        '''
        详细解读GUI程序状态:
        0: 程序初始化 -> (输入midi)1, (输入wav)2
        100: 输入midi -> (播放midi, 输出midi)100, (关闭文件)0
        200: 输入wav -> (转spec)200, (转piano-roll)201, (关闭文件)0
        201: pred后 -> (转spec, piano-roll)201, (输出midi)202, (关闭文件)0
        202: output后 -> (转spec, piano-roll)202, (播放, 输出midi)202, (关闭文件)0
        '''
        if self.programStatus == 0:
            if sender_text == 'Input File' and self.inputfilePath[-3:] == 'mid':
                self.programStatus = 100
            if sender_text == 'Input File' and self.inputfilePath[-3:] == 'wav':
                self.programStatus = 200
        if self.programStatus == 100:
            if sender_text == 'Close File':
                self.programStatus = 0
        if self.programStatus == 200:
            if sender_text == 'Model Predictor':
                self.programStatus = 201
            if sender_text == 'Close File':
                self.programStatus = 0
        if self.programStatus == 201:
            if sender_text == 'Output MIDI':
                self.programStatus = 202
            if sender_text == 'Close File':
                self.programStatus = 0
        if self.programStatus == 202:
            if sender_text == 'Close File':
                self.programStatus = 0

    def matplotDraw(self, data, offset, cmap, figure, canvas):
        # 计算长度
        specLength = data.shape[0] - offset*FIGURE_FRAMES
        if specLength < FIGURE_FRAMES:
            target_data = np.zeros((FIGURE_FRAMES, data.shape[1]))
            target_data[0:specLength, :] = data[offset*FIGURE_FRAMES:data.shape[0], :]
        else:
            target_data = data[offset*FIGURE_FRAMES:(offset+1)*FIGURE_FRAMES, :]

        # 清除原始图片
        fig_title = figure._suptitle.get_text()
        figure.clear()
        # 初始化图片
        figure.suptitle(fig_title)
        ax = figure.add_subplot(111)
        # 绘制数据
        ax.imshow(np.transpose(target_data),
                  cmap=plt.cm.get_cmap(cmap),
                  vmin=0.0, vmax=1.0,
                  aspect='auto',
                  interpolation='none',
                  origin='lower')
        # 替换时间轴
        # 计算对应刻度
        frames_per_second = initial.config['spec.cqt.sampling'] / initial.config['spec.cqt.hop_length']
        label_iter = float(math.ceil(offset*FIGURE_FRAMES / frames_per_second))
        loc_iter = int(round((frames_per_second*label_iter - offset*FIGURE_FRAMES)))
        label_list = []
        loc_list = []
        while frames_per_second*label_iter <= (offset+1)*FIGURE_FRAMES:
            label_list.append(str(label_iter))
            loc_list.append(int(loc_iter))
            label_iter += 1.0
            loc_iter = int(round((frames_per_second*label_iter - offset*FIGURE_FRAMES)))
        # 设置对应刻度
        # 参考: https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
        ax.set_xticks(loc_list)
        ax.set_xticklabels(label_list)
        # 刷新画布
        canvas.draw()
        

if __name__ == "__main__":
    # 创建应用程序对象, sys.argv参数是一个列表, 从命令行输入参数.
    app = QtWidgets.QApplication(sys.argv)
    # QWidget部件是pyqt5所有用户界面对象的基类. 它为QWidget提供默认构造函数.
    mainshow = MainWindow()
    # 系统exit()方法确保应用程序干净地退出
    # exec_()方法有下划线. 因为执行是一个Python关键词. 因此用exec_()代替
    sys.exit(app.exec_())
