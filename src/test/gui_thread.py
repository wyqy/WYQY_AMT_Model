import pyqtgraph as pg
import numpy as np
#import time
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal


class plotarT(QThread):

    signal = pyqtSignal(object)                             # object

    def __init__(self, parent=None):
        super().__init__()
        self.s = None                                      # +++
        self.phase = 0                                      # +++

    def __del__(self):
        self.wait()

    def update(self):
        #        self.phase = 0                                     # ---
        self.t = np.arange(0, 3.0, 0.01)
        self.s = np.sin(2 * np.pi * self.t + self.phase)    # Sin function
        self.phase += 0.1
        QThread.msleep(200)    # time.sleep(0.2)

    def run(self):
        for i in range(100):                                # +++ Some cycle
            self.update()
            self.signal.emit(self.s)                        # self.s


class Window(QDialog):
    def __init__(self):
        #        self.app = QtGui.QApplication(sys.argv)
        super().__init__()
        self.title = "PyQt5 GridLayout"
        self.top = 100
        self.left = 100
        self.width = 1000
        self.height = 600
        self.InitWindow()

        self.traces = dict()
        pg.setConfigOptions(antialias=True)

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.gridLayoutCreation()

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.groupBox)
        self.setLayout(vboxLayout)

        self.show()

    def gridLayoutCreation(self):
        self.groupBox = QGroupBox("Grid Layout Example")

        gridLayout = QGridLayout()
        self.guiplot = pg.PlotWidget()
        gridLayout.addWidget(self.guiplot, 0, 8, 8, 12)
        self.groupBox.setLayout(gridLayout)

        gridLayout.addWidget(QLabel('Tempo'), 0, 0)
        self.timeEdit = QLineEdit('')                       # time <-> timeEdit
        # time <-> timeEdit
        gridLayout.addWidget(self.timeEdit, 1, 0)

    def plotar(self, s):
        self.guiplot.clear()
        self.guiplot.plot(s)
        #self.guiplot.plot(c)

    def teste(self):
        self.get_thread = plotarT()
        self.get_thread.signal.connect(self.displayS)        # <--- +++
        self.get_thread.start()

    def displayS(self, self_s):                              # <--- +++
        """ Here is your `self.s` array. 
            Draw a graph in real time without blocking the graphical interface.
        """
#        print("\n Here is your `self.s` array. \n", self_s)

        self.plotar(self_s)


def main():
    app = QApplication(sys.argv)  # QtGui.
    form = Window()
    form.show()
    form.teste()
    app.exec_()


if __name__ == '__main__':
    main()
