from PyQt5.QtWidgets import QApplication
from MainWindowEx import MainWindowEx
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindowEx()
    window.show()
    sys.exit(app.exec_())
