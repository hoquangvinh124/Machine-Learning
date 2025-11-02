import sys
import os

# Add pyqt directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyqt'))

from PyQt5.QtWidgets import QApplication
from MainWindowEx import MainWindowEx

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindowEx()
    window.show()
    sys.exit(app.exec_())
