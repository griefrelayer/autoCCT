# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'autoCubesbToqUT.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(372, 285)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.start_calibration_button = QPushButton(self.centralwidget)
        self.start_calibration_button.setObjectName(u"start_calibration_button")
        self.start_calibration_button.setGeometry(QRect(80, 220, 75, 23))
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(40, 30, 65, 96))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.matrix_button = QRadioButton(self.layoutWidget)
        self.matrix_button.setObjectName(u"matrix_button")
        self.matrix_button.setChecked(True)

        self.verticalLayout.addWidget(self.matrix_button)

        self.matrices_button = QRadioButton(self.layoutWidget)
        self.matrices_button.setObjectName(u"matrices_button")

        self.verticalLayout.addWidget(self.matrices_button)

        self.cube_button = QRadioButton(self.layoutWidget)
        self.cube_button.setObjectName(u"cube_button")

        self.verticalLayout.addWidget(self.cube_button)

        self.cubes_button = QRadioButton(self.layoutWidget)
        self.cubes_button.setObjectName(u"cubes_button")

        self.verticalLayout.addWidget(self.cubes_button)

        self.layoutWidget1 = QWidget(self.centralwidget)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(130, 30, 96, 46))
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.nophone_button = QRadioButton(self.layoutWidget1)
        self.nophone_button.setObjectName(u"nophone_button")

        self.verticalLayout_2.addWidget(self.nophone_button)

        self.adb_button = QRadioButton(self.layoutWidget1)
        self.adb_button.setObjectName(u"adb_button")

        self.verticalLayout_2.addWidget(self.adb_button)

        self.layoutWidget2 = QWidget(self.centralwidget)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(250, 30, 97, 46))
        self.verticalLayout_3 = QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.photoncam_button = QRadioButton(self.layoutWidget2)
        self.photoncam_button.setObjectName(u"photoncam_button")
        self.photoncam_button.setChecked(True)

        self.verticalLayout_3.addWidget(self.photoncam_button)

        self.gcam_button = QRadioButton(self.layoutWidget2)
        self.gcam_button.setObjectName(u"gcam_button")

        self.verticalLayout_3.addWidget(self.gcam_button)

        self.layoutWidget3 = QWidget(self.centralwidget)
        self.layoutWidget3.setObjectName(u"layoutWidget3")
        self.layoutWidget3.setGeometry(QRect(40, 140, 115, 46))
        self.verticalLayout_4 = QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.warm_button = QRadioButton(self.layoutWidget3)
        self.warm_button.setObjectName(u"warm_button")
        self.warm_button.setEnabled(False)
        self.warm_button.setCheckable(True)

        self.verticalLayout_4.addWidget(self.warm_button)

        self.cool_button = QRadioButton(self.layoutWidget3)
        self.cool_button.setObjectName(u"cool_button")
        self.cool_button.setEnabled(False)

        self.verticalLayout_4.addWidget(self.cool_button)

        self.onlycheck_button = QPushButton(self.centralwidget)
        self.onlycheck_button.setObjectName(u"onlycheck_button")
        self.onlycheck_button.setGeometry(QRect(220, 220, 75, 23))
        self.adb_connect_button = QPushButton(self.centralwidget)
        self.adb_connect_button.setObjectName(u"adb_connect_button")
        self.adb_connect_button.setGeometry(QRect(140, 90, 81, 23))
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(190, 140, 147, 46))
        self.verticalLayout_5 = QVBoxLayout(self.widget)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.show_points_box = QCheckBox(self.widget)
        self.show_points_box.setObjectName(u"show_points_box")

        self.verticalLayout_5.addWidget(self.show_points_box)

        self.nowb_box = QCheckBox(self.widget)
        self.nowb_box.setObjectName(u"nowb_box")

        self.verticalLayout_5.addWidget(self.nowb_box)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 372, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Color correction transform calibration", None))
        self.start_calibration_button.setText(QCoreApplication.translate("MainWindow", u"Calibrate", None))
        self.matrix_button.setText(QCoreApplication.translate("MainWindow", u"matrix", None))
        self.matrices_button.setText(QCoreApplication.translate("MainWindow", u"matrices", None))
        self.cube_button.setText(QCoreApplication.translate("MainWindow", u"cube", None))
        self.cubes_button.setText(QCoreApplication.translate("MainWindow", u"cubes", None))
        self.nophone_button.setText(QCoreApplication.translate("MainWindow", u"local (offline)", None))
        self.adb_button.setText(QCoreApplication.translate("MainWindow", u"phone via ADB", None))
        self.photoncam_button.setText(QCoreApplication.translate("MainWindow", u"PhotonCamera", None))
        self.gcam_button.setText(QCoreApplication.translate("MainWindow", u"Gcam", None))
        self.warm_button.setText(QCoreApplication.translate("MainWindow", u"warm temperature", None))
        self.cool_button.setText(QCoreApplication.translate("MainWindow", u"cool temperature", None))
        self.onlycheck_button.setText(QCoreApplication.translate("MainWindow", u"Re-calibrate", None))
        self.adb_connect_button.setText(QCoreApplication.translate("MainWindow", u"ADB connect", None))
        self.show_points_box.setText(QCoreApplication.translate("MainWindow", u"Show points", None))
        self.nowb_box.setText(QCoreApplication.translate("MainWindow", u"switch off WB calibration", None))
    # retranslateUi

