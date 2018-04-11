# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from video_detectUI import Ui_MainWindow
import os
import datetime

class MainView(QtWidgets.QMainWindow,Ui_MainWindow):
    u"""GUIのクラス."""

    def __init__(self, parent=None):
        u"""GUI初期設定."""
        # QtGui.QWidget.__init__(self,parent)
        super(MainView, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # 変数
        self.painter = QtGui.QPainter()
        self.model = QtWidgets.QDirModel()
        self.points=[]
        # GUIの初期設定
        self.setFixedSize(self.width(), self.height())
        self.model.setFilter(
            QtCore.QDir.AllDirs | QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllEntries)
        self.treeView.setModel(self.model)
        self.treeView.setColumnWidth(0, 200)
        self.treeView.setColumnHidden(2, True)
        self.treeView.setColumnHidden(3, True)
        self.treeView.setIndentation(10)
        self.treeView.setEnabled(False)
        self.detectionTop_Edit.setValidator(
            QtGui.QIntValidator(1, 65535, self))
        self.detectionBottom_Edit.setValidator(
            QtGui.QIntValidator(1, 65535, self))
        self.detectionLeft_Edit.setValidator(
            QtGui.QIntValidator(1, 65535, self))
        self.detectionRight_Edit.setValidator(
            QtGui.QIntValidator(1, 65535, self))
        self.trackSlider.setEnabled(False)
        self.pixmap = QtGui.QPixmap(640, 360)
        self.pixmap.fill(QtCore.Qt.black)
        self.videoFrame.setPixmap(self.pixmap)
        self.imgscale_comboBox.addItem("1.0")
        self.imgscale_comboBox.addItem("0.5")
        self.imgscale_comboBox.addItem("0.25")

    def set_settings(self, settei):
        u"""設定をGUIに反映."""
        self.avi_checkBox.setChecked(True)
        self.avi_checkBox.setChecked(settei.settings["avi"])
        self.mov_checkBox.setChecked(settei.settings["mov"])
        self.mpg_checkBox.setChecked(settei.settings["mpg"])
        self.mp4_checkBox.setChecked(settei.settings["mp4"])
        self.wmv_checkBox.setChecked(settei.settings["wmv"])
        self.flv_checkBox.setChecked(settei.settings["flv"])
        self.mts_checkBox.setChecked(settei.settings["mts"])
        self.m2ts_checkBox.setChecked(settei.settings["m2ts"])
        self.writevideo_checkBox.setChecked(settei.settings["writevideo"])
        self.writejpg_checkBox.setChecked(settei.settings["writejpg"])
        self.bounding_checkBox.setChecked(settei.settings["bounding"])
        self.display_checkBox.setChecked(settei.settings["display"])
        self.verbose_checkBox.setChecked(settei.settings["verbose"])
        self.learning_checkBox.setChecked(settei.settings["learning"])
        if settei.settings["detecttype"] == "detectA":
           self.detectA_radioButton.setChecked(True)
        elif settei.settings["detecttype"] == "detectB":
           self.detectB_radioButton.setChecked(True)
        elif settei.settings["detecttype"] == "detectC":
           self.detectC_radioButton.setChecked(True)
        self.detectionTop_Edit.setText(str(settei.settings["detectionTop"]))
        self.detectionBottom_Edit.setText(
            str(settei.settings["detectionBottom"]))
        self.detectionLeft_Edit.setText(str(settei.settings["detectionLeft"]))
        self.detectionRight_Edit.setText(
            str(settei.settings["detectionRight"]))
        #順番大事（スケール設定は、範囲設定のあと。スケールを設定するとアクションが作動するので）
        imgscaleindex = self.imgscale_comboBox.findText(str(settei.settings["imgscale"]))
        self.imgscale_comboBox.setCurrentIndex(imgscaleindex)
        self.speedSlider.setValue(settei.settings["speedSlider"])
        self.outdirEdit.setText(settei.settings["outdir"])
        self.set_playdir(settei.settings["playdir"])

    def set_format_checkbox(self, state):
        u"""入力フォーマットチェックボックスの有効＆無効化一括変更."""
        self.avi_checkBox.setEnabled(state)
        self.mov_checkBox.setEnabled(state)
        self.mpg_checkBox.setEnabled(state)
        self.mp4_checkBox.setEnabled(state)
        self.wmv_checkBox.setEnabled(state)
        self.flv_checkBox.setEnabled(state)
        self.mts_checkBox.setEnabled(state)
        self.m2ts_checkBox.setEnabled(state)

    def set_stepvideo_button(self, state):
        u"""ステップビデオボタンの有効＆無効化一括変更."""
        self.previousvideoButton.setEnabled(state)
        self.nextvideoButton.setEnabled(state)
        self.nextframeButton.setEnabled(state)

    def set_settingframe(self, state):
        u"""コンボボックスの有効＆無効化"""
        self.frame.setEnabled(state)

    def set_treeview(self, state):
        u"""ツリービューの有効＆無効化"""
        self.treeView.setEnabled(state)

    def set_inputformat(self):
        u"""入力ビデオフォーマットのフィルタリング設定."""
        namefilter = []
        if self.avi_checkBox.isChecked():
            namefilter.extend(["*.AVI", "*.avi"])
        if self.mov_checkBox.isChecked():
            namefilter.extend(["*.MOV", "*.mov"])
        if self.mpg_checkBox.isChecked():
            namefilter.extend(["*.MPG", "*.mpg"])
        if self.mp4_checkBox.isChecked():
            namefilter.extend(["*.MP4", "*.mp4"])
        if self.wmv_checkBox.isChecked():
            namefilter.extend(["*.WMV", "*.wmv"])
        if self.flv_checkBox.isChecked():
            namefilter.extend(["*.FLV", "*.flv"])
        if self.mts_checkBox.isChecked():
            namefilter.extend(["*.MTS", "*.mts"])
        if self.m2ts_checkBox.isChecked():
            namefilter.extend(["*.M2TS", "*.m2ts"])
        if len(namefilter) == 0:
            namefilter = ["*.nothing_input_format"]

        model = self.treeView.model()
        model.setNameFilters(namefilter)
        self.treeView.setModel(model)

    def set_detectselect(self, state):
        u"""検知選択関連を設定."""
        self.detectionArea_Button.setEnabled(state)
        self.paintBackground_Button.setEnabled(state)
        self.detectionTop_Edit.setEnabled(state)
        self.detectionBottom_Edit.setEnabled(state)
        self.detectionLeft_Edit.setEnabled(state)
        self.detectionRight_Edit.setEnabled(state)
        self.resetArea_Button.setEnabled(state)

    def set_paintselect(self, state):
        u"""背景分類のマウス選択を設定."""
        if state:
            self.paintBackground_Button.setText("選択終了")
        else:
            self.paintBackground_Button.setText("背景分類")

    def set_mouseselect(self, state):
        u"""検知範囲のマウス選択を設定."""
        self.detectionArea_Button.setEnabled(state)

    def paint_background(self, event, paintmode):
        u"""検知範囲のマウス選択処理."""
        if paintmode==1:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0),10)
        elif paintmode==2:
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0),10)
        elif paintmode==3:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255),10)
        else:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255),10)

        # クリック開始
        if (event.type() == QtCore.QEvent.MouseButtonPress):
            self.points.append(event.pos())
            self.drag = True
        # マウスドラッグ中
        elif (self.drag and event.type() == QtCore.QEvent.MouseMove):
            self.points.append(event.pos())
            detectpixmap = self.pixmap.copy()
            self.painter.begin(detectpixmap)
            self.painter.setPen(pen)
            self.painter.drawPolyline(*self.points)
            self.painter.end()
            self.videoFrame.setPixmap(detectpixmap)
            self.pixmap = detectpixmap


    def paint_background_release(self):
        self.points=[]
        self.drag = False

    def select_detectionarea_by_mouse(self, event):
        u"""検知範囲のマウス選択処理."""
        x = event.pos().x()
        y = event.pos().y()
        # クリック開始
        if (event.type() == QtCore.QEvent.MouseButtonPress):
            self.x0 = x
            self.y0 = y
            self.drag = True
        # マウスドラッグ中
        elif (self.drag and event.type() == QtCore.QEvent.MouseMove):
            detectpixmap = self.pixmap.copy()
            self.painter.begin(detectpixmap)
            self.painter.setPen(QtGui.QColor(255, 0, 0))
            self.painter.drawRect(
                self.x0, self.y0, x - self.x0, y - self.y0)
            self.painter.end()
            self.videoFrame.setPixmap(detectpixmap)

    def set_detectionarea_by_mouse(self, event,height):
        u"""検知範囲のマウス選択処理."""
        x = event.pos().x()
        y = event.pos().y()
        ratio = height / 360.0
        top = int(self.y0 * ratio)
        bottom = int(y * ratio)
        left = int(self.x0 * ratio)
        right = int(x * ratio)
        self.set_detectionarea(top, bottom, left, right, height)
        self.detectionArea_Button.setEnabled(True)
        self.drag = False

    def draw_detectionarea(self, top, bottom, left, right, height):
        u"""検知範囲の描画."""
        if height == 0:
            height = 480
        ratio = 360.0 / height  # ビデオの高さと画面の高さの比
        x0 = int(left * ratio)
        x1 = int(right * ratio)
        y0 = int(top * ratio)
        y1 = int(bottom * ratio)
        detectpixmap = self.pixmap.copy()
        self.painter.begin(detectpixmap)
        self.painter.setPen(QtGui.QColor(255, 0, 0))
        self.painter.drawRect(x0, y0, x1 - x0, y1 - y0)
        self.painter.end()
        self.videoFrame.setPixmap(detectpixmap)

    def get_detectionarea(self):
        u"""検知範囲を返す."""
        top = int(self.detectionTop_Edit.text())
        bottom = int(self.detectionBottom_Edit.text())
        left = int(self.detectionLeft_Edit.text())
        right = int(self.detectionRight_Edit.text())
        return top, bottom, left, right

    def set_detectionarea(self, top, bottom, left, right, height):
        u"""検知範囲の設定."""
        self.detectionTop_Edit.setText(str(top))
        self.detectionBottom_Edit.setText(str(bottom))
        self.detectionLeft_Edit.setText(str(left))
        self.detectionRight_Edit.setText(str(right))
        self.draw_detectionarea(top, bottom, left, right, height)

    def get_slider_position(self):
        u"""トラックスライダーの位置を返す."""
        return self.trackSlider.value()

    def get_speed(self):
        u"""再生スピード変更."""
        return self.speedSlider.value()

    def get_bounding(self):
        u"""動体検知エリア表示するか？設定."""
        bounding = self.bounding_checkBox.isChecked()
        return bounding

    def set_writevideo(self, checked):
        u"""ビデオ書き出すか？設定."""
        self.writevideo_checkBox.setChecked(checked)

    def get_writevideo(self):
        u"""ビデオ書き出すか?."""
        writevideo = self.writevideo_checkBox.isChecked()
        return writevideo

    def set_writejpg(self, checked):
        u"""jpg書き出すか？設定."""
        self.writejpg_checkBox.setChecked(checked)

    def get_writejpg(self):
        u"""ビデオ書き出すか？."""
        writejpg = self.writejpg_checkBox.isChecked()
        return writejpg

    def get_display(self):
        u"""画面表示するか？設定."""
        display = self.display_checkBox.isChecked()
        return display

    def clear_display(self):
        u"""画面をクリア."""
        self.videoFrame.clear()

    def get_verbose(self):
        u"""ログ出力を多めにするか？設定."""
        verbose = self.verbose_checkBox.isChecked()
        return verbose

    def get_learning(self):
        u"""学習モードか？"""
        learning = self.learning_checkBox.isChecked()
        return learning

    def get_imgscale(self):
        u"""画像スケール？"""
        imgscale = float(self.imgscale_comboBox.currentText())
        return imgscale

    def set_imgsize_label(self,width,height):
        self.imgsize_label.setText("{}×{}".format(width,height))

    def set_outdir(self, outdir):
        u"""出力フォルダのテキストを設定."""
        self.outdirEdit.setText(outdir)

    def get_outdir(self):
        u"""出力フォルダを返す."""
        outdir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output folder")
        return outdir.replace('/', os.sep)

    def get_playdir(self):
        u"""再生ビデオフォルダ選択.プレイリストをセット."""
        playdir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder")
        return playdir.replace('/', os.sep)

    def get_csvdir(self):
        u"""csvフォルダ選択."""
        csvdir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select CSV folder")
        return csvdir.replace('/', os.sep)

    def get_attachfile(self):
        u"""CSV結合ファイルを選択."""
        attach_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Select csv file','attached_all.csv', 'CSV (*.csv)')
        return attach_file[0]

    def get_device(self):
        u"""カメラデバイスを選択.0 or 1."""
        num, ok = QtWidgets.QInputDialog.getInt(
            self, "Input device number", "device:", 0, 0, 1)
        return num, ok

    def get_loadfile(self):
        u"""設定ファイル（読み込み用）を選択."""
        setting_file = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file','', 'JSON (*.json)')
        return setting_file[0]

    def get_savefile(self):
        u"""設定ファイル（保存用）を選択."""
        setting_file = QtWidgets.QFileDialog.getSaveFileName(self, 'Select file','settings.json', 'JSON (*.json)')
        return setting_file[0]

    def set_playdir(self, playdir):
        u"""プレイリストをセット."""
        self.trackSlider.setEnabled(True)
        self.nextframeButton.setEnabled(True)
        #self.previousvideoButton.setEnabled(True)
        #self.nextvideoButton.setEnabled(True)
        self.treeView.setEnabled(True)
        model = self.treeView.model()
        self.treeView.setRootIndex(model.index(playdir))
        self.fileEdit.setText(playdir)

    def get_playlist(self, playdir):
        u"""フォルダ内のファイルリストを取得."""
        playlist = []
        model = self.treeView.model()
        idx = model.index(playdir)
        for i in range(model.rowCount(idx)):
            child = idx.child(i, 0)
            if model.isDir(child):
                dirlist = self.get_playlist(model.filePath(child))
                playlist.extend(dirlist)
            else:
                playlist.append(str(model.filePath(
                    child)).replace('/', os.sep))

        return playlist

    def set_nothing_playlist(self):
        u"""プレイリストをセットする."""
        print("Nothing input video!")
        self.logEdit.insertPlainText("Nothing input video!\n")
        self.videoFrame.clear()
        self.detectionArea_Button.setEnabled(False)
        self.paintBackground_Button.setEnabled(False)
        self.set_detectselect(False)
        # self.set_video(None)

    def set_video_view(self, filename, framecount,fps):
        u"""再生ビデオを変更＆初期設定."""
        model = self.treeView.model()
        index = model.index(filename)
        self.treeView.scrollTo(index, QtWidgets.QAbstractItemView.PositionAtCenter)
        self.treeView.selectionModel().clearSelection()
        self.treeView.selectionModel().select(
            index, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        self.trackSlider.setMaximum(framecount)
        d = datetime.timedelta(seconds=int(
            framecount / fps))
        self.trackalllabel.setText(str(d))

        self.trackSlider.setValue(0)
        self.pixmap.fill(QtCore.Qt.black)
        self.videoFrame.setPixmap(self.pixmap)
        self.detectionArea_Button.setEnabled(True)
        self.paintBackground_Button.setEnabled(True)
        self.set_detectselect(True)

    def get_filename_treeview(self, index):
        u"""ツリービューから選択されたファイルを返す."""
        model = self.treeView.model()
        indexItem = model.index(index.row(), 0, index.parent())
        filename = model.filePath(indexItem).replace('/', os.sep)
        return filename

    def set_webcam_view(self, webcam=True):
        u"""webcamを入力デバイスに設定."""
        # ToDo:書き出ししないとか設定確認
        if webcam:
            self.treeView.setEnabled(False)
            self.pixmap.fill(QtCore.Qt.black)
            self.videoFrame.setPixmap(self.pixmap)
            self.detectionArea_Button.setEnabled(True)
            self.paintBackground_Button.setEnabled(True)
            self.trackSlider.setEnabled(False)
            self.trackalllabel.setText("0:00:00")
            self.nextframeButton.setEnabled(False)
            self.nextvideoButton.setEnabled(False)
            self.previousvideoButton.setEnabled(False)
            self.fileEdit.setText("Input is from WebCam...")
        else:
            self.fileEdit.setText("No connected with WebCam !")

    def set_play_label(self, label):
        u"""ビデオ再生 or 停止処理."""
        self.playButton.setText(label)

    def change_tracklabel(self, fps):
        u"""再生経過秒数表示."""
        if fps != 0:
            d = datetime.timedelta(seconds=int(
                self.trackSlider.value() / fps))
            self.tracklabel.setText(str(d))

    def set_track_position(self, curpos):
        u"""トラックスライダー位置設定."""
        self.trackSlider.setValue(curpos)

    def set_frame(self, frame):
        u"""画面設定."""
        height, width = frame.shape[:2]
        img = QtGui.QImage(frame, width, height, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixmap = pixmap.scaled(640, 360, QtCore.Qt.KeepAspectRatio)
        self.videoFrame.setPixmap(pixmap)
        self.pixmap = pixmap

    def write_log(self, str):
        u"""ログ表示."""
        self.logEdit.insertPlainText(str)

    def clear_log(self):
        u"""ログ消去."""
        self.logEdit.clear()
