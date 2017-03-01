# -*- coding: utf-8 -*-

"""

Usage:
    video_detect [options] --settings=SETTING_FILE

Detect animals from VIDEO file or CAMERA.

Arguments:
    SETTING_FILE        Setting file (json format)

Options:
    --inpdir=IN_PATH    Input directory of VIDEO files.(recursive)
    --outdir=OUT_PATH   Output directory
    --debug             Debug mode
    --help              Show this screen.
    --version           Show version.

Examples:
    video_detect.py --settings=C://Users/mizutani/Desktop/video_detect/settings.json
    video_detect.py --settings=settings.json --inpdir=D://video/2017-02-20 --outdir=C://Users/mizutani/Desktop/out/2017-02-20
"""

from docopt import docopt
import sys
import os
from os.path import join, splitext, basename
from PyQt4 import QtGui, QtCore
from View import MainView
from Settei import Settei
from Video import Video
from MyUtil import find_rootdir

class MainController():
    u"""処理クラス."""

    def __init__(self, settei, main_view):
        u"""初期設定."""
        self.settei = settei
        self.main_view = main_view
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.play)
        self.video = None
        self.playlist = None
        self.playnumber = None
        self.playfile = None
        self.pause = True
        self.mouseselectmode = False

        u"""設定の読み込み."""
        inidir = os.path.abspath(find_rootdir())
        setting_file = join(inidir, "settings.json")
        self.load_settings(setting_file)
        # GUIのアクション設定（設定読み込みの後に実行）
        self.set_actions()

    def set_actions(self):
        u"""アクションの設定."""
        self.main_view.actionOpen_Folder.triggered.connect(self.set_playdir)
        self.main_view.folderButton.clicked.connect(self.set_playdir)
        self.main_view.actionWebCam.triggered.connect(self.get_webcam)
        self.main_view.actionSave_Settings.triggered.connect(
            self.save_settings)
        self.main_view.actionLoad_Settings.triggered.connect(
            self.get_settings)
        self.main_view.outdirButton.clicked.connect(self.set_outdir)
        self.main_view.playButton.clicked.connect(self.control_video)
        self.main_view.nextframeButton.clicked.connect(self.step_nextframe)
        self.main_view.nextvideoButton.clicked.connect(self.step_nextvideo)
        self.main_view.previousvideoButton.clicked.connect(
            self.step_previousvideo)
        self.main_view.trackSlider.valueChanged.connect(self.changed_slider)
        self.main_view.trackSlider.sliderPressed.connect(self.pressed_slider)
        self.main_view.trackSlider.sliderReleased.connect(self.moved_slider)
        self.main_view.speedSlider.valueChanged.connect(self.set_speed)
        self.main_view.treeView.doubleClicked.connect(
            self.select_playfile_treeview)
        self.main_view.writevideo_checkBox.stateChanged.connect(
            self.set_writevideo)
        self.main_view.writejpg_checkBox.stateChanged.connect(
            self.set_writejpg)
        self.main_view.bounding_checkBox.stateChanged.connect(
            self.set_bounding)
        self.main_view.display_checkBox.stateChanged.connect(self.set_display)
        self.main_view.verbose_checkBox.stateChanged.connect(self.set_verbose)
        self.main_view.learning_checkBox.stateChanged.connect(self.set_learning)
        self.main_view.detectA_radioButton.clicked.connect(self.set_detectA)
        self.main_view.detectB_radioButton.clicked.connect(self.set_detectB)
        self.main_view.detectC_radioButton.clicked.connect(self.set_detectC)
        self.main_view.avi_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mov_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mpg_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mp4_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.wmv_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.flv_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mts_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.m2ts_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.resetArea_Button.clicked.connect(
            self.reset_detectarea)
        self.main_view.detectionTop_Edit.editingFinished.connect(
            self.check_detectionarea)
        self.main_view.detectionBottom_Edit.editingFinished.connect(
            self.check_detectionarea)
        self.main_view.detectionLeft_Edit.editingFinished.connect(
            self.check_detectionarea)
        self.main_view.detectionRight_Edit.editingFinished.connect(
            self.check_detectionarea)
        self.main_view.detectionArea_Button.clicked.connect(
            self.active_mouseselect)
        self.main_view.videoFrame.mousePressEvent = self.set_detectionarea_by_mouse
        self.main_view.videoFrame.mouseReleaseEvent = self.set_detectionarea_by_mouse
        self.main_view.videoFrame.mouseMoveEvent = self.set_detectionarea_by_mouse
        self.main_view.keyPressEvent = self.set_label
        self.main_view.imgscale_comboBox.currentIndexChanged.connect(self.set_imgscale)

    def set_label(self,event):
        key = event.key()
        if key == QtCore.Qt.Key_1:#いる
            self.video.set_label(1)
        elif key == QtCore.Qt.Key_0:#いない
            self.video.set_label(0)
        elif key == QtCore.Qt.Key_2:#いるけど、うごかない
            self.video.set_label(2)

    def play(self, singlestep=False):
        u"""再生＆表示＆検知処理.（タイマーからループで呼び出される）."""
        # 1フレーム処理
        ret, skip = self.video.process_nextframe()
        if ret is True:
            curpos = self.video.get_position()
            self.main_view.set_track_position(curpos)
            # ビデオ終わりでなければビデオ＆jpg書き出しチェック＆画面表示
            if not skip:
                if self.settei.settings["writejpg"]:
                    self.video.writeout_jpg()
                if self.settei.settings["writevideo"]:
                    self.video.writeout_video()
                # if self.settei.settings["webcam"]:
                #    self.video.writeout_webcam()
                if self.settei.settings["display"]:
                    RGBframe = self.video.get_RGBframe()
                    self.main_view.set_frame(RGBframe)
        else:
            # ビデオ終わりなら書き込み終了＆ログ出力＆グラフ出力
            self.video.close_video()
            filename = join(self.settei.settings["outdir"], 'result.csv')
            self.video.writeout_result(filename)
            # 次のファイルを確認＆変更
            if len(self.playlist) > self.playnumber + 1:
                # ファイル残っていたら次のビデオを再生
                self.playnumber = self.playnumber + 1
                filename = self.playlist[self.playnumber]
                self.set_video(filename)
                if not singlestep:
                   self.control_video("START")
            else:
                # ファイル残っていなかったら停止
                self.control_video("STOP")


    def control_video(self, action):
        u"""ビデオ再生 or 停止処理."""
        if self.video is not None:
            if action == "START":
                # 再生させる
                self.pause = False
                self.main_view.set_play_label("STOP")
                self.main_view.set_format_checkbox(False)
                self.main_view.set_stepvideo_button(False)
                self.main_view.set_treeview(False)
                self.main_view.set_settingframe(False)
                self.video.set_detecttype(self.settei.settings["detecttype"])
                self._timer.start()
            elif action == "STOP":
                # 停止させる
                self.pause = True
                self._timer.stop()
                self.video.close_video()  # ビデオも一旦書き出し
                self.main_view.set_play_label("START")
                self.main_view.set_settingframe(True)
                if not self.settei.settings["webcam"]:
                    self.main_view.set_format_checkbox(True)
                    self.main_view.set_stepvideo_button(True)
                    self.main_view.set_treeview(True)
            elif self.pause:
                # 停止していたら再生させる
                self.pause = False
                self.main_view.set_play_label("STOP")
                self.main_view.set_format_checkbox(False)
                self.main_view.set_stepvideo_button(False)
                self.main_view.set_treeview(False)
                self.main_view.set_settingframe(False)
                self.video.set_detecttype(self.settei.settings["detecttype"])
                self._timer.start()
            else:
                # 再生していたら停止させる
                self.pause = True
                self._timer.stop()
                self.video.close_video()  # ビデオも一旦書き出し
                self.main_view.set_play_label("START")
                self.main_view.set_settingframe(True)
                if not self.settei.settings["webcam"]:
                    self.main_view.set_format_checkbox(True)
                    self.main_view.set_stepvideo_button(True)
                    self.main_view.set_treeview(True)

    def step_nextframe(self):
        u"""コマ送り再生."""
        #self.control_video("STOP")
        if self.video is not None:
            if self.video.bs is None:
                self.video.set_detecttype(self.settei.settings["detecttype"])
            self.play(singlestep=True)

    def step_previousvideo(self):
        u"""前のビデオに切り替え."""
        if len(self.playlist) > 0:
            if self.playnumber > 0 and self.video.curpos==0:
                self.playnumber = self.playnumber - 1
            # playnumberが0の場合と再生途中の場合は、同じビデオを初期設定する。
            filename = self.playlist[self.playnumber]
            self.set_video(filename)

    def step_nextvideo(self):
        u"""次のビデオに切り替え."""
        if len(self.playlist) > self.playnumber + 1:
            self.playnumber = self.playnumber + 1
            filename = self.playlist[self.playnumber]
            self.set_video(filename)

    def changed_slider(self, pos):
        u"""トラックバー変更でラベル変更."""
        fps = self.video.get_fps()
        self.main_view.change_tracklabel(fps)

    def pressed_slider(self):
        u"""トラックバー移動でビデオ停止."""
        self.control_video("STOP")

    def moved_slider(self):
        u"""トラックバーの移動終了."""
        pos = self.main_view.get_slider_position()
        self.video.set_position(pos)
        self.refresh_view()

    def set_outdir(self):
        u"""出力フォルダを設定."""
        outdir = self.main_view.get_outdir()
        if outdir:
            self.settei.settings["outdir"] = outdir
            self.main_view.set_outdir(outdir)
        if self.video:
            if self.settei.settings["webcam"]:
                self.video.set_outdir(outdir)
            else:
                outdir = self.settei.settings["outdir"]
                playdir = self.settei.settings["playdir"]
                playfile = self.playfile
                recursive_outdir = self.get_recursive_outdir(
                    outdir, playdir, playfile)
                self.video.set_outdir(recursive_outdir)

    def select_playfile_treeview(self, index):
        u"""ツリービューから選択されたファイルを再生ビデオに設定."""
        playfile = self.main_view.get_filename_treeview(index)
        if os.path.isfile(playfile):  # directoryなら何もしない
            playnumber = self.playlist.index(playfile)
            self.playnumber = playnumber
            self.playfile = playfile
            self.set_video(playfile)

    def set_playdir(self):
        u"""プレイフォルダを設定."""
        self.control_video("STOP")
        playdir = self.main_view.get_playdir()
        if playdir:
            self.settei.settings["webcam"] = False
            self.settei.settings["playdir"] = playdir
            self.main_view.set_playdir(playdir)
            self.set_inputformat()

    def set_inputformat(self):
        u"""入力ビデオフォーマットのフィルタリング設定.プレイリストも更新."""
        self.main_view.set_inputformat()
        self.set_playlist()

    def set_playlist(self):
        u"""再生リストの設定."""
        playdir = self.settei.settings["playdir"]
        self.playlist = self.main_view.get_playlist(playdir)
        if len(self.playlist) == 0:
            self.video = None
            self.main_view.set_nothing_playlist()
        else:
            self.playnumber = 0
            playfile = self.playlist[self.playnumber]
            self.playfile = playfile
            self.set_video(playfile)

    def get_recursive_outdir(self, outdir, playdir, playfile):
        u"""playfileのフォルダ構成を考慮したビデオ＆画像出力フォルダ."""
        tmppath = outdir + os.sep + playfile.replace(playdir, "").lstrip(
            '\\')  # playdirがG:\とかだとバックスラッシュが取れるためos.sep追加。取れない場合のために左のバックスラッシュを一旦とる。
        recursive_outdir = os.path.dirname(str(tmppath))
        return recursive_outdir

    def set_video(self, playfile):
        u"""再生ビデオを変更＆初期設定."""
        if self.settei.settings["verbose"]:
            print(playfile)
        self.video = None
        outdir = self.settei.settings["outdir"]
        playdir = self.settei.settings["playdir"]
        recursive_outdir = self.get_recursive_outdir(outdir, playdir, playfile)
        logfunc = self.main_view.write_log
        self.video = Video(playfile, recursive_outdir,logfunc)
        self.video.set_bounding(self.settei.settings["bounding"])
        self.video.set_verbose(self.settei.settings["verbose"])
        self.video.set_learning(self.settei.settings["learning"])
        self.video.set_detecttype(self.settei.settings["detecttype"])
        framecount = self.video.get_framecount()
        self.main_view.set_video_view(playfile, framecount)
        top = self.settei.settings["detectionTop"]
        bottom = self.settei.settings["detectionBottom"]
        left = self.settei.settings["detectionLeft"]
        right = self.settei.settings["detectionRight"]
        self.set_detectionarea(top, bottom, left, right)
        self.set_imgscale()
        self.set_speed()

    def reset_detectarea(self):
        u"""検知範囲リセット."""
        if self.video is not None:
            height, width = self.video.get_size()
            top = 0
            bottom = height
            left = 0
            right = width
            self.set_detectionarea(top, bottom, left, right)


    def refresh_view(self):
        u"""表示を更新する."""
        if self.settei.settings["display"]:
            bounding = self.settei.settings["bounding"]
            currentframe = self.video.get_currentframe(bounding)
            self.main_view.set_frame(currentframe)
        else:
            self.main_view.clear_display()

    def set_detectionarea(self, top, bottom, left, right):
        u"""検知範囲のビデオへの設定."""
        height, width = self.video.get_size()
        self.main_view.set_detectionarea(top, bottom, left, right, height)
        self.settei.settings["detectionTop"] = top
        self.settei.settings["detectionBottom"] = bottom
        self.settei.settings["detectionLeft"] = left
        self.settei.settings["detectionRight"] = right
        self.video.set_detectionarea(top, bottom, left, right)
        self.refresh_view()

    def check_detectionarea(self):
        u"""検知エリアの確認."""
        top, bottom, left, right = self.main_view.get_detectionarea()
        # 逆の場合は、入れ替える。
        if bottom - top < 0:
            tmp = top
            top = bottom
            bottom = tmp
        if right - left < 0:
            tmp = right
            right = left
            left = tmp
        # 左右、上下が同じ値ならもとに戻す
        if bottom - top == 0 or right - left == 0:
            top = self.settei.settings["detectionTop"]
            bottom = self.settei.settings["detectionBottom"]
            left = self.settei.settings["detectionLeft"]
            right = self.settei.settings["detectionRight"]
        self.set_detectionarea(top, bottom, left, right)

    def get_webcam(self):
        u"""カメラデバイスの取得."""
        device, ok = self.main_view.get_device()
        if ok:
            self.set_webcam(device)

    def set_webcam(self, device):
        u"""webcamの設定."""
        self.settei.settings["webcam"] = True
        self.settei.settings["device"] = device
        self.video = None
        outdir = self.settei.settings["outdir"]
        playfile = self.playfile = device
        logfunc = self.main_view.write_log
        self.video = Video(playfile, outdir, logfunc, webcam=True)
        # webcamが接続されていれば初期設定
        if self.video.check_webcam() is True:
            self.video.set_bounding(self.settei.settings["bounding"])
            self.video.set_verbose(self.settei.settings["verbose"])
            self.video.set_learning(self.settei.settings["learning"])
            self.video.set_detecttype(self.settei.settings["detecttype"])
            self.main_view.set_webcam_view()
            top = self.settei.settings["detectionTop"]
            bottom = self.settei.settings["detectionBottom"]
            left = self.settei.settings["detectionLeft"]
            right = self.settei.settings["detectionRight"]
            self.set_detectionarea(top, bottom, left, right)
            self.set_imgscale()
            self.set_speed()
            self.control_video("START")
        else:
            self.main_view.set_webcam_view(webcam=False)

    def get_settings(self):
        u"""設定の読み込み."""
        setting_file = self.main_view.get_loadfile()
        if setting_file:
            self.load_settings(setting_file)

    def load_settings(self, setting_file):
        u"""設定の保存."""
        self.settei.load_settings(setting_file)
        self.main_view.set_settings(self.settei)
        if self.settei.settings["webcam"]:
            device = self.settei.settings["device"]
            self.set_webcam(device)
        else:
            self.set_inputformat()

    def save_settings(self):
        u"""設定の保存."""
        setting_file = self.main_view.get_savefile()
        if setting_file:
            self.settei.save_settings(setting_file)

    def set_speed(self):
        u"""タイマーのインターバルを設定."""
        if self.settei.settings["webcam"]:
            interval = 0
        else:
            speedvalue = self.main_view.get_speed()
            # speed: 0～2 の値 0はできるだけ速く.2は通常の2倍の時間遅い
            speed = -speedvalue / 50.0 + 2
            self.settei.settings["speedSlider"] = speedvalue
            interval = 0
            if self.video is not None:
                interval = speed * self.video.get_fps()
        self._timer.setInterval(interval)

    def set_bounding(self):
        u"""領域表示設定."""
        bounding = self.main_view.get_bounding()
        self.settei.settings["bounding"] = bounding
        if self.video is not None:
            self.video.set_bounding(bounding)
        self.refresh_view()

    def set_writevideo(self):
        u"""ビデオを書き出すか設定."""
        writevideo = self.main_view.get_writevideo()
        self.settei.settings["writevideo"] = writevideo

    def set_writejpg(self):
        u"""jpgを書き出すか設定."""
        writejpg = self.main_view.get_writejpg()
        self.settei.settings["writejpg"] = writejpg

    def set_display(self):
        u"""画面表示設定."""
        display = self.main_view.get_display()
        self.settei.settings["display"] = display
        if display is False:
            self.main_view.clear_display()
        else:
            self.refresh_view()

    def set_verbose(self):
        u"""ログ出力を設定."""
        verbose = self.main_view.get_verbose()
        self.settei.settings["verbose"] = verbose
        if self.video is not None:
            self.video.set_verbose(verbose)

    def set_learning(self):
        u"""ログ出力を設定."""
        learning = self.main_view.get_learning()
        self.settei.settings["learning"] = learning
        if self.video is not None:
            self.video.set_learning(learning)

    def set_imgscale(self):
        u"""画像スケールを設定."""
        imgscale = self.main_view.get_imgscale()
        old_imgscale = self.settei.settings["imgscale"]
        self.settei.settings["imgscale"] = imgscale
        if self.video is not None:
            self.video.set_imgscale(imgscale)
            height,width = self.video.get_size()
            self.main_view.set_imgsize_label(width,height)
            top, bottom, left, right = self.main_view.get_detectionarea()
            aspect = imgscale / old_imgscale
            self.set_detectionarea(
                int(top * aspect), int(bottom * aspect), int(left * aspect), int(right * aspect))

    def set_detectA(self, enabled):
        u"""表示画像を設定."""
        self.settei.settings["detecttype"] = "detectA"
        if self.video is not None:
            self.video.set_detecttype("detectA")

    def set_detectB(self, enabled):
        u"""表示画像を設定."""
        self.settei.settings["detecttype"] = "detectB"
        if self.video is not None:
            self.video.set_detecttype("detectB")

    def set_detectC(self, enabled):
        u"""表示画像を設定."""
        self.settei.settings["detecttype"] = "detectC"
        if self.video is not None:
            self.video.set_detecttype("detectC")

    def set_detectionarea_by_mouse(self, event):
        u"""検出範囲をマウスで選択."""
        if self.video and self.mouseselectmode:
            if (event.type() != QtCore.QEvent.MouseButtonRelease):
                self.main_view.select_detectionarea_by_mouse(event)
            elif (event.type() == QtCore.QEvent.MouseButtonRelease):
                height, width = self.video.get_size()
                self.main_view.set_detectionarea_by_mouse(event, height)
                self.check_detectionarea()
                self.mouseselectmode = False

    def active_mouseselect(self):
        u"""検知範囲のマウス選択を有効化."""
        self.mouseselectmode = True
        self.main_view.set_mouseselect(False)


class CuiController():
    u"""Cuiコントローラークラス."""

    def __init__(self, settei):
        u"""初期設定."""
        self.settei = settei
        self.video = None
        self.playlist = None
        self.playnumber = None
        self.playfile = None
        self.pause = True
        # コマンドライン引数取得
        args = docopt(__doc__, version="0.1.2")
        self.settei.load_cui_settings(args)
        if self.settei.settings["webcam"]:  # カメラ入力
            print("Input from webcam...")
            self.set_webcam_cli()
            ok = True
            while(ok):
                try:
                    ok = self.play_cli()
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt!!!\n")
                    self.video.close_video()
                    sys.exit()
        else:  # ビデオ入力
            playdir = self.settei.settings["playdir"]
            if os.path.exists(playdir):
                self.playlist = self.get_playlist(playdir)
                if len(self.playlist) > 0:
                    # プレイリストのビデオに対してチェック
                    for playfile in self.playlist:
                        self.set_video_cli(playfile)
                        ok = True
                        while(ok):
                            ok = self.play_cli()
                else:
                    print("Nothing playlist.")
            else:
                print("playdir does not exist.")
            sys.exit()

    def set_video_cli(self, playfile):
        u"""ビデオの初期設定."""
        print(playfile)
        tmppath = self.settei.settings[
            "outdir"] + playfile.replace(self.settei.settings["playdir"], "")
        recursive_outdir = os.path.dirname(tmppath)
        self.playfile = playfile
        self.video = Video(playfile, recursive_outdir)
        self.video.set_bounding(
            self.settei.settings["bounding"])
        self.video.set_detecttype(self.settei.settings["detecttype"])
        self.video.set_verbose(self.settei.settings["verbose"])
        self.video.set_learning(self.settei.settings["learning"])
        top = self.settei.settings["detectionTop"]
        bottom = self.settei.settings["detectionBottom"]
        left = self.settei.settings["detectionLeft"]
        right = self.settei.settings["detectionRight"]
        self.video.set_detectionarea(top, bottom, left, right)
        self.video.set_imgscale(self.settei.settings["imgscale"])

    def set_webcam_cli(self):
        u"""カメラの初期設定."""
        outdir = self.settei.settings["outdir"]
        playfile = self.playfile = self.settei.settings["device"]
        self.video = Video(playfile, outdir,webcam=True)
        if self.video.check_webcam() is True:
            self.video.set_bounding(
                self.settei.settings["bounding"])
            self.video.set_detecttype(self.settei.settings["detecttype"])
            self.video.set_verbose(self.settei.settings["verbose"])
            self.video.set_learning(self.settei.settings["learning"])
            top = self.settei.settings["detectionTop"]
            bottom = self.settei.settings["detectionBottom"]
            left = self.settei.settings["detectionLeft"]
            right = self.settei.settings["detectionRight"]
            self.video.set_detectionarea(top, bottom, left, right)
            self.video.set_imgscale(self.settei.settings["imgscale"])
        else:
            print("No connected with WebCam !")
            sys.exit()

    def play_cli(self):
        u"""再生＆検知."""
        # 1フレーム処理
        ret,skip = self.video.process_nextframe()

        if ret is True:
                # ビデオ終わりでなければビデオ＆jpg書き出しチェック＆実行
            if not skip:
                if self.settei.settings["writejpg"]:
                    self.video.writeout_jpg()
                if self.settei.settings["writevideo"]:
                    self.video.writeout_video()
                # if self.settei.settings["webcam"]:
                #    self.video.writeout_webcam()
        else:
            # ビデオ終わりなら書き込み終了＆結果出力＆検知グラフ＆データ出力
            self.video.close_video()

        return ret

    def fild_all_files(self, directory):
        u"""ビデオリスト取得のための関数."""
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def get_playlist(self, playdir):
        u"""ディレクトリ内のビデオのリストを取得."""
        playlist = []
        for file in self.fild_all_files(playdir):
            ext = splitext(basename(file))[1][1:].lower()  # ピリオドを抜いた拡張子
            if ext in self.settei.settings and self.settei.settings[ext]:
                playlist.append(file.replace('/', os.sep))
        return playlist



if __name__ == '__main__':
    u"""メイン."""
    if len(sys.argv) > 1:
        settei = Settei()
        cui_ctrl = CuiController(settei)
    else:
        app = QtGui.QApplication(sys.argv)
        settei = Settei()
        main_view = MainView()
        main_ctrl = MainController(settei, main_view)
        main_view.show()
        sys.exit(app.exec_())
