# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import datetime
from os.path import join, splitext, basename
import codecs

class Video():
    u"""ビデオ解析用クラス.背景差分法により動体を抽出する.フレーム間での動体の動き方によって
    動物かどうか検知する.動物と検知した動画、静止画を出力する.動画全体の動体の特徴量から
    機械学習モデルで動物かどうかの判定を行う."""

    def __init__(self, playfile, outdir, webcam=False):
        u"""ビデオ初期設定."""
        # 再生ファイル、出力フォルダ、映像サイズ、映像フレーム数、 fps、検知範囲 、現在位置、領域出力するかどうか、ログ出力するかどうかの設定
        if webcam:
            self.webcam = True
            self.learningrate = 0  # 1.0/90  # 背景差分の更新頻度
        else:
            self.webcam = False
            self.learningrate = 1.0 / 500  # 150 #1.0/90 # 0だとだめ
        self.learned = False
        self.playfile = playfile
        self.cap = cv2.VideoCapture(self.playfile)
        self.outdir = outdir
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.cap.set(cv2.CAP_PROP_FPS,60)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.detect_left = 0
        self.detect_top = 0
        self.detect_right = self.width
        self.detect_bottom = self.height
        self.curpos = 0
        self.bounding = True
        self.verbose = True
        self.fgbg = cv2.createBackgroundSubtractorKNN(
            dist2Threshold=800, detectShadows=0)  # 背景差分設定
        self.oldx = self.oldy = self.oldw = self.oldh = 0  # 動体の最大輪郭の位置
        self.hits = [0] * 5  # ヒット（動物の可能性）したからどうか.過去5フレームを記録
        self.hitmax = 3  # 過去5フレームの内、何フレームがhitだと検知とするか
        self.state = "NODETECT"  # 検知中かどうか
        self.detecttype = None  # 動物の可能性を判断した条件
        self.BEFORE_VIDEO_LENGTH = 90  # 検知前の動画を保存するフレーム数
        self.AFTER_VIDEO_LENGTH = 300  # 検知後の動画を保存するフレーム数
        self.video_count = -1  # 動画保存の残りフレーム数
        self.frames = []  # 検知前の動画を一時的に保存しておく
        self.writing = False  # 検知して動画を書き込み中かどうか
        self.videoWriter = None  # ビデオ書き出し用オブジェクト
        self.videoWriter_webcam = None  # webcamの全映像書き出し用オブジェクト
        self.webcam_savetime = None  # webcamの映像保存開始時間
        self.frame = None  # 現在のフレーム画像
        self.crop_frame = None  # 検知エリアの切り出しフレーム
        self.exist = False  # ビデオに動物がいるかどうか
        self.func = None  # GUIにログを通知するための関数（observerモデル）
        self.INTERVAL = 60  # 検知間隔フレーム数
        self.interval_count = 0  # 検知間隔のカウント
        self.cars = None  # 車の検知結果
        self.animals = None  # 動物の検知結果
        self.car_cascade = cv2.CascadeClassifier("car_cascade.xml")
        self.animal_cascade = cv2.CascadeClassifier("animal_cascade.xml")
        #self.tanuki_cascade = cv2.CascadeClassifier("tanuki_cascade.xml")


    def init_learning(self):
        u"""背景画像の学習."""
        print("main_model:init_learning")
        self.learned = True
        if not self.webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.framecount - 51)
            # 検知範囲がビデオ内にあるかチェック。
            ret, frame = self.cap.read()
            gframe = frame[self.detect_top:self.detect_bottom,
                           self.detect_left:self.detect_right]
            if gframe.shape[0] == 0 or gframe.shape[1] == 0:
               print("\nWARNING!! Detection area is outside of video.\n")

            # 後半50フレームで学習
            for i in range(50):
                ret, frame = self.cap.read()
                # 画像の前処理
                gframe = frame[self.detect_top:self.detect_bottom,
                               self.detect_left:self.detect_right]  # フレームの検知範囲を設定
                gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)  # グレーに変換
                # gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
                gframe = cv2.equalizeHist(gframe)  # ヒストグラム平坦化（明るさの変化を抑えるため）
                self.fgbg.apply(gframe, learningRate=0.02)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curpos)

    def process_nextframe(self):
        u"""ビデオフレーム読み込み＆処理."""
        fgmask = None
        self.cursec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        self.curpos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.state = "NODETECT"
        self.animals = None
        self.cars = None

        ok, frame = self.cap.read()
        if not ok:
            return False

        # 最初の中央マークが消えるタイミングで背景差分の検知を開始（中央マークの変化に反応してしまうため）
        if self.curpos > 2 or self.webcam:
            # 画像の前処理
            gframe = frame[self.detect_top:self.detect_bottom,
                           self.detect_left:self.detect_right]  # フレームの検知範囲を設定
            gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)  # グレーに変換
            # gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
            gframe = cv2.equalizeHist(gframe)  # ヒストグラム平坦化（明るさの変化を抑えるため）
            # 背景差分適応画像(2値)からの検知処理
            fgmask = self.fgbg.apply(gframe, learningRate=self.learningrate)
            if fgmask is not None:
                # 膨張・収縮でノイズ除去(設置環境に応じてチューニング)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

                ##thframe = cv2.bitwise_and(gframe, gframe, mask=fgmask)
                # メディアンフィルタでノイズ除去
                ## fgmask = cv2.GaussianBlur(fgmask, (41, 41), 0)
                ## fgmask = cv2.medianBlur(fgmask,49)


                # 動体の輪郭抽出
                _, cnts, _ = cv2.findContours(
                    fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if 0 < len(cnts) and (self.curpos > 3 or self.webcam):
                    # 動体があれば。（背景差分適応の1フレームあとから開始、又はWebCam）

                    # 動体の輪郭が最大のものに対して動物かどうかを判別する
                    areas = [cv2.contourArea(cnt) for cnt in cnts]
                    cnt_max = [cnts[areas.index(max(areas))]][0]
                    ### 動体全部を大きい順に並び替えて、最大を選ぶ場合.最大以外も処理したい場合 ###
                    # (cnts, areas) = zip(*sorted(zip(cnts, areas),key=lambda a:a[1], reverse=True))
                    # cnt_max = cnts[0]
                    (x_max, y_max, w_max, h_max) = cv2.boundingRect(cnt_max)

                    # 検知してからINTERVALの回数の間は、検知処理しない。
                    if self.interval_count >= 0:
                        self.interval_count -= 1   # 検知するまでカウントダウンし続ける
                    else:  # 0未満になっているときは検知チェック
                       self.state = self.check_detection(x_max, y_max, w_max, h_max)
                       if self.state == "DETECT":  # 動体検知していたら更に車、動物のチェック
                           crop_frame = self.make_crop(gframe, x_max, y_max, w_max, h_max)
                           self.cars = self.check_cars(crop_frame)
                           self.animals = self.check_animals(crop_frame)
                           ## 動物いる！
                           if self.cars is None and self.animals is not None:
                              self.crop_frame = self.make_crop(frame, x_max+self.detect_left, y_max+self.detect_top, w_max, h_max)
                              self.exist = True
                              print("\nTrue Detect")
                              self.interval_count = self.INTERVAL
                    # oldに位置を保存
                    self.oldx = x_max
                    self.oldy = y_max
                    self.oldw = w_max
                    self.oldh = h_max
                else: #検知しなければ
                    self.oldx = 0
                    self.oldy = 0
                    self.oldw = 0
                    self.oldh = 0

        if self.bounding:
            # 検知範囲を描画（赤色）
            cv2.rectangle(frame, (self.detect_left, self.detect_top), (
                self.detect_right - 2, self.detect_bottom - 2), (0, 0, 255), 2)
            # 動体の輪郭が最大のものを描画（検知なら白、それ以外は青色）
            if self.animals is not None:
                bounding_color = (0, 255, 255)
            elif self.state == "DETECT":
                bounding_color = (255, 255, 255)
            else:
                bounding_color = (255, 0, 0)
            cv2.rectangle(frame, (self.oldx+self.detect_left, self.oldy + self.detect_top), (self.oldx +self.detect_left+ self.oldw, self.oldy + self.detect_top + self.oldh), bounding_color, 2)
        # グレー画像を表示
        # if fgmask is not None:
        #    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        #    frame[self.detect_top:self.detect_bottom,
        #               self.detect_left:self.detect_right] = fgmask
        self.frame = frame
        # 最新フレーム(数秒)を一時的に保存
        self.frames.append(frame)
        if len(self.frames) > self.BEFORE_VIDEO_LENGTH:
            del self.frames[0]

        if self.verbose and not self.webcam:
            sys.stdout.write("\r{0}%".format(int(
                100.0 * self.curpos / self.framecount)))

        return True

    def make_crop(self, frame, x, y, w, h):
        crop_extend_scale = 2.0
        height = frame.shape[0]
        width = frame.shape[1]
        if (y + h * crop_extend_scale) < height:
            top = int(y + h * crop_extend_scale)
        else:
            top = height
        if (y - h * (crop_extend_scale - 1)) > 0:
            bottom = int(y - h * (crop_extend_scale - 1))
        else:
            bottom = 0
        if (x - w * (crop_extend_scale - 1)) > 0:
            left = int(x - w * (crop_extend_scale - 1))
        else:
            left = 0
        if (x + w * crop_extend_scale) < width:
            right = int(x + w * crop_extend_scale)
        else:
            right = width
        return frame[bottom:top, left:right].copy()

    def check_cars(self, crop_frame):
        # 車のHarr-like検知
        cars = self.car_cascade.detectMultiScale(crop_frame, 1.1, 1)
        if len(cars) == 0:
            return None
        else:
            return cars

    def check_animals(self, crop_frame):
        # 動物のHarr-like検知
        animals = self.animal_cascade.detectMultiScale(crop_frame, 1.1, 1)
        if len(animals) == 0:
            return None
        else:
            return animals

    def check_hit(self, x, y, w, h):
        u"""輪郭の大きさと動きから動物の可能性を判断(タイプはA,B,C)."""
        hit = 0
        if not np.isnan(self.oldx):
            # ある程度の大きさのものがゆっくり動く（中型～大型）大きすぎると環境光、まったく動かないものはダメ
            if ((150 < w + h < 400) and (abs(x + w / 2.0 - (self.oldx + self.oldw / 2.0)) >= 0.5 or abs(y + h / 2.0 - (self.oldy + self.oldh / 2.0)) >= 0.5)):
                self.detecttype = "A"
                hit = 1
            # 当たり判定（1フレーム前の輪郭と重なる部分があること）がある条件で、大きく動く（鳥の飛翔、走る動物）
            elif (4 <= abs(x + w / 2.0 - (self.oldx + self.oldw / 2.0)) < 10 or 4 <= abs(y + h / 2.0 - (self.oldy + self.oldh / 2.0)) < 10) and (abs(x - self.oldx) < (w / 2 + self.oldw / 2)) and (abs(y - self.oldy) < (h / 2 + self.oldh / 2)):
                self.detecttype = "B"
                hit = 1
            # 当たり判定がある範囲で、小さいものがものすごく大きく動く(すぐ消えるからhitを加算してゆるく判定)。1フレーム前の判定がBかCのときに限る（大きさが変化するのはおかしいから）
            elif(self.detecttype == "B" or self.detecttype == "C") and (w + h < 200) and (abs(x + w / 2.0 - (self.oldx + self.oldw / 2.0)) >= 10 or abs(y + h / 2.0 - (self.oldy + self.oldh / 2.0)) >= 10) and (abs(x - self.oldx) < (w / 2 + self.oldw / 2)) and (abs(y - self.oldy) < (h / 2 + self.oldh / 2)):
                self.detecttype = "C"
                hit = 2
        return hit

    def check_detection(self, x, y, w, h):
        u"""動体検知最終判断."""
        # hitの確認＆更新
        hit = self.check_hit(x, y, w, h)
        self.hits.append(hit)
        if len(self.hits) > 5:
            del self.hits[0]
        # 検知の確認。過去5フレームの内、3フレームがhitだと検知とする。
        state = "NODETECT"
        if sum(self.hits) >= self.hitmax:
            state = "DETECT"
            self.hits = [0] * 5
            if self.verbose:
                if self.webcam:
                    timestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    sys.stdout.write(
                        "\r    detect! --> {0} {1}".format("webcam", timestr))
                else:
                    sys.stdout.write(
                        "\r    detect! --> {0} {1}sec".format(basename(self.playfile), self.cursec))
        else:
            state = "NODETECT"
        return state

    def close_video(self):
        u"""動画書き込みのクローズ."""
        if self.videoWriter is not None:
            self.writing = False
            self.videoWriter.release()
            self.videoWriter = None
        if self.videoWriter_webcam is not None:
            self.videoWriter_webcam.release()
            self.videoWriter_webcam = None

    def writeout_webcam(self):
        u"""webcamの全動画書き込み."""
        if(self.videoWriter_webcam is None):
            self.webcam_savetime = datetime.datetime.now()
            filename = "ALLwebcam" + \
                self.webcam_savetime.strftime('%Y%m%d_%H%M%S') + '.mov'
            outfile = join(self.outdir, filename)
            fps = 20  # 処理時間の遅延を考慮するとこれぐらいで、だいたい現実時間と一緒。マシンスペックに依存する？
            self.videoWriter_webcam = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v'), fps, (self.width, self.height))
        else:
            self.videoWriter_webcam.write(self.frame)
            # 60分たったら次のファイル
            elapsed_time = (datetime.datetime.now() -
                            self.webcam_savetime).seconds / 60
            if(elapsed_time >= 60):
                self.videoWriter_webcam.release()
                self.videoWriter_webcam = None

    def writeout_video(self):
        u"""動画書き込み."""
        if (self.state == "DETECT" and self.cars is None) or self.writing:
            # 検知もしくは書き込み中なら
            if self.writing:
                # 動画書き込み中なら
                self.videoWriter.write(self.frame)
                self.video_count -= 1
                if self.video_count < 0:  # 書き込み終了
                    self.writing = False
                    self.videoWriter.release()
                    self.videoWriter = None
            else:
                # 動画書き込み中でなければ
                if not os.path.exists(self.outdir):
                    os.makedirs(self.outdir)
                self.writing = True
                self.video_count = self.AFTER_VIDEO_LENGTH
                if self.webcam:
                    filename = "webcam" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.mov'
                    fps = 20
                else:
                    filename = splitext(basename(self.playfile))[
                        0] + '_' + str(self.cursec).zfill(5) + '.mov'
                    fps = self.fps
                outfile = join(self.outdir, filename)
                self.videoWriter = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v'), fps, (self.width, self.height))
                # まず、検知直前30フレームを書き出す
                for v in self.frames:
                    self.videoWriter.write(v)

    def writeout_jpg(self):
        u"""jpg書き出し."""
        # 車でなく動物検知したら書き出し
        if self.animals is not None and self.cars is None:
            dt = datetime.datetime.now()
            self.webcam_savetime = dt.strftime('%Y%m%d_%H%M%S')
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            if self.webcam:
                filename = "webcam" + self.webcam_savetime + '.jpg'
                filename_crop = "webcam" + self.webcam_savetime + '_crop.jpg'
            else:
                filename = splitext(basename(self.playfile))[
                    0] + '_' + str(self.cursec).zfill(5) + '.jpg'
                filename_crop = splitext(basename(self.playfile))[
                    0] + '_' + str(self.cursec).zfill(5) + '_crop.jpg'
            # opencv3(+python3)だと日本語(cp932)だめ
            # https://github.com/opencv/opencv/issues/4292
            outfile = join(self.outdir, filename)
            cv2.imwrite(outfile, self.frame)

            outfile_crop = join(self.outdir, filename_crop)
            cv2.imwrite(outfile_crop, self.crop_frame)


    def writeout_result(self, outfile):
        u"""ログ書き出し."""
        if not os.path.exists(outfile):
            datastr = ",".join(
                [u"ファイル名", u"検知", u"撮影時刻(24h)"])
            f = codecs.open(outfile, 'w', 'cp932')
            f.write(datastr + '\n')
            f.close()
            logstr = datastr.replace(',', '\t') + '\n'
            self.notify(logstr)

        # ToDo 昼夜の判定は、ファイルの作成時刻から。入れるなら。
        dt = datetime.datetime.fromtimestamp(os.stat(self.playfile).st_mtime)
        hour = dt.strftime('%H')
        datastr = ",".join(
            [self.playfile, str(self.exist), hour])
        f = codecs.open(outfile, 'a', 'cp932')
        f.write(datastr + '\n')
        f.close()
        logstr = datastr.replace(',', '\t') + '\n'
        self.notify(logstr)

    def check_webcam(self):
        u"""Webcamがあるか確認"""
        ret = True
        if not self.cap.isOpened():
            ret = False
        return ret

    def attach(self, func):
        u"""notifyで通知する関数."""
        self.func = func

    def notify(self, str):
        u"""関数にstrを通知.GUIでのlogの表示に使用."""
        if self.func is not None:
            self.func(str)

    def set_detectionarea(self, top, bottom, left, right):
        u"""検知範囲の設定."""
        self.detect_top = top
        self.detect_bottom = bottom
        self.detect_left = left
        self.detect_right = right
        self.learned = False

    def set_outdir(self, outdir):
        u"""出力フォルダの設定."""
        self.outdir = outdir

    def set_verbose(self, verbose):
        u"""ログ出力の設定."""
        self.verbose = verbose

    def set_bounding(self, bounding):
        u"""検知領域出力の設定."""
        self.bounding = bounding

    def get_framecount(self):
        u"""フレーム数を返す."""
        return self.framecount

    def get_fps(self):
        u"""fpsを返す."""
        return self.fps

    def get_size(self):
        u"""ビデオの高さ、幅を返す."""
        return self.height, self.width

    def set_position(self, pos):
        u"""再生位置を設定."""
        self.curpos = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def get_position(self):
        u"""再生位置を返す."""
        return self.curpos

    def get_RGBframe(self):
        u"""ビデオフレーム画像（opencv）からQPixmap形式に変換."""
        # Pixmap用にBGRからRGBに入れ替え
        try:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            return frame
        except:
            return None
