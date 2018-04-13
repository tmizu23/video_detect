# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import math
import datetime
from os.path import join, splitext, basename
from BackgroundSubtractor import BackgroundSubtractor


class Video():
    u"""ビデオ解析用クラス.背景差分法により動体を抽出する.フレーム間での動体の動き方によって
    動物かどうか検知する.動物と検知した動画、静止画を出力する.動画全体の動体の特徴量から
    機械学習モデルで動物かどうかの判定を行う."""

    def __init__(self, playfile, outdir, logfunc=None, webcam=False):
        u"""ビデオ初期設定."""
        # 再生ファイル、出力フォルダ、映像サイズ、映像フレーム数、 fps、検知範囲 、現在位置、領域出力するかどうか、ログ出力するかどうかの設定
        self.webcam = webcam
        self.playfile = playfile
        self.cap = cv2.VideoCapture(self.playfile)
        self.FPS = 15  # 強制的にこのfpsにしてビデオを読み書き
        if self.webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.cap.set(cv2.CAP_PROP_FPS, self.FPS)
        self.outdir = outdir
        self.imgscale = 1.0
        self.org_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.org_width * self.imgscale)  # 処理サイズ
        self.height = int(self.org_height * self.imgscale)  # 処理サイズ
        self.detect_mask = np.ones((self.height, self.width),dtype=np.uint8) # 検知範囲のマスク
        self.framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.detect_left = 0
        self.detect_top = 0
        self.detect_right = self.width
        self.detect_bottom = self.height
        self.curpos = 0
        self.bounding = True
        self.crop = True
        self.verbose = True
        self.state = "NODETECT"  # 検知中かどうか
        self.INTERVAL = 0  # self.FPS*10 # 検知間隔フレーム数(10秒、記録が連続になるように)
        self.interval_count = 0  # 検知間隔のカウント
        self.BEFORE_VIDEO_LENGTH = self.FPS * 10  # 検知前の動画を保存するフレーム数(10秒)
        self.AFTER_VIDEO_LENGTH = self.FPS * 5  # 検知後の動画を保存するフレーム数(5秒)
        self.video_count = -1  # 動画保存の残りフレーム数
        self.frames = []  # 検知前の動画を一時的に保存しておく
        self.times = []  # fps計算のため現在時刻を一時的に保存しておく
        self.states = []  # bsの検知状況を一時的に保存しておく.jpg書き出しのため
        self.writing = False  # 検知して動画を書き込み中かどうか
        self.waiting = False
        self.videoWriter = None  # ビデオ書き出し用オブジェクト
        self.videoWriter_webcam = None  # webcamの全映像書き出し用オブジェクト
        self.webcam_savetime = None  # webcamの映像保存開始時間
        self.frame = None  # 現在のフレーム画像
        self.bboxe = None  # 現在の検知エリア
        self.view_frame = None  # 表示用フレーム
        self.crop_frame = None  # 検知エリアの切り出しフレーム
        self.func = logfunc  # GUIにログを通知するための関数（observerモデル）
        self.detecttype = None
        self.m = 0  # fps調整用変数
        self.bs = None  # BackgroundSubtractor
        self.notify("{}\nsize:{}×{} fps:{} frame:{}\n".format(self.playfile,
            self.org_width, self.org_height, self.video_fps, self.framecount))

    def set_detecttype(self, detecttype):
        u"""表示画像の設定."""
        self.detecttype = detecttype

        if self.webcam:
            learningrate = 1.0 / 10  # 背景差分の更新頻度
        else:
            learningrate = 1.0 / 10  # 150 #1.0/90 # 0だとだめ
        self.bs = BackgroundSubtractor(learningrate, skip=5)

    def get_currentframe(self, bounding):
        ok, frame = self.cap.read()
        rframe = cv2.resize(frame, (self.width, self.height))
        cv2.rectangle(rframe, (self.detect_left, self.detect_top), (
            self.detect_right - 2, self.detect_bottom - 2), (0, 0, 255), 2)
        frame = cv2.cvtColor(rframe, cv2.COLOR_BGR2RGB)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curpos)
        return frame

    def process_nextframe(self):
        u"""ビデオフレーム読み込み＆処理."""
        self.state = "NODETECT"
        self.curmsec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.curpos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        bbox = None
        ok, frame = self.cap.read()
        if not ok:
            return False, False
        # fpsがself.FPSになるように調整。
        if not self.webcam:
            #print("{},{},{},{}".format(self.curpos,self.video_fps,self.m,self.curpos % self.video_fps))
            if (self.curpos % self.video_fps) < 1:  # ==0だったけど、29.97対応で<1に変更。
                self.m = 0
            if (self.curpos % self.video_fps) < self.m:
                return True, True
            self.m = self.m + (self.video_fps / self.FPS)
        # リサイズ
        rframe = cv2.resize(frame, (self.width, self.height))
        gframe = cv2.bitwise_and(rframe,rframe,mask=self.detect_mask) #検知範囲でマスク
        # グレー化&ヒストグラム平坦化（明るさの変化を抑えるため）
        gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)
        gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
        gframe = cv2.equalizeHist(gframe)

        ##########
        # 更新処理
        ##########
        # 背景差分
        bframe = self.bs.apply(gframe)

        ##########
        # 検知処理
        ##########
        # 検知してからINTERVALの回数の間は、検知処理しない。
        if self.state == "DETECT" and self.interval_count >= 0:
            self.interval_count -= 1
        else:
            self.interval_count = self.INTERVAL
            self.state,bbox = self.bs.detect(gframe)
            bbox = (np.array(bbox) / self.imgscale).astype(np.int32)

        # ログ表示
        if self.verbose and self.state == "DETECT":
            if self.webcam:
                timestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                print(
                    "\n    detect! --> animal:{} {}".format("webcam", timestr))
            else:
                cursec = self.curmsec / 1000
                h = math.floor(cursec / 3600)
                m = math.floor((cursec - h * 3600) / 60)
                s = cursec - h * 3600 - m * 60
                print(
                    "\n    detect! --> animal: {0} {1:02d}h{2:02d}m{3:04.1f}s".format(basename(self.playfile), h, m, s))
        ##########
        # 表示処理
        ##########
        # 検知表示
        if self.detecttype == "detectA":
            img = self.bs.draw(rframe)
        # 差分表示
        if self.detecttype == "detectB":
            img = cv2.cvtColor(bframe, cv2.COLOR_GRAY2BGR)
        # 処理画像グレー表示
        if self.detecttype == "detectC":
            img = cv2.cvtColor(gframe, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img, (self.detect_left, self.detect_top), (
            self.detect_right - 2, self.detect_bottom - 2), (0, 0, 255), 2)

        # 出力画像に検知範囲を描画する場合
        if self.bounding:
            if self.state == "DETECT":
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] +bbox[2],bbox[1]+bbox[3]),(0, 255, 255),2)
            cv2.rectangle(frame, (int(self.detect_left/self.imgscale), int(self.detect_top/self.imgscale)), (
                int(self.detect_right/self.imgscale) - 2, int(self.detect_bottom/self.imgscale) - 2), (0, 0, 255), 2)


        # ウェブカメの場合反転（内向きカメラ用）
        # if self.webcam:
        #    rframe = cv2.flip(rframe, 1)

        self.frame = frame  # 書き出し用
        self.bbox = bbox  #書き出し用
        self.view_frame = img  # 表示用
        #cv2.imshow("debug",self.view_frame)
        # 最新フレーム(数秒)を一時的に保存
        self.frames.append(self.frame)
        self.times.append(cv2.getTickCount())
        self.states.append(self.state) #BSでの検知状況.jpg書き出し用
        if len(self.frames) > self.BEFORE_VIDEO_LENGTH + self.AFTER_VIDEO_LENGTH:
            del self.frames[0]
            del self.times[0]
            del self.states[0]
        # 進行状況表示
        # if self.verbose and not self.webcam:
        #     print("\r{0}%".format(int(
        #         100.0 * self.curpos / self.framecount)))

        return True, False

    def close_video(self):
        u"""動画書き込みのクローズ."""
        self.video_count = -1
        self.writeout_video()
        # if self.videoWriter_webcam is not None:
        #     self.videoWriter_webcam.release()
        #     self.videoWriter_webcam = None
        self.bs = None

    def writeout_webcam(self):
        u"""webcamの全動画書き込み."""
        if(self.videoWriter_webcam is None):
            self.webcam_savetime = datetime.datetime.now()
            filename = "ALLwebcam" + \
                self.webcam_savetime.strftime('%Y%m%d_%H%M%S') + '.mov'
            outfile = join(self.outdir, filename)
            self.videoWriter_webcam = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v'), self.FPS, (self.width, self.height))
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
        if self.state == "DETECT" or self.waiting:
            # 録画時間分フレームに保存して、最後に書き出す。
            if self.waiting:
                self.video_count -= 1
            else:
                self.video_count = self.AFTER_VIDEO_LENGTH
                self.waiting = True
            if self.video_count < 0:  # 待ち時間終了
                self.waiting = False
                if not os.path.exists(self.outdir):
                    os.makedirs(self.outdir)
                if self.webcam:
                    filename = "webcam" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.mov'
                    fps = round(
                        len(self.frames) / ((self.times[-1] - self.times[0]) / cv2.getTickFrequency()), 1)
                else:
                    cursec = self.curmsec / 1000
                    h = math.floor(cursec / 3600)
                    m = math.floor((cursec - h * 3600) / 60)
                    s = cursec - h * 3600 - m * 60
                    filename = "{0}_{1:02d}h{2:02d}m{3:03.1f}s.mov".format(splitext(basename(self.playfile))[
                        0], h, m, s)
                    fps = self.FPS
                try:
                    outfile = join(self.outdir, filename)
                    self.videoWriter = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
                        'm', 'p', '4', 'v'), fps, (self.org_width, self.org_height))
                    # 書き出す
                    for v in self.frames:
                        self.videoWriter.write(v)
                    self.videoWriter.release()
                    self.videoWriter = None
                except:
                    print("writing video error!")

    def writeout_jpg(self):
        u"""jpg書き出し."""
        # 検知したら書き出し
        if self.state == "DETECT":
            dt = datetime.datetime.now()
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)


            if self.webcam:
                self.webcam_savetime = dt.strftime('%Y%m%d_%H%M%S')
                filename = "webcam{0}.jpg".format(
                    self.webcam_savetime)
                # filename_crop = "webcam{0}_crop.jpg".format(
                #     self.webcam_savetime)
                filename_txt = "webcam{0}.txt".format(
                    self.webcam_savetime)
            else:
                cursec = self.curmsec / 1000
                h = math.floor(cursec / 3600)
                m = math.floor((cursec - h * 3600) / 60)
                s = cursec - h * 3600 - m * 60

                filename = "{0}_{1:02d}h{2:02d}m{3:04.1f}s.jpg".format(
                    splitext(basename(self.playfile))[0], h, m, s)
                # filename_crop = "{0}_{1:02d}h{2:02d}m{3:02.1f}s_crop.jpg".format(
                #     splitext(basename(self.playfile))[0], h, m, s)
                filename_txt = "{0}_{1:02d}h{2:02d}m{3:04.1f}s.txt".format(
                    splitext(basename(self.playfile))[0], h, m, s)

            # opencv3(+python3)だと日本語(cp932)だめ
            #cv2.imwrite(outfile, self.frame)
            #cv2.imwrite(outfile_crop, self.crop_frame)

            # https://github.com/opencv/opencv/issues/4292
            # とりあえずimencodeで代替策
            try:
                outfile = join(self.outdir, filename)
                with open(outfile, 'wb') as f:
                    ret, buf = cv2.imencode('.jpg', self.frame, [
                                            int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    f.write(np.array(buf).tostring())

                # クロップ情報
                if self.crop:
                    bbox = self.bbox
                    x = bbox[0]+round(bbox[2]/2)
                    y = bbox[1]+round(bbox[3]/2)
                    w = bbox[2]
                    h = bbox[3]
                    crop_extend_scale = 1.2
                    classno = 0
                    bounding_str = "{} {} {} {} {}".format(classno, x / self.org_width, y / self.org_height,
                                                           w * crop_extend_scale / self.org_width,
                                                           h * crop_extend_scale / self.org_height)
                    outfile_txt = join(self.outdir, filename_txt)
                    with open(outfile_txt, 'w') as f:
                        f.write(bounding_str)
                    # bbox = self.bbox
                    # crop_frame = self.bs.make_crop(self.frame, bbox[0], bbox[1], bbox[2], bbox[3])
                    # outfile_crop = join(self.outdir, filename_crop)
                    # with open(outfile_crop, 'wb') as f:
                    #     ret, buf = cv2.imencode('.jpg', crop_frame, [
                    #         int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    #     f.write(np.array(buf).tostring())
            except:
                print("writing jpg error!")



    def check_webcam(self):
        u"""Webcamがあるか確認"""
        ret = True
        if not self.cap.isOpened():
            ret = False
        return ret

    # def attach(self, func):
    #     u"""notifyで通知する関数."""
    #     self.func = func

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
        self.detect_mask *= 0
        self.detect_mask[self.detect_top:self.detect_bottom,
                        self.detect_left:self.detect_right] = 255

    def set_outdir(self, outdir):
        u"""出力フォルダの設定."""
        self.outdir = outdir

    def set_verbose(self, verbose):
        u"""ログ出力の設定."""
        self.verbose = verbose

    def set_imgscale(self, imgscale):
        u"""画像スケールの設定."""
        self.imgscale = imgscale
        self.width = int(self.org_width * self.imgscale)
        self.height = int(self.org_height * self.imgscale)
        self.detect_mask = np.zeros((self.height, self.width),dtype=np.uint8)

    def set_bounding(self, bounding):
        u"""検知領域出力の設定."""
        self.bounding = bounding

    def set_crop(self, crop):
        u"""検知領域のクロップ設定."""
        self.crop = crop

    def get_framecount(self):
        u"""フレーム数を返す."""
        return self.framecount

    def get_fps(self):
        u"""fpsを返す."""
        return self.video_fps

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
            frame = cv2.cvtColor(self.view_frame, cv2.COLOR_BGR2RGB)
            return frame
        except:
            return None
