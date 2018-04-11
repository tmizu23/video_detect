# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from MyUtil import find_exedir

class BackgroundSubtractor:

    def __init__(self, learningrate, skip=0):
        u"""パラメータの初期化."""
        self.learningrate = learningrate
        self.fgbg = cv2.createBackgroundSubtractorKNN(history=0,
           dist2Threshold=20, detectShadows=0)  # 背景差分設定
        #self.fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        self.skip = skip
        self.count = 0  # ビデオの最初をスキップ
        self.bframe = None
        self.oldx = self.oldy = self.oldw = self.oldh = 0  # 動体の最大輪郭の位置
        self.hits = [0] * 5  # ヒット（動物の可能性）したからどうか.過去5フレームを記録
        self.hitmax = 3  # 過去5フレームの内、何フレームがhitだと検知とするか
        self.detecttype = None  # 動物の可能性を判断した条件
        self.car_cascade = cv2.CascadeClassifier(find_exedir()+ os.sep + "data/cars.xml")
        # self.animal_cascade = cv2.CascadeClassifier(find_exedir()+ os.sep + "data/animal_cascade.xml")

    def apply(self, gframe):
        u"""背景差分の画像更新."""
        self.draw_frame = np.zeros((gframe.shape[0], gframe.shape[1], 3))
        bframe = self.fgbg.apply(gframe, learningRate=self.learningrate)
        if self.count > self.skip:
            if bframe is None:
                return None
            # 膨張・収縮でノイズ除去(設置環境に応じてチューニング)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            bframe = cv2.morphologyEx(bframe, cv2.MORPH_OPEN, kernel)
            bframe = cv2.morphologyEx(bframe, cv2.MORPH_DILATE, kernel)
            bframe = cv2.morphologyEx(bframe, cv2.MORPH_CLOSE, kernel)
            self.bframe = bframe

        else:
            self.count += 1
            self.bframe = None
        return bframe

    def init_learning(self, video):
        u"""背景画像の学習."""
        video.cap.set(cv2.CAP_PROP_POS_FRAMES, video.framecount - 51)
        """
        # 検知範囲がビデオ内にあるかチェック。
        ret, frame = video.cap.read()
        frame = cv2.resize(frame,(video.width, video.height))
        gframe = frame[video.detect_top:video.detect_bottom,
                       video.detect_left:video.detect_right]
        if gframe.shape[0] == 0 or gframe.shape[1] == 0:
           print("\nWARNING!! Detection area is outside of video.\n")
        """
        # 後半50フレームで学習
        for i in range(50):
            ret, frame = video.cap.read()
            rframe = cv2.resize(frame, (video.width, video.height))
            # 画像の前処理
            cframe = rframe[video.detect_top:video.detect_bottom,
                            video.detect_left:video.detect_right]  # フレームの検知範囲を設定
            gframe = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)  # グレーに変換
            # gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
            gframe = cv2.equalizeHist(gframe)  # ヒストグラム平坦化（明るさの変化を抑えるため）
            self.fgbg.apply(gframe, learningRate=0.02)
        video.cap.set(cv2.CAP_PROP_POS_FRAMES, video.curpos)

    def detect(self, gframe):
        u"""検知処理."""
        state = "NODETECT"
        animals = None
        cars = None
        exist = False
        crop_frame = None
        x_max = 0
        y_max = 0
        w_max = 0
        h_max = 0
        if self.bframe is not None:

            # 動体の輪郭抽出
            _, cnts, _ = cv2.findContours(
                self.bframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                # 動体があれば。（背景差分適応の1フレームあとから開始、又はWebCam）

                # 動体の輪郭が最大のものに対して動物かどうかを判別する
                areas = [cv2.contourArea(cnt) for cnt in cnts]
                cnt_max = [cnts[areas.index(max(areas))]][0]
                ### 動体全部を大きい順に並び替えて、最大を選ぶ場合.最大以外も処理したい場合 ###
                # (cnts, areas) = zip(*sorted(zip(cnts, areas),key=lambda a:a[1], reverse=True))
                # cnt_max = cnts[0]
                (x_max, y_max, w_max, h_max) = cv2.boundingRect(cnt_max)
                #box_all.append((x_max, y_max, w_max, h_max))
                #box_grouped, _ = cv2.groupRectangles(box_all, 1, 0.8)
                state = self.check_detection(x_max, y_max, w_max, h_max)
                if state == "DETECT":  # 動体検知していたら更に車、動物のチェック
                    crop_frame = self.make_crop(gframe, x_max, y_max, w_max, h_max)
                    cars = self.check_cars(crop_frame)
                    if cars is not None:
                        bounding_color = (0, 255, 255)
                        cv2.rectangle(self.draw_frame, (x_max, y_max), (x_max + w_max, y_max + h_max), bounding_color, 2)

                    # animals = self.check_animals(crop_frame)
                    # # 動物いる！
                    # if cars is None and animals is not None:
                    #     #self.crop_frame = self.make_crop(frame, x_max+self.detect_left, y_max+self.detect_top, w_max, h_max)
                    #     exist = True
                # oldに位置を保存
                self.oldx = x_max
                self.oldy = y_max
                self.oldw = w_max
                self.oldh = h_max
            else:  # 検知しなければ
                self.oldx = 0
                self.oldy = 0
                self.oldw = 0
                self.oldh = 0
        self.exist = exist
        return state #, x_max, y_max, w_max, h_max

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
    #
    # def check_animals(self, crop_frame):
    #     # 動物のHarr-like検知
    #     animals = self.animal_cascade.detectMultiScale(crop_frame, 1.1, 1)
    #     if len(animals) == 0:
    #         return None
    #     else:
    #         return animals

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
        else:
            state = "NODETECT"
        return state

    def draw(self, cframe):
        # 動体の輪郭が最大のものを描画（検知なら白、それ以外は青色）
        # if self.bframe is not None:
        #     cframe = cv2.cvtColor(self.bframe, cv2.COLOR_GRAY2BGR)

        if self.exist is not None:
            bounding_color = (0, 255, 255)
        elif self.state == "DETECT":
            bounding_color = (255, 255, 255)
        else:
            bounding_color = (255, 0, 0)
        cframe = cv2.rectangle(cframe, (self.oldx, self.oldy), (self.oldx +
                                                       self.oldw, self.oldy + self.oldh), bounding_color, 2)
        if hasattr(self, "draw_frame"):
            # draw_frameを重ね合わせ（黒以外を置き換える）
            drawfilter = np.where((self.draw_frame[:, :, 0] != 0) | (
                self.draw_frame[:, :, 1] != 0) | (self.draw_frame[:, :, 2] != 0))
            cframe[drawfilter] = self.draw_frame[drawfilter]

        return cframe
