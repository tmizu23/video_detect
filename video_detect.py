# -*- coding: utf-8 -*-

"""

Usage:
    video_detect [options] --inpdir=IN_PATH --outdir=OUT_PATH
    video_detect [options] --webcam --outdir=OUT_PATH

Detect animals from VIDEO file.

Arguments:
    IN_PATH               Input directory of VIDEO files.(recursive)
    OUT_PATH              Output directory

Options:
    -b                Draw bounding box
    -v                Save detected video
    -j                Save jpg image
    -g                Save graph plot image
    -d                Debug mode
    --avi             Input AVI video format
    --mov             Input MOV video format
    --mpg             Input MPG video format
    --mp4             Input MP4 video format
    --wmv             Input WMV video format
    --flv             Input FLV video format
    --mts             Input MTS video format
    --m2ts            Input M2TS video format
    --area=<top>,<bottom>,<left>,<right>    Detection area (top left is 0,0)
    --help       Show this screen.
    --version       Show version.

Examples:
    video_detect.py --inpdir=F://video --outdir=F://capture --avi -w -b -j- -area=0,680,100,1280

"""
#from __future__ import print_function
#from __future__ import unicode_literals
#from builtins import str
from docopt import docopt
import sys
import os
import cv2
import math
import datetime
from os.path import join, splitext, basename
import numpy as np
import json
import codecs
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # need for plot3d
from matplotlib import gridspec
from PyQt4 import QtGui, QtCore
from video_detectUI import Ui_MainWindow
from pandas import DataFrame
import pandas as pd

#import tweepy
#from keras.models import model_from_json
#from keras.preprocessing import sequence


class Video():
    u"""ビデオ解析用クラス.背景差分法により動体を抽出する.フレーム間での動体の動き方によって
    動物かどうか検知する.動物と検知した動画、静止画を出力する.動画全体の動体の特徴量から
    機械学習モデルで動物かどうかの判定を行う."""

    def __init__(self, playfile, outdir, webcam=False):
        u"""ビデオ初期設定."""
        # 再生ファイル、出力フォルダ、映像サイズ、映像フレーム数、 fps、検知範囲 、現在位置、領域出力するかどうか、ログ出力するかどうかの設定
        if webcam:
            self.webcam = True
            self.playfile = "webcam"
            self.learningrate = 0#1.0/90  # 背景差分の更新頻度
            self.cap = cv2.VideoCapture(1)
        else:
            self.webcam = False
            self.playfile = playfile
            self.learningrate = 1.0/500 #150 #1.0/90 # 0だとだめ
            self.cap = cv2.VideoCapture(self.playfile)
        self.outdir = outdir
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.cap.set(cv2.CAP_PROP_FPS,60)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("fps:" + str(self.fps) + "\n")
        self.detect_left = 0
        self.detect_top = 0
        self.detect_right = self.width
        self.detect_bottom = self.height
        self.curpos = 0
        self.bounding = True
        self.verbose = True
        # 背景差分動体
        #self.fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=500, detectShadows=0)  # 混合正規分布
        self.fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=0)  # 混合正規分布
        self.oldx = self.oldy = self.oldw = self.oldh = 0  # 動体の最大輪郭の位置
        self.detectarea_count_list = []  # 動体検知数の履歴
        self.detectarea_x_list = []  # 最大動体の位置とサイズの履歴
        self.detectarea_y_list = []
        self.detectarea_w_list = []
        self.detectarea_h_list = []
        self.detectarea_distance_list = []  # 最大動体位置間の距離の履歴
        self.plotcolor_list = []  # グラフ表示用のカラー設定の履歴.検知の状態による（検知、ヒット、なし）
        self.framenumber_list = []  # フレーム番号の履歴（グラフのX軸用）
        self.hit_list = []  # ヒットしたかどうかの履歴
        self.detect_list = []  # 検知したかどうかの履歴
        self.hits = [0] * 5  # ヒット（動物の可能性）したからどうか.過去5フレームを記録
        self.hitmax = 3  # 過去5フレームの内、何フレームがhitだと検知とするか
        self.state = "NODETECT"  # 検知中かどうか
        self.detecttype = None  # 動物の可能性を判断した条件
        self.frame_length_before_detection = 60  # 検知前の動画を保存するフレーム数
        self.frame_length_after_detection = 600  # 検知後の動画を保存するフレーム数
        self.remaining_frame_count = -1  # 動画保存の残りフレーム数
        self.frames = []  # 検知前の動画を一時的に保存しておく
        self.writing = False  # 検知して動画を書き込み中かどうか
        self.videoWriter = None  # ビデオ書き出し用オブジェクト
        self.videoWriter_webcam = None # webcamの全映像書き出し用オブジェクト
        self.webcam_savetime = None # webcamの映像保存開始時間
        self.frame = None  # 現在のフレーム画像
        self.daytime = "daytime"  # 動画が日中か夜か.init_learningで判定
        self.func = None  # GUIにログを通知するための関数（observerモデル）
        self.predict_animal = False  # 機会学習による予測をするかどうか
        if self.predict_animal:  # 機械学習モデルの設定
            model = os.path.abspath(os.path.dirname(__file__)) + os.sep + "cnn_model.json"
            param = os.path.abspath(os.path.dirname(__file__)) + os.sep + "cnn_model_weights.h5"
            self.set_learning_model(model, param)

        self.tracker = None #トラッキングオブジェクト
        self.car_cascade = cv2.CascadeClassifier("cars.xml")
        self.prev_gframe = None


    def init_learning(self):
        u"""背景画像の学習."""
        if not self.webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.framecount - 51)
            ret, frame = self.cap.read()
            # 昼夜の判定
            gframe = frame[self.detect_top:self.detect_bottom,
                           self.detect_left:self.detect_right]  # フレームの検知範囲を設定
            gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(gframe)  # 彩度
            if max(s.flatten()) > 128:
                self.daytime = "daytime"
            else:
                self.daytime = "night"
            for i in range(50):
                ret, frame = self.cap.read()
                # 画像の前処理
                gframe = frame[self.detect_top:self.detect_bottom,
                               self.detect_left:self.detect_right]  # フレームの検知範囲を設定
                gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)  # グレーに変換
                #gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
                gframe = cv2.equalizeHist(gframe)  # ヒストグラム平坦化（明るさの変化を抑えるため）
                self.fgbg.apply(gframe, learningRate=0.02)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curpos)

    def capture_nextframe(self):
        u"""ビデオフレーム読み込み＆処理."""
        fgmask = None
        self.cursec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        self.curpos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = self.cap.read()
        if ret is False:
            return None

        # 画像の前処理
        gframe = frame[self.detect_top:self.detect_bottom,
                       self.detect_left:self.detect_right]  # フレームの検知範囲を設定
        gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)  # グレーに変換
        #gframe = cv2.GaussianBlur(gframe, (5, 5), 0)  # 映像ノイズ削除
        gframe = cv2.equalizeHist(gframe)  # ヒストグラム平坦化（明るさの変化を抑えるため）
        ###オプティカルフロー###
        if self.prev_gframe is None:
            self.prev_gframe = gframe
            self.hsv = np.zeros_like(frame[self.detect_top:self.detect_bottom,
                           self.detect_left:self.detect_right])
            self.hsv[..., 1] = 255
        flow = cv2.calcOpticalFlowFarneback(self.prev_gframe, gframe, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[...,0] = ang*180/np.pi/2
        self. hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optframe = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        #cv2.imshow('optical',optframe)
        self.prev_gframe = gframe
        ###
        ###車のHarr-like検知
        #cars = self.car_cascade.detectMultiScale(gframe, 1.1, 1)
        #for (x,y,w,h) in cars:
        #    x = x + self.detect_left
        #    y = y + self.detect_top
        #    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        ###
        # 最初の中央マークが消えるタイミングで背景差分の検知を開始（中央マークの変化に反応してしまうため）
        if self.curpos > 2 or self.webcam:
            # 背景差分適応画像(2値)からの検知処理
            fgmask = self.fgbg.apply(gframe, learningRate=self.learningrate)
            if fgmask is not None:
                # 膨張・収縮でノイズ除去(設置環境に応じてチューニング)
                #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
                #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
                #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
                #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                #thframe = cv2.bitwise_and(gframe, gframe, mask=fgmask)

                ## メディアンフィルタでノイズ除去
                ## fgmask = cv2.GaussianBlur(fgmask, (41, 41), 0)
                ## fgmask = cv2.medianBlur(fgmask,49)

                if self.bounding:
                    # 検知範囲を描画（赤色）
                    cv2.rectangle(frame, (self.detect_left, self.detect_top), (
                        self.detect_right - 2, self.detect_bottom - 2), (0, 0, 255), 2)
                # 動体の輪郭抽出
                _, cnts, _ = cv2.findContours(
                    fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if 0 < len(cnts) and (self.curpos > 3 or self.webcam):
                    # 動体があれば。（背景差分適応の1フレームあとから開始、又はWebCam）

                    # 動体の輪郭が最大のものに対して動物かどうかを判別する
                    areas = [cv2.contourArea(cnt) for cnt in cnts]
                    (cnts, areas) = zip(*sorted(zip(cnts, areas),key=lambda a:a[1], reverse=True))
                    #cnt_max = [cnts[areas.index(max(areas))]][0]
                    cnt_max = cnts[0]
                    (x_max, y_max, w_max, h_max) = cv2.boundingRect(cnt_max)+ np.array([self.detect_left,self.detect_top,0,0])
                    ##
                    # 最大検知の輪郭線
                    #if self.bounding:
                        #hull = cv2.convexHull(cnt_max) + np.array([self.detect_left,self.detect_top])
                        #cv2.polylines(frame,[hull],True,(255,0,0),2)
                        #epsilon = 0.01*cv2.arcLength(cnt_max,True)
                        #approx = cv2.approxPolyDP(cnt_max,epsilon,True)
                        #cv2.fillConvexPoly(frame, approx,(0,0,255))


                    # hitしたか確認。最新の5フレームのhitの状況を保存
                    hit = self.check_hit(x_max, y_max, w_max, h_max)
                    if not self.webcam:
                        self.hit_list.append(hit)
                    # 検知したか確認
                    self.state = self.check_detection()
                    if not self.webcam:
                        self.detect_list.append(self.state)

                    if self.state == "DETECT": #and self.tracking is False:# and 700 < cv2.arcLength(cnt_max,True):
                        pass
                        #トラッキング
                        #self.tracker = cv2.MultiTracker("MIL")
                        #bboxes = [cv2.boundingRect(cnt) for cnt in cnts[0:1]]
                        #for bbox in bboxes:
                        #   self.tracker.add(fgmask,bbox)

                    # 動体の輪郭をすべて描画（緑色）
                    if self.bounding:
                        for c in cnts[0:3]:
                            x, y, w, h = cv2.boundingRect(c)
                            top = int(y+h*1.1) if (y+h*1.1) < len(gframe) else len(gframe)
                            bottom = int(y-h*0.1) if (y-h*0.1)>0 else 0
                            left = int(x-w*0.1) if (x-w*0.1)>0 else 0
                            right = int(x+w*1.1) if (x+w*1.1)<len(gframe[0]) else len(gframe[0])
                            #car_frame = gframe[bottom:top,left:right]
                            #cars = self.car_cascade.detectMultiScale(car_frame, 1.01, 1)
                            #if len(cars)>0:
                            #    print(str(len(cars)) + "\n")
                            #x,y,w,h = [x, y, w, h]+ np.array([self.detect_left,self.detect_top,0,0])
                            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.rectangle(frame, (left+self.detect_left, bottom+self.detect_top), (right+self.detect_left, top+self.detect_top), (0, 255, 0), 2)
                        # 動体の輪郭が最大のものを描画（検知なら白、それ以外は青色）
                        #if self.state == "DETECT":
                        #    bounding_color = (255,255,255)
                        #else:
                        #    bounding_color = (255,0,0)
                        #cv2.rectangle(frame, (x_max, y_max),(x_max + w_max, y_max + h_max), bounding_color, 2)

                    if not self.webcam:
                        # 動体検知数の履歴
                        self.detectarea_count_list.append(len(areas))
                        # 最大動体位置の履歴
                        center_x_max = int(self.detect_left + x_max + w_max / 2.0)
                        center_y_max = int(self.height - (self.detect_top + y_max + h_max / 2.0))
                        self.detectarea_x_list.append(center_x_max)
                        self.detectarea_y_list.append(center_y_max)
                        self.detectarea_w_list.append(w_max)
                        self.detectarea_h_list.append(h_max)
                        # 最大動体のプロットカラー設定（グラフ用）
                        self.set_plotcolor(hit)
                        # 最大動体位置間の距離の履歴
                        if not np.isnan(self.oldx):
                            self.detectarea_distance_list.append(math.sqrt(((x_max + w_max / 2.0) - (
                                self.oldx + self.oldw / 2.0))**2 + ((y_max + h_max / 2.0) - (self.oldy + self.oldh / 2.0))**2))
                        else:
                            self.detectarea_distance_list.append(np.nan)
                    if self.tracker is not None:
                        ok, newboxes = self.tracker.update(fgmask)
                        for newbox in newboxes:
                           newbox = newbox + np.array([self.detect_left,self.detect_top,0,0])
                           p1 = (int(newbox[0]), int(newbox[1]))
                           p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                           trframe = cv2.rectangle(frame, p1, p2, (0,255,255),2)
                        #cv2.imshow('tracking',trframe)
                    # oldに位置を保存
                    self.oldx = x_max
                    self.oldy = y_max
                    self.oldw = w_max
                    self.oldh = h_max
                else:
                    if not self.webcam:
                        self.detectarea_count_list.append(0)
                        self.detectarea_x_list.append(np.nan)
                        self.detectarea_y_list.append(np.nan)
                        self.detectarea_w_list.append(np.nan)
                        self.detectarea_h_list.append(np.nan)
                        self.detectarea_distance_list.append(np.nan)
                        self.plotcolor_list.append('w')
                        self.hit_list.append(np.nan)
                        self.detect_list.append(np.nan)
                    self.oldx = np.nan
                    self.oldy = np.nan
                    self.oldw = np.nan
                    self.oldh = np.nan
            if not self.webcam:
               self.framenumber_list.append(self.curpos)
        # 最新の30フレーム(約1秒)を一時的に保存
        self.frames.append(frame)
        if len(self.frames) > self.frame_length_before_detection:
            del self.frames[0]

        # 動体のマスクウインドウ
        #if fgmask is not None:
        #   cv2.imshow('thresh',car_frame)


            #dst = cv2.calcBackProject([gframe],[0],self.roi_hist,[0,255],1)
            #ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
            #x,y,w,h = self.track_window
            #img2 = cv2.rectangle(gframe, (x,y), (x+w,y+h), 255,2)


        ####グレー画像を表示
        #if fgmask is not None:
        #    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        #    frame[self.detect_top:self.detect_bottom,
        #               self.detect_left:self.detect_right] = fgmask
        ####オプティカルフローを表示
        if fgmask is not None:
            frame[self.detect_top:self.detect_bottom,
                       self.detect_left:self.detect_right] = optframe
        ####
        self.frame = frame

        if self.verbose and not self.webcam:
            sys.stdout.write("\r{0}%".format(int(
                100.0 * self.curpos / self.framecount)))

        if frame is not None:
            ret = True
        elif frame is None or self.state == "DETECT":
            ret = False
        return ret

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

    def get_detectarea_count_list(self):
        u"""検知数のリストを返す.グラフ表示用."""
        return self.framenumber_list, self.detectarea_count_list

    def get_detectarea_center_list(self):
        u"""検知位置のリストを返す.グラフ表示用."""
        return self.detectarea_x_list, self.detectarea_y_list

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

        self.hits.append(hit)
        if len(self.hits) > 5:
            del self.hits[0]

        return hit

    def check_detection(self):
        u"""動体検知最終判断."""
        state = "NODETECT"
        if sum(self.hits) >= self.hitmax:
            # 過去5フレームの内、3フレームがhitだと検知とする。
            state = "DETECT"
            self.hits = [0] * 5
            if self.verbose:
                if not self.webcam:
                    sys.stdout.write("\r    detect! --> {0} {1}sec".format(basename(self.playfile), self.cursec))
                else:
                    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                    sys.stdout.write("\r    detect! --> {0} {1}".format(basename(self.playfile), timestr))
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
           filename = "ALLwebcam" + self.webcam_savetime.strftime('%Y-%m-%d_%H_%M_%S') + '.mov'
           outfile = join(self.outdir, filename)
           fps = 20 # 処理時間の遅延を考慮するとこれぐらいで、だいたい現実時間と一緒。マシンスペックに依存する？
           self.videoWriter_webcam = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('m','p','4','v'), fps,(self.width, self.height))
        else:
           self.videoWriter_webcam.write(self.frame)
           # 60分たったら次のファイル
           elapsed_time = (datetime.datetime.now()-self.webcam_savetime).seconds / 60
           if(elapsed_time>=60):
              self.videoWriter_webcam.release()
              self.videoWriter_webcam = None

    def writeout_video(self):
        u"""動画書き込み."""
        if self.state == "DETECT" or self.writing:
            # 検知もしくは書き込み中なら
            if self.writing:
                # 動画書き込み中なら
                self.videoWriter.write(self.frame)
                self.remaining_frame_count = self.remaining_frame_count - 1
                if self.remaining_frame_count < 0:  # 書き込み終了
                    self.writing = False
                    self.videoWriter.release()
                    self.videoWriter = None
            else:
                # 動画書き込み中でなければ
                if not os.path.exists(self.outdir):
                    os.makedirs(self.outdir)
                self.writing = True
                self.remaining_frame_count = self.frame_length_after_detection
                if self.webcam:
                    filename = "webcam" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.mov'
                    fps = 20
                else:
                    filename = splitext(basename(self.playfile))[
                        0] + '_' + str(self.cursec).zfill(5) + '.mov'
                    fps = self.fps
                outfile = join(self.outdir, filename)
                self.videoWriter = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('m','p','4','v'), fps,(self.width, self.height))
                # まず、検知直前30フレームを書き出す
                for v in self.frames:
                    self.videoWriter.write(v)

    def writeout_jpg(self):
        u"""jpg書き込み."""
        if self.state == "DETECT" and not self.writing:
            # 検知で書き込み中でないなら
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            if self.webcam:
                filename = "webcam" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.jpg'
                filename_crop = "webcam" + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '_crop.jpg'
            else:
                filename = splitext(basename(self.playfile))[
                    0] + '_' + str(self.cursec).zfill(5) + '.jpg'
                filename_crop = splitext(basename(self.playfile))[
                    0] + '_' + str(self.cursec).zfill(5) + '_crop.jpg'
            ###### opencv3(+python3)だと日本語(cp932)だめ
            outfile = join(self.outdir, filename)
            outfile_crop = join(self.outdir, filename_crop)
            cv2.imwrite(outfile, self.frame)
            if  not np.isnan(self.oldx):
                x = self.oldx
                y = self.oldy
                w = self.oldw
                h = self.oldh
                cv2.imwrite(outfile_crop, self.frame[y:y+h, x:x+w])

            #cv2.imwrite(outfile, self.frame)
            # if self.webcam:
            #   self.tweet(outfile,"photo")


    def set_learning_model(self, modelpath, parampath):
        self.maxlen=920
        # 深層学習のモデルを読み込み
        json_string = open(modelpath).read()
        self.learning_model = model_from_json(json_string)
        #model.summary()
        self.learning_model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        # 学習済みのパラメータを読み込み
        self.learning_model.load_weights(parampath)

    def calc_detect_probability(self):
        u"""検知確率を計算(deep learning)."""
        #print(self.detectarea_x_list)
        detectarea_count_list = sequence.pad_sequences([np.nan_to_num(self.detectarea_count_list)], maxlen=self.maxlen)
        detectarea_x_list = sequence.pad_sequences([np.nan_to_num(self.detectarea_x_list)], maxlen=self.maxlen)
        detectarea_y_list = sequence.pad_sequences([np.nan_to_num(self.detectarea_y_list)], maxlen=self.maxlen)
        detectarea_w_list = sequence.pad_sequences([np.nan_to_num(self.detectarea_w_list)], maxlen=self.maxlen)
        detectarea_h_list = sequence.pad_sequences([np.nan_to_num(self.detectarea_h_list)], maxlen=self.maxlen)
        X = np.dstack((detectarea_count_list, detectarea_x_list, detectarea_y_list, detectarea_w_list, detectarea_h_list))
        return self.learning_model.predict(X)


    def writeout_result(self, outfile):
        u"""ログ書き出し."""
        if not os.path.exists(outfile):
            datastr = ",".join(
                [u"ファイル名", u"検知確率", u"昼夜", u"動体数中央値", u"動体数平均", u"動体数標準偏差", u"動体間距離中央値", u"動体間距離平均", u"動体間距離標準偏差"])
            f = codecs.open(outfile, 'w','cp932')
            f.write(datastr + '\n')
            f.close()
            logstr = datastr.replace(',', '\t') + '\n'
            self.notify(logstr)

        if len(self.detectarea_count_list) > 0:
            detectarea_count_median = int(np.median(self.detectarea_count_list))
            detectarea_count_mean = int(np.mean(self.detectarea_count_list))
            detectarea_count_std = int(np.std(self.detectarea_count_list))
            detectarea_distance_median = np.median(pd.Series(self.detectarea_distance_list).dropna())
            detectarea_distance_mean = np.mean(pd.Series(self.detectarea_distance_list).dropna())
            detectarea_distance_std = np.std(pd.Series(self.detectarea_distance_list).dropna())
        else:
            detectarea_count_median = 0
            detectarea_count_mean = 0
            detectarea_count_std = 0
            detectarea_distance_median = 0
            detectarea_distance_mean = 0
            detectarea_distance_std = 0
        # depp learningの場合
        if self.predict_animal:
            detect_probability = self.calc_detect_probability()
        else:
            detect_probability = 0.5
        # 機械学習の場合
        # detect_probability = self.calc_detect_probability(
        #    self.daytime, detectarea_count_median, detectarea_count_mean, detectarea_count_std, detectarea_distance_median, detectarea_distance_mean, detectarea_distance_std)

        datastr = ",".join(
            [self.playfile, str(detect_probability), self.daytime, str(detectarea_count_median), str(detectarea_count_mean), str(detectarea_count_std), str(detectarea_distance_median), str(detectarea_distance_mean), str(detectarea_distance_std)])
        f = codecs.open(outfile, 'a','cp932')
        f.write(datastr + '\n')
        f.close()
        logstr = datastr.replace(',', '\t') + '\n'
        self.notify(logstr)

    def set_plotcolor(self, hit):
        u"""最大動体のプロットカラーの設定."""
        col = '#aaaaaa'  # 検知もhitもない状態
        if self.state == "DETECT":
            col = 'r'  # 検知した状態
        elif hit > 0:
            col = 'b'  # 検知してなくてヒットの状態
        self.plotcolor_list.append(col)

    def writeout_graph(self):
        u"""グラフ書き出し."""
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        filename = join(self.outdir, splitext(
                        basename(self.playfile))[0] + '_graph.png')
        gs = gridspec.GridSpec(4, 1, height_ratios=[9, 9, 9, 9])
        plt.figure(figsize=(8, 18))

        # 動体検知数の時系列グラフ（風の状態を知る）
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.framenumber_list, self.detectarea_count_list)
        ax1.axes.set_title("detected area count(Time series)")
        ax1.axes.title.set_fontsize(10)
        ax1.axes.set_xlabel("Time(Frame Number)", fontsize=10)
        ax1.axes.set_ylabel("Count", fontsize=10)
        # 動体検知数のヒストグラム（風の状態を知る）
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(self.detectarea_count_list, bins=20, range=(
            0, 1000), normed=False, facecolor='g', alpha=0.8)
        ax2.axes.set_title("detected area count(Histgram)")
        ax2.axes.title.set_fontsize(10)
        ax2.axes.set_xlabel("Count", fontsize=10)
        ax2.axes.set_ylabel("Frequency", fontsize=10)
        # 最大動体の中心位置をプロット（動体の動きを知る）
        ax3 = plt.subplot(gs[2, 0])
        ax3.set_xlim([0, 1280])
        ax3.set_ylim([0, 720])
        ax3.scatter(self.detectarea_x_list, self.detectarea_y_list,
                    marker='x', color=self.plotcolor_list, s=0.5)
        ax3.axes.set_title("maximum detected area position")
        ax3.axes.title.set_fontsize(10)
        ax3.axes.set_xlabel("X", fontsize=10)
        ax3.axes.set_ylabel("Y", fontsize=10)
        # 最大動体の中心位置をプロット時間軸3D（動体の動きを知る）
        ax4 = plt.subplot(gs[3, 0], projection='3d')
        ax4.set_xlim([0, 1280])
        ax4.set_ylim([0, 720])
        ax4.scatter(self.detectarea_x_list, self.detectarea_y_list,
                    self.framenumber_list, marker='x', c=self.plotcolor_list, s=0.5)
        # ax4.scatter(self.detectarea_x_list, self.detectarea_y_list,
        #            self.framenumber_list, marker='x', c='r', s=0.1)
        ax4.axes.set_title("Maximum Detected Area Position with Time")
        ax4.axes.title.set_fontsize(10)
        ax4.axes.set_xlabel("X", fontsize=10)
        ax4.axes.set_ylabel("Y", fontsize=10)
        ax4.axes.set_zlabel("Time", fontsize=10)

        plt.tight_layout()

        plt.savefig(filename, dpi=300)
        plt.close()

    def writeout_data(self):
        u"""データ書き出し."""
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        filename = join(self.outdir, splitext(
                        basename(self.playfile))[0] + '_data.csv')
        data = {'framenumber': self.framenumber_list,
                'detectarea_count': self.detectarea_count_list,
                'detectarea_x': self.detectarea_x_list,
                'detectarea_y': self.detectarea_y_list,
                'detectarea_w': self.detectarea_w_list,
                'detectarea_h': self.detectarea_h_list,
                'detectarea_distance': self.detectarea_distance_list,
                'detect': self.detect_list,
                'hit': self.hit_list,
                'plotcolor': self.plotcolor_list}

        df = DataFrame(data)
        df.to_csv(filename, index=False,
                  columns=['framenumber',
                           'detectarea_count',
                           'detectarea_x',
                           'detectarea_y',
                           'detectarea_w',
                           'detectarea_h',
                           'detectarea_distance',
                           'detect',
                           'hit',
                           'plotcolor'])

    def get_frame(self):
        u"""ビデオフレーム画像（opencv）からQPixmap形式に変換."""
        # Pixmap用にBGRからRGBに入れ替え
        try:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            img = QtGui.QImage(frame, width, height,
                               QtGui.QImage.Format_RGB888)
            img = QtGui.QPixmap.fromImage(img)
            return img
        except:
            return None


    """
    def tweet(self, photo_path, tweet_type):
        # Consumer keys and access tokens, used for OAuth
        consumer_key = 'rRyMJB01tYAmESV6mVSIzQWTH'
        consumer_secret = 'usHKnVB46EqtrJcQi9xjsri2V0mOpyjAAnhpjqrLvsOLlMMUw9'
        access_token = '753815107860205568-rNYJoDDvTxQxieixceKXc6lnGHkgyV4'
        access_token_secret = 'aQuSmVmudL0Ss6YBCVFo05nmwx6p0ckPMtCkjrlzugNCE'

        # OAuth process, using the keys and tokens
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        # Creation of the actual interface, using authentication
        api = tweepy.API(auth)
        status = "detect!"
        if tweet_type in "photo":
            api.update_with_media(photo_path, status=status)
        else:
            video_path = 'output.mp4'
            media = api.media_upload(video_path)
            api.update_status(status=status, media_ids=[media.media_id])
            print("tweet")
"""
class Settei():
    u"""処理設定用クラス."""

    def __init__(self):
        u"""初期設定."""
        self.settings = OrderedDict([
            ("webcam", False),
            ("playdir", os.path.abspath(os.path.dirname(__file__))),
            ("outdir", os.path.abspath(os.path.dirname(__file__))),
            ("avi", True),
            ("mov", True),
            ("mpg", True),
            ("mp4", True),
            ("wmv", True),
            ("flv", True),
            ("mts", True),
            ("m2ts", True),
            ("writevideo", True),
            ("writejpg", True),
            ("writegraph", True),
            ("bounding", True),
            ("display", True),
            ("plotgraph", True),
            ("verbose", True),
            #("checkall", True),
            ("detectionTop", 0),
            ("detectionBottom", 720),
            ("detectionLeft", 0),
            ("detectionRight", 1080),
            ("speedSlider", 50)])

    def load_cui_settings(self):
        u"""コマンドライン引数の処理."""
        self.settings["writevideo"] = False
        self.settings["writejpg"] = False
        self.settings["writegraph"] = False
        self.settings["bounding"] = False
        #self.settings["checkall"] = False
        self.settings["avi"] = False
        self.settings["mov"] = False
        self.settings["mpg"] = False
        self.settings["mp4"] = False
        self.settings["wmv"] = False
        self.settings["flv"] = False
        self.settings["mts"] = False
        self.settings["m2ts"] = False
        args = docopt(__doc__, version="0.1.2")
        if args["--inpdir"]:
            inpdir = args["--inpdir"]
            self.settings["playdir"] = inpdir.replace('/',os.sep)
        if args["--outdir"]:
            outdir = args["--outdir"]
            self.settings["outdir"] = outdir.replace('/',os.sep)
        if args["--webcam"]:
            self.settings["webcam"] = True
        if args["--avi"]:
            self.settings["avi"] = args["--avi"]
        if args["--mov"]:
            self.settings["mov"] = args["--mov"]
        if args["--mpg"]:
            self.settings["mpg"] = args["--mpg"]
        if args["--mp4"]:
            self.settings["mp4"] = args["--mp4"]
        if args["--wmv"]:
            self.settings["wmv"] = args["--wmv"]
        if args["--flv"]:
            self.settings["flv"] = args["--flv"]
        if args["--mts"]:
            self.settings["mts"] = args["--mts"]
        if args["--m2ts"]:
            self.settings["m2ts"] = args["--m2ts"]
        if args["-v"]:
            self.settings["writevideo"] = True
        if args["-j"]:
            self.settings["writejpg"] = True
        if args["-g"]:
            self.settings["writegraph"] = True
        if args["-b"]:
            self.settings["bounding"] = True
        #if args["--all"]:
        #    self.settings["checkall"] = True
        if args["--area"]:
            detectarea = args["--area"].split(",")
            self.settings["detectionTOP"] = detectarea[0]
            self.settings["detectionBottom"] = detectarea[1]
            self.settings["detectionLeft"] = detectarea[2]
            self.settings["detectionRight"] = detectarea[3]
        if args["-d"]:
            self.settings["verbose"] = True
        #if args["--resume"]:
        #    resumefw = open(args["--resume"], 'a+')
        #    resumefr = open(args["--resume"], 'r')

    def load_settings(self):
        u"""設定ファイル（settings.json）の読み込み."""
        inidir = os.path.abspath(os.path.dirname(__file__))
        setting_file = join(inidir, "settings.json")
        if os.path.exists(setting_file):
            f = codecs.open(setting_file, 'r','utf-8')  # 書き込みモードで開く
            self.settings = json.load(f)
            # outdirとplaydirの存在確認
            outdir = self.settings["outdir"]
            if not os.path.exists(outdir):
                self.settings["outdir"] = os.path.abspath(os.path.dirname(__file__))
            playdir = self.settings["playdir"]
            if not os.path.exists(playdir):
                self.settings["playdir"] = os.path.abspath(os.path.dirname(__file__))

    def save_settings(self):
        u"""設定ファイルの書き出し."""
        inidir = os.path.abspath(os.path.dirname(__file__))
        f = codecs.open(join(inidir, "settings.json"), 'w', 'utf-8')  # 書き込みモードで開く
        json.dump(self.settings, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.close()

class MainView(QtGui.QMainWindow, Ui_MainWindow):
    u"""GUIのクラス."""

    def __init__(self, parent=None):
        u"""GUI初期設定."""
        # QtGui.QWidget.__init__(self,parent)
        super(MainView, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.setupUi(self)

        # 変数
        self.painter = QtGui.QPainter()
        self.model = QtGui.QDirModel()
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
        self.writegraph_checkBox.setChecked(settei.settings["writegraph"])
        self.bounding_checkBox.setChecked(settei.settings["bounding"])
        self.display_checkBox.setChecked(settei.settings["display"])
        self.plotgraph_checkBox.setChecked(settei.settings["plotgraph"])
        self.verbose_checkBox.setChecked(settei.settings["verbose"])
        self.detectionTop_Edit.setText(str(settei.settings["detectionTop"]))
        self.detectionBottom_Edit.setText(str(settei.settings["detectionBottom"]))
        self.detectionLeft_Edit.setText(str(settei.settings["detectionLeft"]))
        self.detectionRight_Edit.setText(str(settei.settings["detectionRight"]))
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

    def set_inputformat(self):
        u"""入力ビデオフォーマットのフィルタリング設定."""
        print("main_view:set_inputformat")
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

    def set_mouseselect(self,state):
        u"""検知範囲のマウス選択を設定."""
        print("main_view:set_mouseselect")
        self.detectionArea_Button.setEnabled(state)

    def select_detectionarea_by_mouse(self, event):
        u"""検知範囲のマウス選択処理."""
        print("main_view:select_detectionarea_by_mouse")
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

    def set_detectionarea_by_mouse(self, event, height):
        u"""検知範囲のマウス選択処理."""
        print("main_view:set_detectionarea_by_mouse")
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
        print("main_view:draw_detectionarea")
        if height==0:
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
        print("main_view:get_detectionarea")
        top = int(self.detectionTop_Edit.text())
        bottom = int(self.detectionBottom_Edit.text())
        left = int(self.detectionLeft_Edit.text())
        right = int(self.detectionRight_Edit.text())
        return top, bottom, left, right

    def set_detectionarea(self, top, bottom, left, right, height):
        u"""検知範囲の設定."""
        print("main_view:set_detectionarea")
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

    def set_writevideo(self,checked):
        u"""ビデオ書き出すか？設定."""
        self.writevideo_checkBox.setChecked(checked)

    def get_writevideo(self):
        u"""ビデオ書き出すか?."""
        writevideo = self.writevideo_checkBox.isChecked()
        return writevideo

    def set_writejpg(self,checked):
        u"""jpg書き出すか？設定."""
        self.writejpg_checkBox.setChecked(checked)

    def get_writejpg(self):
        u"""ビデオ書き出すか？."""
        writejpg = self.writejpg_checkBox.isChecked()
        return writejpg

    def set_writegraph(self,checked):
        u"""グラフ書き出すか？設定."""
        self.writegraph_checkBox.setChecked(checked)

    def get_writegraph(self):
        u"""グラフ書き出すか？."""
        writegraph = self.writegraph_checkBox.isChecked()
        return writegraph

    def get_display(self):
        u"""画面表示するか？設定."""
        display = self.display_checkBox.isChecked()
        return display

    def clear_display(self):
        u"""画面をクリア."""
        self.videoFrame.clear()

    def get_plotgraph(self):
        u"""グラフ表示するか？設定."""
        plotgraph = self.plotgraph_checkBox.isChecked()
        return plotgraph

    def set_plotgraph(self,checked):
        u"""グラフ表示するか？設定."""
        self.plotgraph_checkBox.setChecked(checked)

    def get_verbose(self):
        u"""ログ出力を多めにするか？設定."""
        verbose = self.verbose_checkBox.isChecked()
        return verbose

    def get_outdir(self):
        u"""出力フォルダを返す."""
        outdir = QtGui.QFileDialog.getExistingDirectory(
            self, "Select Output folder")
        self.outdirEdit.setText(outdir)
        return outdir

    def get_playdir(self):
        u"""再生ビデオフォルダ選択.プレイリストをセット."""
        playdir = QtGui.QFileDialog.getExistingDirectory(
                    self, "Select folder")
        return playdir

    def set_playdir(self, playdir):
        u"""プレイリストをセット."""
        self.trackSlider.setEnabled(True)
        self.nextframeButton.setEnabled(True)
        self.nextvideoButton.setEnabled(True)
        self.plotgraph_checkBox.setEnabled(True)
        self.writegraph_checkBox.setEnabled(True)
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
        # self.set_video(None)

    def set_video_view(self, filename, framecount):
        u"""再生ビデオを変更＆初期設定."""
        print("main_view:set_video_view")
        model = self.treeView.model()
        index = model.index(filename)
        self.treeView.scrollTo(index, QtGui.QAbstractItemView.PositionAtCenter)
        self.treeView.selectionModel().clearSelection()
        self.treeView.selectionModel().select(
            index, QtGui.QItemSelectionModel.Select | QtGui.QItemSelectionModel.Rows)
        self.trackSlider.setMaximum(framecount)
        self.trackSlider.setValue(0)
        self.pixmap.fill(QtCore.Qt.black)
        self.videoFrame.setPixmap(self.pixmap)
        self.detectionArea_Button.setEnabled(True)

    def get_filename_treeview(self, index):
        u"""ツリービューから選択されたファイルを返す."""
        model = self.treeView.model()
        indexItem = model.index(index.row(), 0, index.parent())
        filename = model.filePath(indexItem).replace('/', os.sep)
        return filename

    def set_webcam_view(self, webcam = True):
        u"""webcamを入力デバイスに設定."""
        # ToDo:書き出ししないとか設定確認
        if webcam:
            self.treeView.setEnabled(False)
            self.pixmap.fill(QtCore.Qt.black)
            self.videoFrame.setPixmap(self.pixmap)
            self.detectionArea_Button.setEnabled(True)
            self.trackSlider.setEnabled(False)
            self.nextframeButton.setEnabled(False)
            self.nextvideoButton.setEnabled(False)
            self.plotgraph_checkBox.setEnabled(False)
            self.writegraph_checkBox.setEnabled(False)
            self.fileEdit.setText("Input is from WebCam...")
        else:
            self.fileEdit.setText("No connected with WebCam !")

    def set_play_label(self, label):
        u"""ビデオ再生 or 停止処理."""
        self.playButton.setText(label)

    def change_tracklabel(self, fps):
        u"""再生経過秒数表示."""
        if fps!=0:
            d = datetime.timedelta(seconds=int(
                self.trackSlider.value() / fps))
            self.tracklabel.setText(str(d))

    def set_track_position(self, curpos):
        u"""トラックスライダー位置設定."""
        self.trackSlider.setValue(curpos)

    def set_pixmap(self, pixmap):
        u"""画面設定."""
        pixmap = pixmap.scaled(640, 360, QtCore.Qt.KeepAspectRatio)
        self.videoFrame.setPixmap(pixmap)
        self.pixmap = pixmap

    def write_log(self, str):
        u"""ログ表示."""
        self.logEdit.insertPlainText(str)

    def plot_graph(self, id, x, y, title, ylabel, color, marker, size, xlim=None, ylim=None):
        u"""グラフ表示."""
        if id == 0:
            graph = self.graph1_plot
        elif id == 1:
            graph = self.graph2_plot
        graph.axes.scatter(x, y, marker=marker, color=color, s=size)
        if xlim is not None:
            graph.axes.set_xlim([0, xlim])
        if ylim is not None:
            graph.axes.set_ylim([0, ylim])
        graph.axes.set_title(title)
        graph.axes.title.set_fontsize(7)
        graph.axes.set_ylabel(ylabel, fontsize=7)
        labels_x = graph.axes.get_xticklabels()
        labels_y = graph.axes.get_yticklabels()
        for xlabel in labels_x:
            xlabel.set_fontsize(6)
        for ylabel in labels_y:
            ylabel.set_fontsize(6)
        graph.draw()


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
        print("main_ctr:load_settings")
        self.settei.load_settings()
        self.main_view.set_settings(self.settei)
        self.set_inputformat()
        self.set_actions()
        if self.settei.settings["webcam"]:
            self.set_webcam()

    def set_actions(self):
        u"""アクションの設定."""
        print("main_ctr:set_actions")
        self.main_view.actionOpen_Folder.triggered.connect(self.set_playdir)
        self.main_view.folderButton.clicked.connect(self.set_playdir)
        self.main_view.actionWebCam.triggered.connect(self.set_webcam)
        self.main_view.actionSave_Settings.triggered.connect(self.save_settings)
        self.main_view.outdirButton.clicked.connect(self.set_outdir)
        self.main_view.playButton.clicked.connect(self.play_video)
        self.main_view.nextframeButton.clicked.connect(self.step_nextframe)
        self.main_view.nextvideoButton.clicked.connect(self.step_nextvideo)
        self.main_view.trackSlider.valueChanged.connect(self.slider_changed)
        self.main_view.trackSlider.sliderPressed.connect(self.slider_pressed)
        self.main_view.trackSlider.sliderReleased.connect(self.slider_moved)
        self.main_view.speedSlider.valueChanged.connect(self.set_speed)
        self.main_view.treeView.doubleClicked.connect(self.select_playfile_treeview)
        self.main_view.writevideo_checkBox.stateChanged.connect(self.set_writevideo)
        self.main_view.writejpg_checkBox.stateChanged.connect(self.set_writejpg)
        self.main_view.writegraph_checkBox.stateChanged.connect(self.set_writegraph)
        self.main_view.bounding_checkBox.stateChanged.connect(self.set_bounding)
        self.main_view.display_checkBox.stateChanged.connect(self.set_display)
        self.main_view.plotgraph_checkBox.stateChanged.connect(self.set_plotgraph)
        self.main_view.verbose_checkBox.stateChanged.connect(self.set_verbose)
        self.main_view.avi_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mov_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mpg_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mp4_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.wmv_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.flv_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.mts_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.m2ts_checkBox.stateChanged.connect(self.set_inputformat)
        self.main_view.detectionTop_Edit.editingFinished.connect(self.check_detectionarea)
        self.main_view.detectionBottom_Edit.editingFinished.connect(self.check_detectionarea)
        self.main_view.detectionLeft_Edit.editingFinished.connect(self.check_detectionarea)
        self.main_view.detectionRight_Edit.editingFinished.connect(self.check_detectionarea)
        self.main_view.detectionArea_Button.clicked.connect(self.active_mouseselect)
        self.main_view.videoFrame.mousePressEvent = self.set_detectionarea_by_mouse
        self.main_view.videoFrame.mouseReleaseEvent = self.set_detectionarea_by_mouse
        self.main_view.videoFrame.mouseMoveEvent = self.set_detectionarea_by_mouse

    def plot_graph(self):
        u"""グラフ表示."""
        print("main_ctr:plot_graph")
        # 動体検知数の時系列グラフ（風の状態を知る）
        id = 0
        x, y = self.video.get_detectarea_count_list()
        title = "detected area count(Time series)"
        ylabel = "count"
        color = "red"
        marker = 'o'
        size = 0.5
        self.main_view.plot_graph(id, x, y, title, ylabel, color, marker, size)

        # 最大動体の中心位置をプロット（動体の動きを知る）
        id = 1
        x, y = self.video.get_detectarea_center_list()
        title = "maximum detected area position(XY)"
        ylabel = "Y"
        color = self.video.plotcolor_list
        marker = 'x'
        size = 0.5
        height, width = self.video.get_size()
        xlim = width
        ylim = height
        self.main_view.plot_graph(id, x, y, title, ylabel, color, marker, size, xlim, ylim)

        """
        # 動体検知数のヒストグラム（風の状態を知る）
        self.graph3_plot.axes.hist(self.video.detectarea_count_list, bins=50, range=(
            0, 1000), normed=False, facecolor='g', alpha=0.8)
        self.graph3_plot.draw()
        """

    def play(self, singlestep=False):
        u"""ビデオ再生＆表示＆検知＆グラフ処理.（タイマーからループで呼び出される）."""
        # 1フレーム処理
        ret = self.video.capture_nextframe()

        if ret is True:
            curpos = self.video.get_position()
            self.main_view.set_track_position(curpos)
            # ビデオ終わりでなければビデオ＆jpg書き出しチェック＆画面表示＆グラフプロット
            if self.settei.settings["writejpg"]:
                self.video.writeout_jpg()
            if self.settei.settings["writevideo"]:
                self.video.writeout_video()
            if self.settei.settings["webcam"]:
                self.video.writeout_webcam()
            if self.settei.settings["display"]:
                pixmap = self.video.get_frame()
                self.main_view.set_pixmap(pixmap)
            if self.settei.settings["plotgraph"]:
                if curpos % 30 == 0:
                    # 30フレームに1回グラフ表示
                    self.plot_graph()
        else:
            # ビデオ終わりなら書き込み終了＆ログ出力＆グラフ出力
            self.video.close_video()
            filename = join(self.settei.settings["outdir"], 'result.csv')
            self.video.writeout_result(filename)
            if self.settei.settings["writegraph"]:
                self.video.writeout_graph()
                self.video.writeout_data()
            # 次のファイルを確認＆変更
            if len(self.playlist) > self.playnumber + 1:
                # ファイル残っていたら次のビデオを再生
                self.playnumber = self.playnumber + 1
                filename = self.playlist[self.playnumber]
                self.set_video(filename)
            else:
                # ファイル残っていなかったら停止
                self.play_video("STOP")
        # コマ送りの場合は停止
        if singlestep:
            self.play_video("STOP")

    def play_video(self, action):
        u"""ビデオ再生 or 停止処理."""
        print("main_ctr:play_video")
        if(action == "START" and self.video is not None):
            # 再生させる
            self.pause = False
            self._timer.start()
            self.main_view.set_play_label("STOP")
            self.main_view.set_format_checkbox(False)
            self.main_view.set_mouseselect(False)
        elif action == "STOP":
            # 停止させる
            self.pause = True
            self._timer.stop()
            self.video.close_video()#ビデオも一旦書き出し
            self.main_view.set_play_label("START")
            self.main_view.set_format_checkbox(True)
            self.main_view.set_mouseselect(True)
        elif self.pause and self.video is not None:
            # 停止していたら再生させる
            self.pause = False
            self._timer.start()
            self.main_view.set_play_label("STOP")
            self.main_view.set_format_checkbox(False)
            self.main_view.set_mouseselect(False)
        else:
            # 再生していたら停止させる
            self.pause = True
            self._timer.stop()
            self.video.close_video()#ビデオも一旦書き出し
            self.main_view.set_play_label("START")
            self.main_view.set_format_checkbox(True)
            self.main_view.set_mouseselect(True)


    def step_nextframe(self):
        u"""コマ送り再生."""
        self.play(singlestep=True)

    def step_nextvideo(self):
        u"""次のビデオに切り替え."""
        if len(self.playlist) > self.playnumber + 1:
            self.playnumber = self.playnumber + 1
            filename = self.playlist[self.playnumber]
            self.set_video(filename)

    def slider_changed(self, pos):
        u"""トラックバー変更でラベル変更."""
        fps = self.video.get_fps()
        self.main_view.change_tracklabel(fps)

    def slider_pressed(self):
        u"""トラックバー移動でビデオ停止."""
        print("main_ctr:slider_pressed")
        self.play_video("STOP")

    def slider_moved(self):
        u"""トラックバーの移動終了."""
        print("main_ctr:slider_moved")
        pos = self.main_view.get_slider_position()
        self.video.set_position(pos)
        self.play()

    def set_outdir(self):
        u"""出力フォルダを設定."""
        print("main_ctr:set_outdir")
        outdir = self.main_view.get_outdir()
        if outdir:
            self.settei.settings["outdir"] = outdir
        if self.video:
            if self.settei.settings["webcam"]:
                self.video.set_outdir(outdir)
            else:
                tmppath = self.settei.settings["outdir"] + self.playfile.replace(self.settei.settings["playdir"], "")
                recursive_outdir = os.path.dirname(tmppath)
                self.video.set_outdir(recursive_outdir)

    def select_playfile_treeview(self, index):
        u"""ツリービューから選択されたファイルを再生ビデオに設定."""
        print("main_ctr:select_playfile_treeview")
        playfile = self.main_view.get_filename_treeview(index)
        if os.path.isfile(playfile): # directoryなら何もしない
            playnumber = self.playlist.index(playfile)
            self.playnumber = playnumber
            self.playfile = playfile
            self.set_video(playfile)

    def set_playdir(self):
        u"""プレイフォルダを設定."""
        print("main_ctr:set_playdir")
        self.play_video("STOP")
        playdir = self.main_view.get_playdir()
        if playdir:
            self.settei.settings["webcam"] = False
            self.settei.settings["playdir"] = playdir
            self.main_view.set_playdir(playdir)
            self.set_playlist()

    def set_inputformat(self):
        u"""入力ビデオフォーマットのフィルタリング設定.プレイリストも更新."""
        print("main_ctr:set_inputformat")
        self.main_view.set_inputformat()
        self.set_playlist()

    def set_playlist(self):
        u"""再生リストの設定."""
        print("main_ctr:set_playlist")
        playdir = self.settei.settings["playdir"]
        playlist = self.main_view.get_playlist(playdir)
        if len(playlist) == 0:
            self.video = None
            self.main_view.set_nothing_playlist()
        else:
            self.playlist = playlist
            self.playnumber = 0
            playfile = self.playlist[self.playnumber]
            self.playfile = playfile
            self.set_video(playfile)

    def set_video(self, playfile):
        u"""再生ビデオを変更＆初期設定."""
        print("main_ctr:set_video")
        self.webcam = False
        self.video = None
        tmppath = self.settei.settings["outdir"] + playfile.replace(self.settei.settings["playdir"], "")
        recursive_outdir = os.path.dirname(str(tmppath))
        self.video = Video(playfile, recursive_outdir)
        self.video.set_bounding(self.settei.settings["bounding"])
        self.video.set_verbose(self.settei.settings["verbose"])
        framecount = self.video.get_framecount()
        self.main_view.set_video_view(playfile, framecount)
        self.video.attach(self.main_view.write_log)
        top = self.settei.settings["detectionTop"]
        bottom = self.settei.settings["detectionBottom"]
        left = self.settei.settings["detectionLeft"]
        right = self.settei.settings["detectionRight"]
        self.set_detectionarea(top, bottom, left, right)
        self.set_speed()

    def set_detectionarea(self, top, bottom, left, right):
        u"""検知範囲のビデオへの設定."""
        print("main_ctr:set_detectionarea")
        height, width = self.video.get_size()
        self.main_view.set_detectionarea(top, bottom, left, right, height)
        self.settei.settings["detectionTop"] = top
        self.settei.settings["detectionBottom"] = bottom
        self.settei.settings["detectionLeft"] = left
        self.settei.settings["detectionRight"] = right
        self.video.set_detectionarea(top, bottom, left, right)
        self.video.init_learning()

    def check_detectionarea(self):
        u"""検知エリアの確認."""
        print("main_ctr:check_detectionarea")
        top, bottom, left, right = self.main_view.get_detectionarea()
        if bottom - top <= 0 or right - left <= 0:  # 有効でなければもとに戻す
            top = self.settei.settings["detectionTop"]
            bottom = self.settei.settings["detectionBottom"]
            left = self.settei.settings["detectionLeft"]
            right = self.settei.settings["detectionRight"]
        self.set_detectionarea(top, bottom, left, right)

    def set_webcam(self):
        u"""webcamの設定."""
        print("main_ctr:set_webcam")
        self.settei.settings["webcam"] = True
        self.video = None
        outdir = self.settei.settings["outdir"]
        playfile = self.playfile = 0
        self.video = Video(playfile, outdir, webcam=True)
        # webcamが接続されていれば初期設定
        if self.video.check_webcam() is True:
            height, width = self.video.get_size()
            top = 0
            bottom = height
            left = 0
            right = width
            self.set_detectionarea(top, bottom, left, right)
            self.video.set_bounding(self.settei.settings["bounding"])
            self.video.set_verbose(self.settei.settings["verbose"])
            self.set_plotgraph(False)
            self.set_writegraph(False)
            self.main_view.set_webcam_view()
            self.play_video("START")
        else:
            self.main_view.set_webcam_view(webcam=False)

    def save_settings(self):
        u"""設定の保存."""
        print("main_ctr:save_settings")
        self.settei.save_settings()

    def set_speed(self):
        u"""タイマーのインターバルを設定."""
        print("main_ctr:set_speed")
        speedvalue = self.main_view.get_speed()
        # speed: 0～2 の値 0はできるだけ速く.2は通常の2倍の時間遅い
        speed = -speedvalue / 50.0 + 2
        self.settei.settings["speedSlider"] = speedvalue
        fps = 0
        if self.video is not None:
            fps = self.video.get_fps() #webcamは0になる
        self._timer.setInterval(speed * fps)

    def set_bounding(self, bounding):
        u"""領域表示設定."""
        print("main_ctr:set_bounding")
        bounding = self.main_view.get_bounding()
        self.settei.settings["bounding"] = bounding
        if self.video is not None:
            self.video.set_bounding(bounding)

    def set_writevideo(self, writevideo):
        u"""ビデオを書き出すか設定."""
        if writevideo is not None:
            self.main_view.set_writevideo(writevideo)
        writevideo = self.main_view.get_writevideo()
        self.settei.settings["writevideo"] = writevideo

    def set_writejpg(self, writejpg):
        u"""jpgを書き出すか設定."""
        if writejpg is not None:
            self.main_view.set_writejpg(writejpg)
        writejpg = self.main_view.get_writejpg()
        self.settei.settings["writejpg"] = writejpg

    def set_writegraph(self, writegraph):
        u"""グラフを書き出すか設定."""
        if writegraph is not None:
            self.main_view.set_writegraph(writegraph)
        writegraph = self.main_view.get_writegraph()
        self.settei.settings["writegraph"] = writegraph

    def set_display(self, diplay):
        u"""画面表示設定."""
        display = self.main_view.get_display()
        self.settei.settings["display"] = display
        if display is False:
            self.main_view.clear_display()

    def set_plotgraph(self, plotgraph):
        u"""グラフをプロットするか設定."""
        if plotgraph is not None:
            self.main_view.set_plotgraph(plotgraph)
        plotgraph = self.main_view.get_plotgraph()
        self.settei.settings["plotgraph"] = plotgraph

    def set_verbose(self, verbose):
        u"""ログ出力を設定."""
        verbose = self.main_view.get_verbose()
        self.settei.settings["verbose"] = verbose
        if self.video is not None:
            self.video.set_verbose(verbose)

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


    def set_webcam(self):
        u"""webcamの設定."""
        print("main_ctr:set_webcam")
        self.settei.settings["webcam"] = True
        self.video = None
        outdir = self.settei.settings["outdir"]
        playfile = self.playfile = 0
        self.video = Video(playfile, outdir, webcam=True)
        # webcamが接続されていれば初期設定
        if self.video.check_webcam() is True:
            height, width = self.video.get_size()
            top = 0
            bottom = height
            left = 0
            right = width
            self.set_detectionarea(top, bottom, left, right)
            self.video.set_bounding(self.settei.settings["bounding"])
            self.video.set_verbose(self.settei.settings["verbose"])
            self.set_plotgraph(False)
            self.set_writegraph(False)
            self.main_view.set_webcam_view()
            self.play_video("START")
        else:
            self.main_view.set_webcam_view(webcam=False)

    def __init__(self, settei):
        u"""初期設定."""
        self.settei = settei
        self.video = None
        self.playlist = None
        self.playnumber = None
        self.playfile = None
        self.pause = True
        # コマンドライン引数取得
        self.settei.load_cui_settings()
        if self.settei.settings["webcam"]:
            print("Input from webcam...")
            outdir = self.settei.settings["outdir"]
            playfile = self.playfile = 0
            self.video = Video(playfile, outdir, webcam=True)
            if self.video.check_webcam() is True:
                height, width = self.video.get_size()
                top = 0
                bottom = height
                left = 0
                right = width
                self.video.set_detectionarea(top, bottom, left, right)
                self.video.set_bounding(self.settei.settings["bounding"])
                self.video.set_verbose(self.settei.settings["verbose"])
                ret = True
                # ビデオ再生＆検知
                while(ret):
                    try:
                       ret = self.play()
                    except KeyboardInterrupt:
                       print("\nKeyboardInterrupt!!!\n")
                       self.video.close_video()
                       sys.exit()

            else:
                print("Nothing Input webcam.")
        else:
           playdir = self.settei.settings["playdir"]
           if os.path.exists(playdir):
            self.playlist = self.get_playlist(playdir)
            if len(self.playlist) > 0:
                #プレイリストのビデオに対してチェック
                for playfile in self.playlist:
                    # 再生ビデオの初期設定.
                    print(playfile)
                    tmppath = self.settei.settings["outdir"] + playfile.replace(self.settei.settings["playdir"], "")
                    recursive_outdir = os.path.dirname(tmppath)
                    self.playfile = playfile
                    self.video = Video(playfile, recursive_outdir)
                    self.video.set_bounding(self.settei.settings["bounding"])
                    self.video.set_verbose(self.settei.settings["verbose"])
                    top = self.settei.settings["detectionTop"]
                    bottom = self.settei.settings["detectionBottom"]
                    left = self.settei.settings["detectionLeft"]
                    right = self.settei.settings["detectionRight"]
                    self.video.set_detectionarea(top, bottom, left, right)
                    ret = True
                    # ビデオ再生＆検知
                    while(ret):
                        ret = self.play()
            else:
                print("Nothing playlist.")
           else:
              print("playdir does not exist.")
           sys.exit()

    def play(self):
        u"""ビデオ再生＆検知."""
        # 1フレーム処理
        ret = self.video.capture_nextframe()

        if ret is True:
            # ビデオ終わりでなければビデオ＆jpg書き出しチェック＆実行
            if self.settei.settings["writejpg"]:
                self.video.writeout_jpg()
            if self.settei.settings["writevideo"]:
                self.video.writeout_video()
            if self.settei.settings["webcam"]:
                self.video.writeout_webcam()
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


class App(QtGui.QApplication):
    u"""Appクラス.引数なければGUIで実行.あればコマンドラインで実行する."""

    def __init__(self, sys_argv):
        u"""初期設定."""
        super(App, self).__init__(sys_argv)
        self.useGUI = True
        if len(sys_argv) > 1:
            self.useGUI = False
        if self.useGUI:
            self.settei = Settei()
            self.main_view = MainView()
            self.main_ctrl = MainController(self.settei, self.main_view)
            self.main_view.show()
        else:
            self.settei = Settei()
            self.cui_ctrl = CuiController(self.settei)

    def __del__(self):
        u"""終了処理."""
        # webcamの状態で閉じると終了できないので、capにNoneを入れる.
        if self.useGUI and self.main_ctrl.video is not None:
            self.main_ctrl.video.cap = None

if __name__ == '__main__':
    u"""メイン."""
    app = App(sys.argv)
    sys.exit(app.exec_())
