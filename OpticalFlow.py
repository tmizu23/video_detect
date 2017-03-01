import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import pickle
from MyUtil import find_exedir


class MotionDetection():

    def __init__(self, fps, height, labeling, bounding):
        u"""パラメータ初期設定."""
        self.height = height  # ログ書き出し用に必要
        self.FRAME = fps * 5  # 判定するフレーム数(5秒)
        self.RANDOMPOINTS = 50  # 1フレームで背景差分に打つポイント数
        self.EPS = 20  # クラスタリングコアの距離
        self.MINPOINTS = 5  # クラスタリング最小ポイント数
        self.FEATURE_CLASS = 5  # 特徴量出力のためのクラスター数（固定長にするため）
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.init_once = False
        self.N = 2  # トラッキングする長さ
        self.p = np.zeros((1, 1))  # トラッキングポイントのフレームカウントを保存
        self.p0 = np.zeros((1, 1, 2), dtype=np.float32)  # トラッキングポイント
        self.hist = np.zeros((1, self.N, 2), dtype=np.float32)

        self.stack_points = np.zeros((1, 2), dtype=np.float32)  # 判定用ポイント位置
        self.old_points = np.zeros((1, 2), dtype=np.float32)  # 判定用ポイント位置（1個前）
        self.stack_direction = np.zeros((1, 1), dtype=np.float32)  # 判定用方向
        self.labeling = labeling  # 学習データラベリングかどうか
        self.label = 0  # ラベルの値
        self.bounding = bounding  # 検知位置表示するか.処理を速くするなら表示しない
        self.count = 0  # フレームのカウント数
        self.period_no = 0  # 判定のカウント数
        self.logfile_l = None  # ラベル用ログファイル
        self.logfile_p = None  # ポイント位置ログファイル
        self.clf = pickle.load(
            open(find_exedir() + os.sep + "data/model.pkl", 'rb'))  # 学習モデル

    def close(self):
        if self.logfile_p is not None:
            self.logfile_p.close()
        if self.logfile_l is not None:
            self.logfile_l.close()

    def setTrackingPoints(self, gframe, bframe):
        u"""背景差分からトラックポイントの作成."""
        if cv2.findNonZero(bframe) is not None:
            # 背景差分にランダムに50ポイント新規トラッキングポイントを打つ
            tmp = np.float32(cv2.findNonZero(bframe)).reshape(-1, 1, 2)
            tmp = tmp[np.random.choice(tmp.shape[0], self.RANDOMPOINTS)]
        else:
            # 背景がない場合は、前回のポイントをひきつぐ（動いていないけどいる場合に対応）
            tmp = self.p0[self.p == 1].reshape(-1, 1, 2)
        if tmp is not None:
            self.p0 = np.vstack((self.p0, tmp))
            self.p = np.vstack((self.p, np.zeros((tmp.shape[0], 1))))
            # 新規ポイントを履歴0に代入
            a = np.zeros((tmp.shape[0], self.N, 2), dtype=np.float32)
            a[:, 0, :] = tmp.reshape(-1, 2)
            self.hist = np.vstack((self.hist, a))

    def apply(self, gframe, bframe):
        u"""トラックポイントの更新."""
        if not self.init_once:
            # 最初の一回だけ描画用のフレームを作成
            self.draw_frame = np.zeros((gframe.shape[0], gframe.shape[1], 3))
            self.init_once = True
        elif len(self.p0) != 0:
            # トラックポイントの更新
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gframe,
                                                   self.p0, None, **self.lk_params)

            # 良いトラッキングポイントだけを抽出
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            good_num = self.p[st == 1]
            a = np.repeat((st == 1), self.N).reshape(-1, self.N)
            good_hist = self.hist[a].reshape(-1, self.N, 2)
            # Nフレーム以前のトラッキングポイントは削除
            good_new = good_new[good_num < self.N]
            good_old = good_old[good_num < self.N]
            good_hist = good_hist[good_num < self.N]
            good_num = good_num[good_num < self.N]
            # numの更新と履歴の追加
            good_num += 1
            for i in range(1, self.N):
                good_hist[good_num == i, i, :] = good_new[good_num == i]

            self.p0 = good_new.reshape(-1, 1, 2)
            self.p = good_num.reshape(-1, 1)
            self.hist = good_hist
        self.setTrackingPoints(gframe, bframe)
        self.old_gray = gframe

    def check_animal(self):
        self.count += 1
        state = "NODETECT"
        N = 2
        cp = self.hist[(self.p == N - 1).ravel(), :, :]
        if(len(cp) > 0):
            # 点、距離、方向の保存
            cp_new = cp[:, N - 1, :]
            cp_old = cp[:, 0, :]
            diff = cp_new - cp_old
            distance = np.sum(diff**2, axis=1)**.5
            direction = np.rad2deg(
                np.arctan2(-diff[:, 1], diff[:, 0])) % 360  # 負の値を0-360に変換
            # 動かないポイントは除く
            MIN_DIST = 1
            dist_filter = (MIN_DIST < distance)
            self.stack_points = np.vstack(
                (self.stack_points, cp_new[dist_filter]))
            self.old_points = np.vstack((self.old_points, cp_old[dist_filter]))
            self.stack_direction = np.vstack(
                (self.stack_direction, direction[dist_filter].reshape(-1, 1)))
            # ポイント描画
            if self.bounding:
                # for (a, b), (c, d) in zip(cp_new[dist_filter], cp_old[dist_filter]):
                #    self.draw_arrow(self.draw_frame, (c, d),(a, b),(0, 0, 255))
                for (a, b) in cp_new[dist_filter]:
                    cv2.circle(self.draw_frame, (a, b), 1, (0, 0, 255), -1)

            # logfile_p書き出し
            if self.labeling:
                for x, y, dist, dire in zip(cp_new[:, 0], cp_new[:, 1], distance, direction):
                    y = self.height - y
                    self.logfile_p.write("{0:.0f},{1:.0f},{2:.0f},{3:.0f},{4:.0f}\n".format(
                        x, y, dist, dire, self.count))

        # 90フレーム(18fps：5秒間)に動いたポイントデータをクラスタリングして分析
        if(self.count % self.FRAME == 0):
            # 2次元位置でクラスタリング。（3次元の必要ない？）
            model = DBSCAN(eps=self.EPS, min_samples=self.MINPOINTS).fit(
                self.stack_points)
            cl_count = np.max(model.labels_) + 1
            #print("eps:",self.EPS,",minPts:", self.MINPOINTS)
            # クラスタがあれば、クラスタの情報を出力
            self.draw_frame *= 0
            col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                   (255, 0, 255), (120, 120, 120), (0, 0, 0)] * 10

            point_count = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            area = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            perimeter = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            long_axis = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            axis_direction = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            centroid = np.zeros((self.FEATURE_CLASS, 2))
            directivity = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            max_direction = np.zeros(self.FEATURE_CLASS, dtype=np.int32)
            max_direccount = np.zeros(self.FEATURE_CLASS, dtype=np.int32)

            for cl in range(self.FEATURE_CLASS):
                if(cl < cl_count):  # クラスターがあれば計算
                    cl_filter = (model.labels_ == cl)
                    # クラスタの面積、周長、長軸の長さ
                    points = self.stack_points[cl_filter]
                    hull = ConvexHull(points)
                    point_count[cl] = len(points)
                    area[cl] = hull.volume
                    perimeter[cl] = hull.area
                    # クラスタの長軸の長さ、位置、方向の計算
                    pairdist = pdist(points, 'euclidean')
                    D = squareform(pairdist)
                    I = np.argmax(D)
                    I_row, I_col = np.unravel_index(I, D.shape)
                    a = points[I_row]
                    b = points[I_col]
                    long_axis[cl] = np.max(pairdist)
                    axis_direction[cl] = np.rad2deg(
                        np.arctan2(-(a - b)[1], (a - b)[0])) % 180  # 0-180°の範囲で
                    centx = np.mean(points[:, 0])
                    centy = np.mean(points[:, 1])
                    centroid[cl] = (centx, centy)
                    # 方向の分布
                    data_direc = self.stack_direction[cl_filter]
                    #data_direc = data_direc[data_direc<0]+360
                    hist, bin_edges = np.histogram(
                        data_direc, bins=16, range=(0, 360))
                    # 指向性は、360°方位の反対に動いた差の総和。大きければ、どちらかに動き、小さければ風のように行ったり来たり
                    directivity[cl] = np.sum(
                        [abs(hist[i] - hist[i + 8]) for i in range(8)])
                    max_direction[cl] = np.argmax(hist) * 22.5
                    max_direccount[cl] = np.max(hist)
                    # 最外郭、長軸、クラスタ中心の描画
                    if self.bounding and not self.labeling:
                        cv2.polylines(self.draw_frame, [np.int32(
                            points[hull.vertices].reshape(-1, 1, 2))], True, col[cl], 2)
                        cv2.circle(self.draw_frame,
                                   (centx, centy), 3, col[cl], -1)
                        x0, y0 = a.ravel()
                        x1, y1 = b.ravel()
                        cv2.line(self.draw_frame, (x0, y0),
                                 (x1, y1), col[cl], 1)
                else:  # クラスターがなければ-1を出力
                    point_count[cl] = 0
                    area[cl] = 0
                    perimeter[cl] = 0
                    long_axis[cl] = 0
                    axis_direction[cl] = 0
                    centroid[cl] = 0
                    directivity[cl] = 0
                    max_direction[cl] = 0
                    max_direccount[cl] = 0
            # 平均クラス間距離(-1を除く)
            if cl_count > 1:
                pairdist = pdist(centroid[:cl_count, :], 'euclidean')
                cl_meandist = np.int(np.mean(pairdist))
            else:
                cl_meandist = 0

            # 手動判定
            # if (long_axis > 100).any() and (directivity > 70).any():
            #     label = 1
            # else:
            #     label = 0
            # 学習データのラベリング
            if self.labeling:
                label = self.label
            else:
                # 機械学習判定
                if cl_count > 0:
                    X = [cl_count, cl_meandist]
                    for i in range(2):  # モデルで使ったクラス数で評価
                        X.extend((point_count[i], area[i], perimeter[i], long_axis[i], axis_direction[
                                 i], directivity[i], max_direction[i], max_direccount[i]))
                    X = np.vstack(X).T
                    label = self.clf.predict(X)[0]
                else:
                    label = 0
            # ログ書き出し

            if self.labeling:
                print("{},{},{},{},".format(
                    self.period_no, label, cl_count, cl_meandist))
                self.logfile_l.write("{},{},{},{},".format(
                    self.period_no, label, cl_count, cl_meandist))
                # クラスをareaで降順にソートして、クラス情報を出力
                areaindexlist = np.argsort(area)[::-1]
                for i in areaindexlist:
                    print("{},{},{},{},{},{},{},{},".format(point_count[i], area[i], perimeter[i], long_axis[
                          i], axis_direction[i], directivity[i], max_direction[i], max_direccount[i]))
                    self.logfile_l.write("{},{},{},{},{},{},{},{},".format(point_count[i], area[i], perimeter[
                                         i], long_axis[i], axis_direction[i], directivity[i], max_direction[i], max_direccount[i]))
                self.logfile_l.write("\n")
            self.period_no += 1
            # 期間用変数の初期化
            self.stack_points = np.zeros((1, 2), dtype=np.float32)
            self.old_points = np.zeros((1, 2), dtype=np.float32)
            self.stack_direction = np.zeros((1, 1), dtype=np.float32)
            self.count = 0

            # 判定
            if label == 1:
                state = "DETECT"

        return state

    def detect(self, imgscale, logfile):
        state = "NODETECT"
        if self.labeling and self.logfile_l is None:
            # ポイント解析用
            self.logfile_p = open(logfile.replace('.csv', '_p.csv'), "w")
            self.logfile_p.write("x,y,dist,dire,pos\n")
            # 学習用
            self.logfile_l = open(logfile.replace('.csv', '_l.csv'), "w")
            self.logfile_l.write("期間番号,ラベル,クラス数,平均クラス間距離,")
            self.logfile_l.write(
                "ポイント数_1,面積_1,周長_1,長軸_1,長軸方向_1,指向性_1,最大方向_1,最大方向数_1,")
            self.logfile_l.write(
                "ポイント数_2,面積_2,周長_2,長軸_2,長軸方向_2,指向性_2,最大方向_2,最大方向数_2,")
            self.logfile_l.write(
                "ポイント数_3,面積_3,周長_3,長軸_3,長軸方向_3,指向性_3,最大方向_3,最大方向数_3,")
            self.logfile_l.write(
                "ポイント数_4,面積_4,周長_4,長軸_4,長軸方向_4,指向性_4,最大方向_4,最大方向数_4,")
            self.logfile_l.write(
                "ポイント数_5,面積_5,周長_5,長軸_5,長軸方向_5,指向性_5,最大方向_5,最大方向数_5,\n")

        # 動物の検知
        state = self.check_animal()

        return state

    def draw_arrow(self, image, p, q, color, arrow_magnitude=3, thickness=1, line_type=8, shift=0):
        # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html
        # draw arrow tail
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # calc angle of the arrow
        angle = np.arctan2(p[1] - q[1], p[0] - q[0])
        # starting point of first line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi / 4)),
             int(q[1] + arrow_magnitude * np.sin(angle + np.pi / 4)))
        # draw first half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # starting point of second line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi / 4)),
             int(q[1] + arrow_magnitude * np.sin(angle - np.pi / 4)))
        # draw second half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)

    def draw(self, cframe):

        if hasattr(self, "draw_frame"):
            # draw_frameを重ね合わせ（黒以外を置き換える）
            drawfilter = np.where((self.draw_frame[:, :, 0] != 0) | (
                self.draw_frame[:, :, 1] != 0) | (self.draw_frame[:, :, 2] != 0))
            cframe[drawfilter] = self.draw_frame[drawfilter]

        return cframe
