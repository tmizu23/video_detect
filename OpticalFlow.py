
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.csgraph._validation
import scipy.spatial.ckdtree
import pickle


class MotionDetection():

    def __init__(self,fps,labeling):
        #パラメータ
        self.FRAME = fps*5 #判定するフレーム数(5秒)
        self.RANDOMPOINTS = 50 #1フレームで背景差分に打つポイント数
        self.EPS = 20 #クラスタリングコアの距離
        self.MINPOINTS = 5 #クラスタリング最小ポイント数
        self.FEATURE_CLASS = 5 #特徴量出力のためのクラスター数（固定長にするため）

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0, 255, (10000, 3))
        self.init_once = False
        self.N = 2
        self.p = np.zeros((1, 1))
        self.hist = np.zeros((1, self.N, 2), dtype=np.float32)
        self.p0 = np.zeros((1, 1, 2), dtype=np.float32)
        self.detect_points = []
        self.stack_points = np.zeros((1, 2), dtype=np.float32)
        self.old_points = np.zeros((1, 2), dtype=np.float32)
        self.stack_direction = np.zeros((1, 1), dtype=np.float32)
        self.wind_points = []
        self.labeling = labeling #学習データラベリングかどうか
        self.label = 0
        self.count=0
        self.period_no=0
        self.logfile_l = None
        self.logfile_p = None
        # 5個でいいか？
        self.clf = pickle.load(open("C:/Users/mizutani/Desktop/video_detect/test.pkl",'rb'))


    def close(self):
        #if self.logfile_p is not None:
           #self.logfile_p.close()
        if self.logfile_l is not None:
           self.logfile_l.close()

    def set1stFrame(self, frame, bframe):
        if cv2.findNonZero(bframe) is not None:
            self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tmp = np.float32(cv2.findNonZero(bframe)).reshape(-1,1,2)
            tmp = tmp[np.random.choice(tmp.shape[0],self.RANDOMPOINTS)]
            if tmp is not None:
                self.p0 = np.vstack((self.p0, tmp))
                self.p = np.vstack((self.p, np.zeros((tmp.shape[0], 1))))
                a = np.zeros((tmp.shape[0], self.N, 2), dtype=np.float32)
                a[:, 0, :] = tmp.reshape(-1, 2)
                self.hist = np.vstack((self.hist, a))
            # p0に回数と距離、正負を加える

            # マスクの各領域の中心orgoodFeatureを追う
            #
            # Create a mask image for drawing purposes
            self.mask = np.zeros_like(frame)
            if not self.init_once:
                self.mask2 = np.zeros_like(frame)
            self.init_once = True

    def apply(self, frame, bframe):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # マスク
        if self.init_once and self.old_gray.shape == frame_gray.shape and len(self.p0) != 0:

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
                                                   self.p0, None, **self.lk_params)

            # Select good points
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

            # Now update the previous frame and previous points
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            self.p = good_num.reshape(-1, 1)
            self.hist = good_hist
            self.set1stFrame(frame, bframe)
        else:
            self.set1stFrame(frame, bframe)

        return

    def check_wind(self, imgscale):
        state = "NODETECT"
        N = 30
        # B 60フレームの間、存在できたトラッキングポイントに対して検知チェック
        # 風、車の抽出用
        cp = self.hist[(self.p == N-1).ravel(), :, :]
        if(len(cp) > 0):
            # A.始点、終点間移動距離
            cp_new = cp[:, N - 1, :]
            cp_old = cp[:, 0, :]  # 途中で迷子になった点を省くために、後半半分の距離でフィルタ
            distance = np.sum((cp_new - cp_old)**2, axis=1)**.5

            # B.トータル移動距離
            total_distance=np.zeros(cp.shape[0])
            # Nフレームの間にX軸、Y軸の一定方向に行った回数
            for i in range(N - 1):
                cp_new = cp[:, i + 1, :]
                cp_old = cp[:, i, :]
                v = np.absolute(cp_new - cp_old)
                total_distance += np.sum(v**2, axis=1)**.5
            #self.logfile.write("{0:},{1:.0f},{2:.0f},".format(len(distance),np.max(distance),np.max(total_distance)))

            detect_filter = (total_distance > distance*1.5) & (len(distance)>=5)

            detect_points = cp[detect_filter] #こちらはself.detect_pointsではない

            if len(detect_points) > 0:
                state = "DETECT"
                print("{}WIND!!!!!!!!".format(len(detect_points)))
        else:
            pass
            #self.logfile.write(",,,")
        return state
#
    def check_car(self,imgscale):
        state = "NODETECT"
        N = 90#50
        # B 60フレームの間、存在できたトラッキングポイントに対して検知チェック
        # 風、車の抽出用
        cp = self.hist[(self.p == N-1).ravel(), :, :]
        if(len(cp) > 0):
            # A.移動距離でフィルタ作成
            cp_new = cp[:, N - 1, :]
            cp_old = cp[:, 0, :]  # 途中で迷子になった点を省くために、後半半分の距離でフィルタ
            distance = np.sum((cp_new - cp_old)**2, axis=1)**.5
            MIN_DIST = 6*imgscale
            MAX_DIST = 40*imgscale #200*imgscale
            dist_filter = (MIN_DIST < distance) & (distance < MAX_DIST)
            # B.移動方向でフィルタ作成
            x_pos = np.zeros(cp.shape[0])
            y_pos = np.zeros(cp.shape[0])
            x_neg = np.zeros(cp.shape[0])
            y_neg = np.zeros(cp.shape[0])
            # Nフレームの間にX軸、Y軸の一定方向に行った回数
            for i in range(N - 1):
                cp_new = cp[:, i + 1, :]
                cp_old = cp[:, i, :]
                v = cp_new - cp_old
                x_pos = x_pos + (v[:, 0] >= 0)
                y_pos = y_pos + (v[:, 1] >= 0)
                x_neg = x_neg + (v[:, 0] <= 0)
                y_neg = y_neg + (v[:, 1] <= 0)
            # 一定方向に行った回数の条件でフィルタ作成
            CN = 2
            direction_filter = (np.absolute(x_pos-x_neg) <= CN) & (np.absolute(y_pos-y_neg) <= CN)
            #direction_filter = (x_pos >= CN) | (
            #    y_pos >= CN) | (x_neg >= CN) | (y_neg >= CN)
            #print("")
            #print(np.max(distance))
            #print(np.max([np.max(x_pos),np.max(y_pos),np.max(x_neg),np.max(y_neg)]))
            #print(np.max(x_pos))
            #print(np.max(y_pos))
            #print(np.max(x_neg))
            #print(np.max(y_neg))
            # 距離フィルタと方向フィルタで検知フィルタを作成
            detect_filter = dist_filter & direction_filter
            self.wind_points = cp[detect_filter] #こちらはself.detect_pointsではない
            #self.wind_points = cp
            #self.wind_points = self.wind_points[dist_filter]
            #self.wind_points = self.wind_points[direction_filter]

            if len(self.wind_points) > 1:
                state = "DETECT"
                print("{}WIND!!!!!!!!".format(len(self.wind_points)))
        else:
            self.wind_points = []
        return state


    def check_animal(self,imgscale,curpos,height):
        state = "NODETECT"
        N = 2 #90#30
        # A 30フレームの間、存在できたトラッキングポイントに対して検知チェック
        # 動物の抽出用
        cp = self.hist[(self.p == N-1).ravel(), :, :]
        if(len(cp) > 0):
            # A.移動距離でフィルタ作成
            cp_new = cp[:, N - 1, :]
            cp_old = cp[:, 0, :]  # 途中で迷子になった点を省くために、後半半分の距離でフィルタ
            diff = cp_new - cp_old
            distance = np.sum((cp_new - cp_old)**2, axis=1)**.5
            direction = np.rad2deg(np.arctan2(-diff[:,1],diff[:,0]))
            MIN_DIST = 1#10*imgscale
            MAX_DIST = 10*N*imgscale
            dist_filter = (MIN_DIST < distance) & (distance < MAX_DIST)
            # B.移動方向でフィルタ作成
            x_pos = np.zeros(cp.shape[0])
            y_pos = np.zeros(cp.shape[0])
            x_neg = np.zeros(cp.shape[0])
            y_neg = np.zeros(cp.shape[0])
            # Nフレームの間にX軸、Y軸の一定方向に行った回数
            for i in range(N - 1):
                cp_new = cp[:, i + 1, :]
                cp_old = cp[:, i, :]
                v = cp_new - cp_old
                x_pos = x_pos + (v[:, 0] > 0)
                y_pos = y_pos + (v[:, 1] > 0)
                x_neg = x_neg + (v[:, 0] < 0)
                y_neg = y_neg + (v[:, 1] < 0)
            # 一定方向に行った回数の条件でフィルタ作成
            CN = N-1#25
            direction_filter = (x_pos >= CN) | (
                y_pos >= CN) | (x_neg >= CN) | (y_neg >= CN)

            #self.logfile_p.write("{0:.0f},{1:.0f},".format(np.max(distance),np.max([np.max(x_pos),np.max(y_pos),np.max(x_neg),np.max(y_neg)])))
            for x,y,dist,dire in zip(cp_new[:,0],cp_new[:,1],distance,direction):
                if dist>1:
                   self.logfile_p.write("{0:.0f},{1:.0f},{2:.0f},{3:.0f},{4:.0f}\n".format(x,height-y,dist,dire,curpos))

            # 距離フィルタと方向フィルタで検知フィルタを作成
            detect_filter = dist_filter & direction_filter
            #self.detect_points = cp[detect_filter]
            self.detect_points = cp
            self.detect_points = self.detect_points[dist_filter]
            #print(np.max(distance))
            #print(np.max([np.max(x_pos),np.max(y_pos),np.max(x_neg),np.max(y_neg)]))
            #self.detect_points = self.detect_points[direction_filter]

            if len(self.detect_points) > 0:
                #print("{}ANIMAL!!!!!!!!".format(len(self.detect_points)))
                state = "DETECT"
        else:
            self.detect_points = []
            #self.logfile_p.write(",,")
        return state

    def check_animal2(self):
        self.count+=1
        state = "NODETECT"
        N = 2
        cp = self.hist[(self.p == N-1).ravel(), :, :]
        if(len(cp) > 0):
            # 点、距離、方向の保存
            cp_new = cp[:, N - 1, :]
            cp_old = cp[:, 0, :]
            diff = cp_new - cp_old
            distance = np.sum(diff**2, axis=1)**.5
            direction = np.rad2deg(np.arctan2(-diff[:,1],diff[:,0])) % 360 #負の値を0-360に変換
            #動かないポイントは除く
            MIN_DIST = 1
            dist_filter = (MIN_DIST < distance)
            self.stack_points = np.vstack((self.stack_points,cp_new[dist_filter]))
            self.old_points = np.vstack((self.old_points,cp_old[dist_filter]))
            self.stack_direction = np.vstack((self.stack_direction,direction[dist_filter].reshape(-1,1)))
            #logfile_p書き出し
            #for x,y,dist,dire in zip(cp_new[:,0],cp_new[:,1],distance,direction):
                #y=height-y
            #    self.logfile_p.write("{0:.0f},{1:.0f},{2:.0f},{3:.0f},{4:.0f}\n".format(x,y,dist,dire,curpos))

        #90フレーム(18fps：5秒間)に動いたポイントデータをクラスタリングして分析
        if(self.count % self.FRAME == 0):
            # 2次元位置でクラスタリング。（3次元の必要ない？）
            model = DBSCAN(eps=self.EPS,min_samples=self.MINPOINTS).fit(self.stack_points)
            cl_count = np.max(model.labels_)+1
            #print("eps:",self.EPS,",minPts:", self.MINPOINTS)
            #クラスタがあれば、クラスタの情報を出力
            self.mask2 *=0
            col = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(120,120,120),(0,0,0)]*10

            point_count=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            area=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            perimeter=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            long_axis=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            axis_direction=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            centroid=np.zeros((self.FEATURE_CLASS,2))
            directivity=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            max_direction=np.zeros(self.FEATURE_CLASS,dtype=np.int32)
            max_direccount=np.zeros(self.FEATURE_CLASS,dtype=np.int32)

            for cl in range(self.FEATURE_CLASS):
                if(cl < cl_count): #クラスターがあれば計算
                    cl_filter = (model.labels_== cl)
                    #クラスタの面積、周長、長軸の長さ
                    points = self.stack_points[cl_filter]
                    hull = ConvexHull(points)
                    point_count[cl] = len(points)
                    area[cl] = hull.volume
                    perimeter[cl] = hull.area
                    #クラスタの長軸の長さ、位置、方向の計算
                    pairdist = pdist(points, 'euclidean')
                    D = squareform(pairdist)
                    I = np.argmax(D)
                    I_row, I_col = np.unravel_index(I, D.shape)
                    a = points[I_row]
                    b = points[I_col]
                    long_axis[cl] = np.max(pairdist)
                    axis_direction[cl] = np.rad2deg(np.arctan2(-(a-b)[1],(a-b)[0])) % 180 #0-180°の範囲で
                    centx=np.mean(points[:,0])
                    centy=np.mean(points[:,1])
                    centroid[cl] = (centx,centy)
                    #方向の分布
                    data_direc = self.stack_direction[cl_filter]
                    #data_direc = data_direc[data_direc<0]+360
                    hist, bin_edges = np.histogram(data_direc, bins=16,range=(0,360))
                    #指向性は、360°方位の反対に動いた差の総和。大きければ、どちらかに動き、小さければ風のように行ったり来たり
                    directivity[cl] = np.sum([abs(hist[i]-hist[i+8]) for i in range(8)])
                    max_direction[cl] = np.argmax(hist)*22.5
                    max_direccount[cl] = np.max(hist)
                    #最外郭、長軸、クラスタ中心の描画
                    if not self.labeling:
                        cv2.polylines(self.mask2, [np.int32(points[hull.vertices].reshape(-1,1,2))],True,col[cl],2)
                        cv2.circle(self.mask2, (centx, centy), 3, col[cl], -1)
                        x0, y0 = a.ravel()
                        x1, y1 = b.ravel()
                        cv2.line(self.mask2, (x0,y0), (x1,y1), col[cl], 1)
                else: #クラスターがなければ-1を出力
                    point_count[cl] = 0
                    area[cl] = 0
                    perimeter[cl] = 0
                    long_axis[cl] = 0
                    axis_direction[cl] = 0
                    centroid[cl] = 0
                    directivity[cl] = 0
                    max_direction[cl] = 0
                    max_direccount[cl] = 0
            #平均クラス間距離(-1を除く)
            if cl_count > 1:
                pairdist = pdist(centroid[:cl_count,:], 'euclidean')
                cl_meandist = np.int(np.mean(pairdist))
            else:
                cl_meandist = 0

            #手動判定
            # if (long_axis > 100).any() and (directivity > 70).any():
            #     label = 1
            # else:
            #     label = 0
            # 学習データのラベリング
            if self.labeling:
                label = self.label
            else:
                #機械学習判定
                if cl_count > 0:
                    X = [cl_count,cl_meandist]
                    for i in range(2):#モデルで使ったクラス数で評価
                       X.extend((point_count[i],area[i],perimeter[i],long_axis[i],axis_direction[i],directivity[i],max_direction[i],max_direccount[i]))
                    X = np.vstack(X).T
                    label = self.clf.predict(X)[0]
                else:
                    label = 0
            print(label)
            #ログ書き出し
            print("{},{},{},{},".format(self.period_no,label,cl_count,cl_meandist))
            if self.labeling:
                self.logfile_l.write("{},{},{},{},".format(self.period_no,label,cl_count,cl_meandist))
                #クラスをareaで降順にソートして、クラス情報を出力
                areaindexlist = np.argsort(area)[::-1]
                for i in areaindexlist:
                    print("{},{},{},{},{},{},{},{},".format(point_count[i],area[i],perimeter[i],long_axis[i],axis_direction[i],directivity[i],max_direction[i],max_direccount[i]))
                    self.logfile_l.write("{},{},{},{},{},{},{},{},".format(point_count[i],area[i],perimeter[i],long_axis[i],axis_direction[i],directivity[i],max_direction[i],max_direccount[i]))
                self.logfile_l.write("\n")
            self.period_no +=1
            #期間用変数の初期化
            self.stack_points = np.zeros((1, 2), dtype=np.float32)
            self.old_points = np.zeros((1, 2), dtype=np.float32)
            self.stack_direction = np.zeros((1, 1), dtype=np.float32)
            self.count=0

            #判定
            if label==1:
                state = "DETECT"

        return state

    def detect(self,imgscale,height,logfile):
        state = "NODETECT"
        exist = False
        state_animal = "NODETECT"
        state_wind = "NODETECT"
        if self.logfile_l is None:
            #self.logfile_p = open(logfile.replace('.csv','_p.csv'),"w")
            self.logfile_l = open(logfile.replace('.csv','_l.csv'),"w")
            #self.logfile_p.write("x,y,dist,dire,pos\n")
            self.logfile_l.write("期間番号,ラベル,クラス数,平均クラス間距離,")
            self.logfile_l.write("ポイント数_1,面積_1,周長_1,長軸_1,長軸方向_1,指向性_1,最大方向_1,最大方向数_1,")
            self.logfile_l.write("ポイント数_2,面積_2,周長_2,長軸_2,長軸方向_2,指向性_2,最大方向_2,最大方向数_2,")
            self.logfile_l.write("ポイント数_3,面積_3,周長_3,長軸_3,長軸方向_3,指向性_3,最大方向_3,最大方向数_3,")
            self.logfile_l.write("ポイント数_4,面積_4,周長_4,長軸_4,長軸方向_4,指向性_4,最大方向_4,最大方向数_4,")
            self.logfile_l.write("ポイント数_5,面積_5,周長_5,長軸_5,長軸方向_1,指向性_5,最大方向_5,最大方向数_5,\n")



        # A 30フレームの間、存在できたトラッキングポイントに対して検知チェック
        # 動物の抽出用
        #state_animal = self.check_animal(imgscale,height)
        state_animal = self.check_animal2()
        # B 60フレームの間、存在できたトラッキングポイントに対して検知チェック
        # 風、車の抽出用
        #state_wind = self.check_car(imgscale)
        #self.logfile_p.write("{},{}\n".format(state_animal, state_wind))
        # 最終判断
        if state_animal == "DETECT" and state_wind == "NODETECT":
            state = "DETECT"
            exist = True

        return state

    def draw_points(self,cframe,detect_points,N,color):
        # draw the tracks
        # N個連続のもの
        if(len(detect_points) > 0):
            #_, cnts, _ = cv2.findContours(bframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for contour in cnts:
            #    (x, y, w, h) = cv2.boundingRect(contour)
            #    cv2.rectangle(cframe,(x,y),(x+w,y+h), (255, 0, 0), 2)
            good_new = detect_points[:, N - 1, :]
            good_old = detect_points[:, 0, :]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.draw_arrow(cframe, (c, d),(a, b),(0, 0, 255))
                # self.mask2 = cv2.line(
                #     self.mask2, (a, b), (c, d), (255, 255, 255), 1)
                # self.mask2 = cv2.circle(self.mask2, (a, b), 5, color, -1)
            img = cv2.add(cframe, self.mask2)
        elif hasattr(self,"mask2"):
            img = cv2.add(cframe, self.mask2)
        else:
            img = cframe
        return img


    def draw_arrow(self,image, p, q, color, arrow_magnitude=3, thickness=1, line_type=8, shift=0):
        # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

        # draw arrow tail
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # calc angle of the arrow
        angle = np.arctan2(p[1]-q[1], p[0]-q[0])
        # starting point of first line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
        int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
        # draw first half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)
        # starting point of second line of arrow head
        p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
        int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
        # draw second half of arrow head
        cv2.line(image, p, q, color, thickness, line_type, shift)

    def draw_all(self, cframe,bframe):
        # draw the tracks
        # if bframe is not None:
        #     _, cnts, _ = cv2.findContours(bframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for contour in cnts:
        #        (x, y, w, h) = cv2.boundingRect(contour)
        #        cv2.rectangle(cframe,(x,y),(x+w,y+h), (255, 0, 0), 2)
        # N個連続のもの
        N = 2
        cp = self.hist[(self.p == N).ravel(), :, :]
        if(len(cp) > 0):

            good_new = cp[:, N - 1, :]
            good_old = cp[:, 0, :]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.draw_arrow(cframe, (c, d),(a, b),(0, 0, 255))
                # self.mask2 = cv2.line(
                #     self.mask2, (a, b), (c, d), (255, 255, 255), 1)
                # self.mask2 = cv2.circle(
                #     self.mask2, (c, d), 1, (0, 0, 255), -1)
            img = cv2.add(cframe, self.mask2)
        elif hasattr(self,"mask2"):
            img = cv2.add(cframe, self.mask2)
        else:
            img = cframe
        return img

    def draw_all2(self, cframe,bframe):

        if(len(self.stack_points) > 0):
            for (a, b),(c, d) in zip(self.stack_points,self.old_points):
                cv2.circle(cframe, (a, b), 1, (0, 0, 255), -1)
                #self.draw_arrow(cframe, (c, d),(a, b),(0, 0, 255))
        if hasattr(self,"mask2"):
           blackfilter = np.where((self.mask2[:,:,0]!=0)|(self.mask2[:,:,1]!=0)|(self.mask2[:,:,2]!=0))
           cframe[blackfilter] = self.mask2[blackfilter]

        return cframe

    def draw(self, cframe, bframe):
        red =(0,0,255)
        blue  =(255,0,0)
        #img = self.draw_points(cframe,self.stack_points,2,red)
        #img = self.draw_points(cframe, self.wind_points, 90, red)
        img = self.draw_all2(cframe,bframe)
        return img
