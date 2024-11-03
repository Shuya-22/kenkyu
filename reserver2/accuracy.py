#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hosakaで実行
すべての部位でX座標を用いて10回ずつ波形分類実行
それぞれの精度と10回の平均を表示
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from model import ESN, Tikhonov
from datetime import datetime
import os
import random



# 出力のスケーリング
class ScalingShift:
    def __init__(self, scale, shift):
        self.scale = np.diag(scale)
        self.shift = np.array(shift)
        self.inv_scale = np.linalg.inv(self.scale)
        self.inv_shift = -np.dot(self.inv_scale, self.shift)

    def __call__(self, x):
        return np.dot(self.scale, x) + self.shift

    def inverse(self, x):
        return np.dot(self.inv_scale, x) + self.inv_shift

def read_csv(file_path):
    # 各列ごとのリストを格納するための辞書を作成
    columns = {}

    with open(file_path, 'r', encoding='utf-8') as csv_file:
        # CSVファイルを読み込む
        csv_reader = csv.reader(csv_file)

        # ヘッダー行を読み込み、各列ごとのリストを初期化
        headers = next(csv_reader)
        for header in headers:
            columns[header] = []

        # 各行のデータを読み込み、各列のリストに追加
        for row in csv_reader:
            for header, value in zip(headers, row):
                columns[header].append(float(value))

    return columns

def fix_outliers(ar,T):
    arr = ar.copy()

    for i in range(1,len(arr)):
        if abs(arr[i]-arr[i-1])>T:
            for j in range(i+1,len(arr)):
                if abs(arr[i-1]-arr[j])<T:
                    arr[i] = (arr[i-1]+arr[j])/2
                    break
    return arr

def hosei(r1201ar,l1201ar,r1208ar,l1208ar,r1211ar,l1211ar):
    # 配列の要素の平均値を計算
    av_r01 = np.mean(r1201ar)
    av_l01 = np.mean(l1201ar)
    av_r08 = np.mean(r1208ar)
    av_l08 = np.mean(l1208ar)
    av_r11 = np.mean(r1211ar)
    av_l11 = np.mean(l1211ar)

    AVr = (av_r01 + av_r08 + av_r11)/3
    AVl = (av_l01 + av_l08 + av_l11)/3

    der01 = av_r01 - AVr
    del01 = av_l01 - AVl

    der08 = av_r08 - AVr
    del08 = av_l08 - AVl

    der11 = av_r11 - AVr
    del11 = av_l11 - AVl

    r1201ar = r1201ar - der01
    l1201ar = l1201ar - del01
    r1208ar = r1208ar - der08
    l1208ar = l1208ar - del08
    r1211ar = r1211ar - der11
    l1211ar = l1211ar - del11


    R = np.hstack((r1201ar, r1208ar, r1211ar))
    L = np.hstack((l1201ar, l1208ar, l1211ar))
    return r1201ar,l1201ar,r1208ar,l1208ar,r1211ar,l1211ar


if __name__ == '__main__':
    tr_n = 82       # トレーニング回数
    ts_n = 82       # テスト回数
    frame_n = 100   # 一回のフレーム数
    threshold = 150 # 外れ値処理の閾値
    #bui = input("部位: ")
    #kakudo = input("確度: ")
    kakudo = str(0.5)
    #lebel = input("レベル(1 or 2): ")
    lebel = str(2)

    kaisuu = 30
    average = np.zeros(25)
    for q in range(25):
        kekka = np.zeros((kaisuu))
        if q == 0:
                bui ="Nose_x"
        if q == 1:
                bui ="Neck_x"
        if q == 2:
                bui ="RShoulder_x"
        if q == 3:
                bui ="RElbow_x"
        if q == 4:
                bui ="RWrist_x"
        if q == 5:
                bui ="LShoulder_x"
        if q == 6:
                bui ="LElbow_x"
        if q == 7:
                bui ="LWrist_x"
        if q == 8:
                bui ="MidHip_x"
        if q == 9:
                bui ="RHip_x"
        if q == 10:
                bui ="RKnee_x"
        if q == 11:
                bui ="RAnkle_x"
        if q == 12:
                bui ="LHip_x"
        if q == 13:
                bui ="LKnee_x"
        if q == 14:
                bui ="REye_x"
        if q == 15:
                bui ="LAnkle_x"
        if q == 16:
                bui ="LEye_x"
        if q == 17:
                bui ="REar_x"
        if q == 18:
                bui ="LEar_x"
        if q == 19:
                bui ="LBigToe_x"
        if q == 20:
                bui ="LSmallToe_x"
        if q == 21:
                bui ="LHeel_x"
        if q == 22:
                bui ="RBigToe_x"
        if q == 23:
                bui ="RSmallToe_x"
        if q == 24:
                bui ="RHeel_x"
        for p in range(kaisuu):
            import random
            T = int(tr_n/2) * frame_n # トレーニングデータ数の半分
            T2 = int(ts_n/2) * frame_n # テストデータのデータ数の半分
            np.set_printoptions(threshold=np.inf)
            if lebel == '1':
                tr_n = 54
                ts_n = 54
                dn1 = -5400
                dn2 = 5400

            if lebel =='2':
                tr_n = 82
                ts_n = 82
                dn1 = -3000
                dn2 = 3000



            # CSVファイルのパスを指定
            Path1 = 'B4/openposewin/out_csv/1201/1201r_' + kakudo + ".csv"
            Path2 = 'B4/openposewin/out_csv/1201/1201l_' + kakudo + ".csv"
            Path3 = 'B4/openposewin/out_csv/1208/1208r_' + kakudo + ".csv"
            Path4 = 'B4/openposewin/out_csv/1208/1208l_' + kakudo + ".csv"
            Path5 = 'B4/openposewin/out_csv/1211/1211r_' + kakudo + ".csv"
            Path6 = 'B4/openposewin/out_csv/1211/1211l_' + kakudo + ".csv"
                
            # CSVファイルを読み込み、各列ごとのリストを取得
            datar_1201 = read_csv(Path1)
            datal_1201 = read_csv(Path2)

            datar_1208 = read_csv(Path3)
            datal_1208 = read_csv(Path4)

            datar_1211 = read_csv(Path5)
            datal_1211 = read_csv(Path6)

            # データをNumpy配列に直して格納
            arr1 = np.array(datar_1201[bui])
            arl1 = np.array(datal_1201[bui])

            arr2 = np.array(datar_1208[bui])
            arl2 = np.array(datal_1208[bui])

            arr3 = np.array(datar_1211[bui])
            arl3 = np.array(datal_1211[bui])



            # 外れ値処理
            arr1 = fix_outliers(arr1, threshold)
            arl1 = fix_outliers(arl1, threshold)
            arr2 = fix_outliers(arr2, threshold)
            arl2 = fix_outliers(arl2, threshold)
            arr3 = fix_outliers(arr3, threshold)
            arl3 = fix_outliers(arl3, threshold)

            # 値の補正
            arr1,arl1,arr2,arl2,arr3,arl3 = hosei(arr1,arl1,arr2,arl2,arr3,arl3)

            # トレーニング用のデータ（前半T個）
            rtr1 = arr1[:12*100]
            ltr1 = arl1[:12*100]
            rtr2 = arr2[:15*100]
            ltr2 = arl2[:15*100]
            rtr3 = arr3[:14*100]
            ltr3 = arl3[:14*100]

            # テスト用データ
            rts1 = arr1[12*100:]
            lts1 = arl1[12*100:]
            rts2 = arr2[15*100:]
            lts2 = arl2[15*100:]
            rts3 = arr3[14*100:]
            lts3 = arl3[14*100:]

            if lebel == '1':
                trR = np.vstack((rtr1, rtr2))
                tsR = np.vstack((rts1, rts2))

                trL = np.vstack((ltr1, ltr2))
                tsL = np.vstack((lts1, lts2))

                TR = np.vstack((trR, trL))
                TS = np.vstack((tsR, tsL))

            if lebel == '2':
                trR = np.hstack((rtr1, rtr2, rtr3))
                tsR = np.hstack((rts1, rts2, rts3))

                trL = np.hstack((ltr1, ltr2, ltr3))
                tsL = np.hstack((lts1, lts2, lts3)) 
            

            # 二次元配列を格納するリストを初期化
            TR = []
            label_tr = np.zeros((T2*2, 2))

            # 100個ずつのサブ配列を作成して二次元配列に格納
            TRR = np.array_split(trR, len(trR) // 100)
            TRL = np.array_split(trL, len(trL) // 100)


            r = 0
            l = 0
            random1 = random.sample(range(1, 83), 82)

            for i in range(82):
                if random1[i] % 2 == 0:
                    TR.append(TRR[r])
                    for j in range(100):
                        label_tr[i*100+j, 1] = 1.0
                    r += 1

                else :
                    TR.append(TRL[l])
                    for j in range(100):
                        label_tr[i*100+j, 0] = 1.0
                    l += 1
            TR = np.hstack(TR)
            TR = TR.reshape(-1,1)

            
            # 二次元配列を格納するリストを初期化
            TS = []
            label_ts = np.zeros((T2*2, 2))

            # 100個ずつのサブ配列を作成して二次元配列に格納
            TSR = np.array_split(tsR, len(tsR) // 100)
            TSL = np.array_split(tsL, len(tsL) // 100)


            r = 0
            l = 0
            random = random.sample(range(1, 83), 82)

            for i in range(82):
                if random[i] % 2 == 0:
                    TS.append(TSR[r])
                    for j in range(100):
                        label_ts[i*100+j, 1] = 1.0
                    r += 1

                else :
                    TS.append(TSL[l])
                    for j in range(100):
                        label_ts[i*100+j, 0] = 1.0
                    l += 1
            TS = np.hstack(TS)
            TS = TS.reshape(-1,1)


            # 訓練・検証用情報
            train_U = TR         # トレーニング用データ
            train_D = label_tr         # トレーニング用ラベル

            test_U = TS         # テスト用データ
            test_D = label_ts          # テスト用ラベル

            # 出力のスケーリング関数
            #output_func = ScalingShift([50, 50], [50, 50])
            output_func = ScalingShift([0.5, 0.5], [0.5, 0.5])


            # ESNモデル
            N_x = 200 # リザバーのノード数
            model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                        input_scale=0.2, rho=0.9, fb_scale=None, 
                        output_func=output_func, inv_output_func=output_func.inverse, 
                        classification = True, average_window=frame_n)


            # 学習（リッジ回帰）
            train_Y = model.train(train_U, train_D, 
                                Tikhonov(N_x, train_D.shape[1], 0.1)) 


            # 訓練データに対するモデル出力
            test_Y = model.predict(test_U)  # 学習後のモデル出力
            #print(test_Y)

            # 評価（正解率, accracy）
            mode = np.empty(0, np.int)
            for i in range(ts_n):
                tmp = test_Y[frame_n*i:frame_n*(i+1), :]  # 各ブロックの出力
                max_index = np.argmax(tmp, axis=1)  # 最大値をとるインデックス
                histogram = np.bincount(max_index)  # そのインデックスのヒストグラム
                mode = np.hstack((mode, np.argmax(histogram)))  #  最頻値      

            target = test_D[0:frame_n*ts_n:frame_n,1]
            accuracy = 1-np.linalg.norm(mode.astype(np.float)-target, 1)/ts_n
            print('accuracy =', accuracy)
            kekka[p] = float(accuracy)
            #print(test_Y)

            """
            # グラフ表示用データ
            T_disp = (dn1, dn2)
            t_axis = np.arange(T_disp[0], T_disp[1])  # 時間軸
            disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]])) # トレーニングデータとテストデータ
            disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]])) # トレーニングラベルとテストラベル
            disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]])) # 学習前のモデルによる出力と学習後のモデルによる出力

            #print(disp_Y[1000])

            # グラフ表示
            plt.rcParams['font.size'] = 9
            fig = plt.figure(figsize=(7, 5))
            plt.subplots_adjust(hspace=0.4)

            ax1 = fig.add_subplot(2, 1,1)
            ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
            ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
            ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
            plt.plot(t_axis, disp_U[:,0], color='k')
            plt.ylabel('Input')
            plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
            plt.plot(t_axis, disp_D[:,0], color='k', linestyle='-', label='Target')
            plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
            plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
            plt.ylim([-0.3, 1.3])
            plt.ylabel('Output 1')
            plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
            plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

            current_date = datetime.now().strftime('%Y%m%d')

        # ファイル名に日付を組み込んで保存する
            #file_name = f'LWrist_x_1217.png'
            date_time = datetime.now()

            # 日付だけを取得
            current_date = date_time.date()

            # フォーマットを指定して日付を文字列として取得
            formatted_date = current_date.strftime("%m-%d")
            folder_name = "sozai/reserver/" + bui 
            file_name = folder_name + "/" + bui + formatted_date + "_rand" + ".pdf"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            #plt.savefig(file_name)

            #print(f'グラフが {file_name} として保存されました。')
            #plt.show()
            """
        average[q] = sum(kekka) / len(kekka)
        average[q] = int(average[q] * 100) / 100  # 2桁目を切り捨てる
        print(average)
