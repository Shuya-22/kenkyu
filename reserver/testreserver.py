#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
from model import ESN, Tikhonov
from datetime import datetime
import maketestdata



# 出力のスケーリング
class ScalingShift:
    def __init__(self, scale, shift):
        '''
        :param scale: 出力層のスケーリング（scale[n]が第n成分のスケーリング）
        :param shift: 出力層のシフト（shift[n]が第n成分のシフト）
        '''
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

def replace_outliers(arr, threshold):   #arr: 配列　　　threshold: 閾値
    #result_array = arr.copy()  # 元の配列を変更せずに新しい配列を作成

    for i in range(1, len(arr) - 1):
        if arr[i] == 0:
            for j in range(i,len(arr-1)):
                if arr[j] != 0:
                    avg_value = (arr[i-1] + arr[j]) / 2.0
                    arr[i] = avg_value
    return arr


if __name__ == '__main__':
    tr_n = 24       # トレーニング回数
    ts_n = 24       # テスト回数
    frame_n = 100   # 一回のフレーム数
    threshold = 500 # 外れ値処理の閾値



    # データをNumpy配列に直して格納
    Nosexr, Nosexl = maketestdata.go()
    
    # 外れ値処理
    Nosexr = replace_outliers(Nosexr, threshold)
    Nosexl = replace_outliers(Nosexl, threshold)


    T = int(tr_n/2) * frame_n # トレーニングデータ数の半分
    T2 = int(ts_n/2) * frame_n # テストデータのデータ数の半分


    # トレーニング用のデータ（前半T個）
    Nosexr_tr_h = Nosexr[:T]
    Nosexl_tr_h = Nosexl[:T]

    Nosexr_tr_v = Nosexr[:T].reshape(-1, 1)
    Nosexl_tr_v = Nosexl[:T].reshape(-1, 1)

    # テスト用データ
    Nosexr_ts_h = Nosexr[T:]
    Nosexl_ts_h = Nosexl[T:]

    Nosexr_ts_v = Nosexr[T:].reshape(-1, 1)
    Nosexl_ts_v = Nosexl[T:].reshape(-1, 1)

    # 結合
    Nosexall_h = np.hstack((Nosexr, Nosexl))    # 全データ（横一行）
    Nosexall_v = np.vstack((Nosexr, Nosexl))    # 全データ（縦一列）

    Nosexall_tr_h = np.hstack((Nosexr_tr_h, Nosexl_tr_h))   
    Nosexall_tr_v = np.vstack((Nosexr_tr_v, Nosexl_tr_v))   # トレーニング用データ

    Nosexall_ts_h = np.hstack((Nosexr_ts_h, Nosexl_ts_h))
    Nosexall_ts_v = np.vstack((Nosexr_ts_v, Nosexl_ts_v))   # テスト用データ

    # ラベル生成
    label1_tr = np.zeros((T, 2))
    label1_tr[:, 1] = 1.0
    label2_tr = np.zeros((T, 2))
    label2_tr[:, 0] = 1.0
    label_tr = np.vstack((label1_tr, label2_tr))    # トレーニング用ラベル

    label1_ts = np.zeros((T2, 2))
    label1_ts[:, 1] = 1.0
    label2_ts = np.zeros((T2, 2))
    label2_ts[:, 0] = 1.0
    label_ts = np.vstack((label1_ts, label2_ts))    # テスト用ラベル

    alllabel = np.vstack((label_tr, label_ts))
    #print(alllabel)


    # 時系列入力データ生成


    # 訓練・検証用情報
    train_U = Nosexall_tr_v         # トレーニング用データ
    train_D = label_tr              # トレーニング用ラベル

    test_U = Nosexall_ts_v           # テスト用おデータ
    test_D = label_ts                # テスト用ラベル

    # 出力のスケーリング関数
    output_func = ScalingShift([0.5, 0.5], [0.5, 0.5])
    """
    scale: 出力層のスケーリングを指定します。[0.5, 0.5]という具体的な値が与えられています。これは、出力の各成分に対して0.5倍することを意味します。
    shift: 出力層のシフトを指定します。同様に、[0.5, 0.5]という具体的な値が与えられています。これは、出力の各成分に対して0.5を加算することを意味します。
    """

    # ESNモデル
    N_x = 50  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                input_scale=0.2, rho=0.9, fb_scale=0.05, 
                output_func=output_func, inv_output_func=output_func.inverse, 
                classification = True, average_window=frame_n)
    '''
        def __init__(self, N_u, N_y, N_x, density=0.05, input_scale=1.0,
                 rho=0.95, activation_func=np.tanh, fb_scale = None,
                 fb_seed=0, noise_level = None, leaking_rate=1.0,
                 output_func=identity, inv_output_func=identity,
                 classification = False, average_window = None):
        param N_u: 入力次元
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param density: リザバーのネットワーク結合密度
        param input_scale: 入力スケーリング
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: リザバーノードの活性化関数
        param fb_scale: フィードバックスケーリング（default: None）
        param fb_seed: フィードバック結合重み行列生成に使う乱数の種
        param leaking_rate: leaky integratorモデルのリーク率
        param output_func: 出力層の非線形関数（default: 恒等写像）
        param inv_output_func: output_funcの逆関数
        param classification: 分類問題の場合はTrue（default: False）
        param average_window: 分類問題で出力平均する窓幅（default: None）
    '''

    # 学習（リッジ回帰）
    train_Y = model.train(train_U, train_D, 
                          Tikhonov(N_x, train_D.shape[1], 0.1)) 
    # Tikhonv→リザバーコンピューティングのモデルにおいて、正則化された状態から得られる出力結合重み行列が取得できます。
    
    '''
    def train(self, U, D, optimizer, trans_len = None):
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力, データ長×N_y
    '''
    '''
    def __init__(self, N_x, N_y, beta):
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param beta: 正則化パラメータ
    '''

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

    # グラフ表示用データ
    T_disp = (-2400, 2400)
    t_axis = np.arange(T_disp[0], T_disp[1])  # 時間軸
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]])) # トレーニングデータとテストデータ
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]])) # トレーニングラベルとテストラベル
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]])) # 学習前のモデルによる出力と学習後のモデルによる出力

    #print(disp_Y[1000])

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.1, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.1, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.ylabel('Output 1')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax3 = fig.add_subplot(3, 1, 3)
    plt.plot(t_axis, disp_D[:, 1], color='k', linestyle='-', label='Target')
    plt.plot(t_axis, disp_Y[:, 1], color='gray', linestyle='--', label='Model')
    plt.plot([-500, 500], [0.5, 0.5], color='k', linestyle = ':')
    plt.ylim([-0.3, 1.3])
    plt.xlabel('n')
    plt.ylabel('Output 2')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    current_date = datetime.now().strftime('%Y%m%d')

# ファイル名に日付を組み込んで保存する
    file_name = f'sozai/PK_1201_1.png'
    plt.savefig(file_name)

    print(f'グラフが {file_name} として保存されました。')
    plt.show()
