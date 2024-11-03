#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hosakaで実行
指定した部位、確度、キック方向の入力データ表示
ランダムソートなし
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from model import ESN, Tikhonov
from datetime import datetime
import os



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
    bui = input("部位: ")
    kakudo = input("確度: ")
    sayuu = input("L or R: ")
    #lebel = input("レベル(1 or 2): ")
    lebel = str(2)

    T = int(tr_n/2) * frame_n # トレーニングデータ数の半分
    T2 = int(ts_n/2) * frame_n # テストデータのデータ数の半分

    if lebel == '1':
        tr_n = 54
        ts_n = 54
        dn1 = -5400
        dn2 = 5400

    if lebel =='2':
        tr_n = 82
        ts_n = 82
        dn1 = -8200
        dn2 = 8200


    # CSVファイルのパスを指定
    Path1 = 'openposewin/out_csv/1201/1201r_' + kakudo + ".csv"
    Path2 = 'openposewin/out_csv/1201/1201l_' + kakudo + ".csv"
    Path3 = 'openposewin/out_csv/1208/1208r_' + kakudo + ".csv"
    Path4 = 'openposewin/out_csv/1208/1208l_' + kakudo + ".csv"
    Path5 = 'openposewin/out_csv/1211/1211r_' + kakudo + ".csv"
    Path6 = 'openposewin/out_csv/1211/1211l_' + kakudo + ".csv"
        
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

    """
    arr1 = arr1[:100]
    arl1 = arl1[:100]
    arr2 = arr2[:100]
    arl2 = arl2[:100]
    arr3 = arr3[:100]
    arl3 = arl3[:100]
    """


    # 値の補正
    #arr1,arl1,arr2,arl2,arr3,arl3 = hosei(arr1,arl1,arr2,arl2,arr3,arl3)

    R = np.hstack((arr1,arr2,arr3))
    L = np.hstack((arl1,arl2,arl3))

    if sayuu == 'R':
        plt.figure(figsize=(15, 5))
        #plt.subplot(2, 1, 1)  # 2行1列の1番目のサブプロット
        plt.plot(range(len(R)), R, color='blue', linestyle='-', label='Array 1')
        #plt.ylim(0, 1300)
        plt.title('R',fontsize=13)

    if sayuu == 'L':
        plt.figure(figsize=(15, 5))
        #plt.subplot(2, 1, 2)  # 2行1列の1番目のサブプロット
        plt.plot(range(len(L)), L, color='blue', linestyle='-', label='Array 1')
        #plt.ylim(0, 1300)
        plt.title('L',fontsize=13)

    base_folder = "sozai"
    file_prefix = "bui_{}_{}_{}_{}.pdf"

    file_name = os.path.join(base_folder, file_prefix.format(bui, kakudo, threshold,sayuu))

    #plt.savefig(file_name)

    print(f'グラフが {file_name} として保存されました。')
    plt.show()
