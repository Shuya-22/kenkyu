#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# waveform_classification.py: 本書の図4.5に対応するサンプルコード 
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# 正弦波とのこぎり波の混合波形生成
class SinSaw:
    def __init__(self, period):
        self.period = period  # 周期

    # 正弦波
    def sinusoidal(self):
        n = np.arange(self.period)
        x = np.sin(2*np.pi*n/self.period)

        return x

    # のこぎり波
    def saw_tooth(self):
        n = np.arange(self.period)
        x = 2*(n/self.period - np.floor(n/self.period+0.5))

        return x

    def make_output(self, label):
        y = np.zeros((self.period, 2))
        y[:, label] = 1

        return y

    # 混合波形及びラベルの出力
    def generate_data(self, label):
        '''
        :param label: 0または1を要素に持つリスト
        :return: u: 混合波形
        :return: d: 2次元ラベル（正弦波[1,0], のこぎり波[0,1]）
        '''
        u = np.empty(0)
        d = np.empty((0, 2))
        for i in label:
            if i:
                u = np.hstack((u, self.saw_tooth()))
            else:
                u = np.hstack((u, self.sinusoidal()))
            d = np.vstack((d, self.make_output(i)))

        return u, d


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


if __name__ == '__main__':

    # 訓練データ，検証データの数
    n_wave_train = 10
    n_wave_test = 10
    np.set_printoptions(threshold=np.inf)
    # 時系列入力データ生成
    period = 50
    dynamics = SinSaw(period)   #dynamics:周期50の生成関数たち
    label = np.random.choice(2, n_wave_train+n_wave_test)
    u, d = dynamics.generate_data(label)    # u:波形データ d:ラベル
    #print(d)
    print(d)
    T = period*n_wave_train                 # T:トレーニングデータの総時間ステップ数

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, 1)          #最初のT個のデータを取得して一列のベクトルに
    train_D = d[:T]                         #ラベル切り出し

    test_U = u[T:].reshape(-1, 1)           #T個以降の切り出して一列のベクトルに
    test_D = d[T:]                          #ラベル切り出し


