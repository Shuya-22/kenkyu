"""
hosakaで実行
撮影日による誤差を軽減する前と後の比較グラフを保存
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

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

def hosei(R1,R2,R3):
    # 配列の要素の平均値を計算
    av_r01 = np.mean(R1)
    av_r08 = np.mean(R2)
    av_r11 = np.mean(R3)

    AVr = (av_r01 + av_r08 + av_r11)/3

    der01 = av_r01 - AVr
    der08 = av_r08 - AVr
    der11 = av_r11 - AVr

    R1 = R1 - der01
    R2 = R2 - der08
    R3 = R3 - der11


    return R1,R2,R3


# CSVファイルのパスを指定
Path1 = 'openposewin/out_csv/1201/1201r_0.5.csv'
Path2 = 'openposewin/out_csv/1208/1208r_0.5.csv'
Path3 = 'openposewin/out_csv/1211/1211r_0.5.csv'


# CSVファイルを読み込み、各列ごとのリストを取得
P1 = read_csv(Path1)
P2 = read_csv(Path2)
P3 = read_csv(Path3)

T=150
R1 = P1['LElbow_x']
R2 = P2['LElbow_x']
R3 = P3['LElbow_x']

# 外れ値処理
R1 = fix_outliers(R1,T)
R2 = fix_outliers(R2,T)
R3 = fix_outliers(R3,T)

d1 = 2400
d2 = 5400
d3 = 8200

R4 = np.hstack((R1,R2,R3))
R4 = np.copy(R4)
R4_1 = R4[0:d1]
R4_2 = R4[d1:d2+1]
R4_3 = R4[d2:d3+1]

R1,R2,R3 = hosei(R1,R2,R3)

R5 = np.hstack((R1,R2,R3))
R5_1 = R5[0:d1]
R5_2 = R5[d1:d2+1]
R5_3 = R5[d2:d3+1]

x_partial = [i for i in range(8200)]



plt.figure(figsize=(20, 8))
plt.subplot(2, 1, 1)  # 2行1列の1番目のサブプロット
plt.plot( x_partial, R4, color='blue', linestyle='-', label='Day 1')
plt.plot( x_partial[d1:d2+1] ,R4_2, color='red', linestyle='-', label='Day 2')
plt.plot( x_partial[d2:d3+1], R4_3, color='green', linestyle='-', label='Day 3')
plt.legend(fontsize=18)
plt.title('Before processing')

plt.subplot(2, 1, 2)  # 2行1列の1番目のサブプロット
plt.plot( x_partial, R5, color='blue', linestyle='-', label='Day 1')
plt.plot( x_partial[d1:d2+1] ,R5_2, color='red', linestyle='-', label='Day 2')
plt.plot( x_partial[d2:d3+1], R5_3, color='green', linestyle='-', label='Day 3')
plt.legend(fontsize=18)
plt.title('Before processing')

#plt.savefig("sozai/gosa.pdf")
plt.show(block=True)

 