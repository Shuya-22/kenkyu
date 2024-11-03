"""
hosakaで表示
確度の閾値によるデータの違いを表示
部位は左肘X
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

#kakudo = input("確度: ")
kakudo = str(0.5)

# CSVファイルのパスを指定
Path1 = 'openposewin/out_csv/1201/1201r_0.1.csv'
Path2 = 'openposewin/out_csv/1201/1201r_0.5.csv'
Path3 = 'openposewin/out_csv/1201/1201r_0.9.csv'


# CSVファイルを読み込み、各列ごとのリストを取得
No1 = read_csv(Path1)
No2 = read_csv(Path2)
No3 = read_csv(Path3)


"""
# 各列ごとのデータを表示
for header, values in data.items():
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f'{header}: {values}')
"""


T=100
AR1 = No1['LElbow_x']
AR2 = No2['LElbow_x']
AR3 = No3['LElbow_x']

AR1 = AR1[:300]
AR2 = AR2[:300]
AR3 = AR3[:300]



plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)  # 2行1列の1番目のサブプロット
plt.plot(range(len(AR1)), AR1, color='blue', linestyle='-', label='Array 1')
plt.ylim(0, 1300)
plt.title('Threshold = 0.1',fontsize=13)


plt.subplot(3, 1, 2)  # 2行1列の1番目のサブプロット
plt.plot(range(len(AR2)), AR2, color='blue', linestyle='-', label='Array 1')
plt.title('Threshold = 0.5',fontsize=13)

plt.subplot(3, 1, 3)  # 2行1列の1番目のサブプロット
plt.plot(range(len(AR3)), AR3, color='blue', linestyle='-', label='Array 1')
plt.title('Threshold = 0.9',fontsize=13)


#plt.savefig("sozai/kakudo.pdf")


plt.show(block=True)

 