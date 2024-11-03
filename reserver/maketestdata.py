"""
波形分類のテストを行うためのデータ作成
"""

import numpy as np
import matplotlib.pyplot as plt  # matplotlibのimport修正

def go():
    # 0から119まで一つずつ増える配列
    up = np.arange(100)
    result1 = up

    # 50回繰り返し
    for _ in range(23):
        result1 = np.concatenate((result1, up))

    np.set_printoptions(threshold=np.inf)
    # print(result1)
    # print(len(result1))

    # 119から1ずつ0まで減っていく配列を作成
    do = np.arange(99, -1, -1)
    result2 = do

    # 50回繰り返し
    for _ in range(23):
        result2 = np.concatenate((result2, do))
    # print(result2)
    # print(len(result2))
    return result1,result2    
'''
    result = np.concatenate((result1, result2))
    
    print(result)
    print(len(result))

    d = result

    plt.figure()
    plt.plot(range(len(d)), d, color='blue', marker='o', linestyle='-', linewidth=0.5, label='Array 1')  # linewidthを追加


    plt.title('Line Plot of Array 1')
    plt.ylabel('nosex')
    plt.xlabel('flame')
    plt.legend()
    plt.grid(True)

    plt.show(block=True)
if __name__ == '__main__':
    go()
'''    

