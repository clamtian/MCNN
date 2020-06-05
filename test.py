import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def sigmoid(x):
    # 直接返回sigmoid函数
    return np.where(x < 0, 0.2 * x, x)


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    path = "./test_save/Set9/psnr_save/"
    filenames = os.listdir(path)
    max = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(filenames.__len__()):
        data = pd.read_csv(path + filenames[i])
        print(data)
        col = data.columns.values[0]
        print(col)
        data = data[col].values.tolist()
        print(data)
        for j in range(len(max)):
            if max[j] < data[j]:
                max[j] = data[j]
        print(max)