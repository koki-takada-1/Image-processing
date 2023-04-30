# 補完テスト
import math
import sys

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def affine(src, af, interp):
    height, width = src.shape[0:2]
    a, b, c, d, e, f = (
        af[0][0],
        af[0][1],
        af[1][0],
        af[1][1],
        af[0][2],
        af[1][2],
    )

    maxx = (int)(
        max([a * width + e, b * height + e, a * width + b * height + e, e])
    )
    maxy = (int)(
        max([c * width + f, d * height + f, c * width + d * height + f, f])
    )
    # 第一引数は画像、第二引数はアフィン変換行列、第三引数は画像の大きさ(横軸ピクセル×縦軸ピクセル)、第四引数は補完方法
    dst = cv2.warpAffine(src, af, (maxx, maxy), flags=interp)
    return dst


if __name__ == "__main__":
    img = cv2.imread("lena_q.bmp")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fig = plt.figure(dpi=100, figsize=(8, 8))

    # 全部縦横2倍
    # 1 nearest neighbor
    af = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    dst = affine(img, af, cv2.INTER_NEAREST)
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("1.nearest neighbor", loc="center")
    plt.imshow(dst)

    # 2 bilinear
    af = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    dst = affine(img, af, cv2.INTER_LINEAR)
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("2.bilinear", loc="center")
    plt.imshow(dst)

    # 3 cubic
    af = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    dst = affine(img, af, cv2.INTER_CUBIC)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("3.cubic", loc="center")
    plt.imshow(dst)
    plt.show()
