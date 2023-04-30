# affine transformation
import math
import sys

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rotate(center, angle):
    affine_trans = cv2.getRotationMatrix2D(center, angle, 1.0)
    return affine_trans


def affine(src, af):
    height, width = src.shape[0:2]
    a, b, c, d, e, f = (
        af[0][0],
        af[0][1],
        af[1][0],
        af[1][1],
        af[0][2],
        af[1][2],
    )
    # maxx及びmaxyは、 a,b,c,dの正負全パターンの中の最大値. 定数項は必ず足される.
    # affine変換した画像に応じて、平面拡張
    maxx = (int)(
        max([a * width + e, b * height + e, a * width + b * height + e, e])
    )
    maxy = (int)(
        max([c * width + f, d * height + f, c * width + d * height + f, f])
    )

    dst = cv2.warpAffine(src, af, (maxx, maxy), flags=cv2.INTER_CUBIC)
    return dst


if __name__ == "__main__":
    img = cv2.imread("color/lena.bmp")

    # BGRからRGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fig = plt.figure(dpi=100, figsize=(8, 8))
    # fig.subplots_adjust()はいらない、後からmatplotlibのGUIでパラメータ調整できるから
    ax = fig.add_subplot(3, 3, 2)
    ax.set_title("source", loc="center")
    plt.imshow(img)

    # 1 平行移動
    af = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]])
    dst = affine(img, af)
    ax = fig.add_subplot(3, 3, 4)
    ax.set_title("1.Translation", loc="center")
    plt.imshow(dst)

    # 2 拡大、縮小
    af = np.array([[0.5, 0.0, 0.0], [0.0, 2.0, 0.0]])
    dst = affine(img, af)
    ax = fig.add_subplot(3, 3, 5)
    ax.set_title("2.Scaling", loc="center")
    plt.imshow(dst)

    # 3 回転
    center = (0, 0)
    angle = 30
    af = rotate(center, angle)
    dst = affine(img, af)
    ax = fig.add_subplot(3, 3, 6)
    ax.set_title("3.Rotation", loc="center")
    plt.imshow(dst)

    # 4 スキュー
    # ※　tan30度と表現したい時、math.tan(30)としないこと弧度法を採用してるため30ラジアンになってしまう。
    af = np.array([[1.0, math.tan(math.radians(30)), 0.0], [0.0, 1.0, 0.0]])
    dst = affine(img, af)
    ax = fig.add_subplot(3, 3, 7)
    ax.set_title("4.Skew", loc="center")
    plt.imshow(dst)

    # 5 フリップ
    af = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 256.0]])
    dst = affine(img, af)
    ax = fig.add_subplot(3, 3, 8)
    ax.set_title("5.Flip", loc="center")
    plt.imshow(dst)

    # 6 合成
    translation = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]])
    scaling = np.array([[0.5, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = rotate((0, 0), 30)
    skew = np.array([[1.0, math.tan(math.radians(30)), 0.0], [0.0, 1.0, 0.0]])
    flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 256.0]])
    # 平行移動、拡大縮小、回転、スキュー、反転を合成
    dst = affine(
        affine(
            affine(affine(affine(img, translation), scaling), rotation), flip
        ),
        skew,
    )
    ax = fig.add_subplot(3, 3, 9)
    ax.set_title("6.combination", loc="center")
    plt.imshow(dst)

    plt.show()
