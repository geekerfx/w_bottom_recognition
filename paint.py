import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import json
import numpy as np


def get_EMA(df, N):
    for i in range(len(df)):
        if i == 0:
            df.ix[i, 'ema'] = df.ix[i, 'close']
        if i > 0:
            df.ix[i, 'ema'] = (2 * df.ix[i, 'close'] + (N - 1) * df.ix[i - 1, 'ema']) / (N + 1)
    ema = list(df['ema'])
    return ema


def get_MACD(df, short=12, long=26, M=9):
    a = get_EMA(df, short)
    b = get_EMA(df, long)
    df['diff'] = np.array(a) - np.array(b)

    for i in range(len(df)):
        if i == 0:
            df.ix[i, 'dea'] = df.ix[i, 'diff']
        if i > 0:
            df.ix[i, 'dea'] = (2 * df.ix[i, 'diff'] + (M - 1) * df.ix[i - 1, 'dea']) / (M + 1)
    df['macd'] = 2 * (df['diff'] - df['dea'])
    return df


def ma(n, data):
    ma = []
    for i in range(len(data) - 2 - n):
        ma.append(np.mean(data[i:i + n]))
    ma = [np.nan] * (len(data) - len(ma)) + ma
    return ma


def add_k(df, ax):
    """向图中添加K线图"""
    quotes = []
    current_day = 0
    for idx in range(len(df)):  # 舍弃前边五天的数据
        line = df.iloc[idx]
        # time open high low close
        quotes.append((current_day, line['open'], line['high'], line['low'], line['close']))
        current_day += 1
    candlestick_ohlc(ax, quotes, colorup='r', colordown='g', width=0.5)


def add_volume(df, ax):
    """向图中添加成交量，采用柱状图"""
    ax.bar([i for i in range(len(df))], df["volume"], color="b")

def add_ma5(df, ax):
    """向图中添加五日均线"""
    close = df['close']
    result = []
    for i in range(5, len(close)):
        result.append(sum(close[i - 5: i]) / 5)
    ax.plot(close, color="b", linewidth=1)


def add_macd(df, ax):
    """向图中添加macd曲线"""
    df = get_MACD(df)
    ax.plot(df["macd"], color="g", linewidth=1)


def add_ema(df, ax):
    """添加ema曲线"""
    df = get_MACD(df)
    ax.plot(df["ema"], linewidth=1)

def add_diff(df, ax):
    """添加diff曲线"""
    df = get_MACD(df)
    ax.plot(df["diff"], linewidth=1)


def add_dea(df, ax):
    df = get_MACD(df)
    ax.plot(df["dea"], linewidth=1)


def paint(csv_path, conf, idx, label):
    # 添加指标曲线对应的函数
    indicators = {"k": add_k, "volume": add_volume, "ma5": add_ma5, "macd": add_macd,
                  "eam": add_ema, "diff": add_diff, "dea": add_dea}
    df = pd.read_csv(csv_path)
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    channel_num = conf["channel_num"]  # 一共几个子图
    channels = conf["channels"]  # 每个子图中包含的指标
    fig, axs = plt.subplots(nrows=channel_num, ncols=1, figsize=(4, 4))
    fig.subplots_adjust(top=1, bottom=0.05, right=1, left=0.05, hspace=0.01)
    for c_idx, channel in enumerate(channels):  # 对每一个子图进行绘制
        for indicator_name in channel:  # 向子图中添加指标曲线
            indicators[indicator_name](df, axs[c_idx])
    for ax in axs:  # 关闭坐标轴显示
        ax.axis("off")
    out_path = conf["out_path"]
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(os.path.join(out_path, "%d_img%d.png" % (label, idx)))
    plt.show()


def main():
    conf = json.load(open("conf/paint.json"))
    # paint("data/csvfiles/1_000001.XSHE2010-11-24.csv", conf, 0, 0)  # for test
    csv_path = conf["csv_path"]
    print("Start to paint...")
    for idx, csv_name in enumerate(os.listdir(csv_path)):
        label = int(csv_name.split("_")[0])
        paint(os.path.join(csv_path, csv_name), conf, idx, label)
    print("Painting done!")


if __name__ == '__main__':
    main()
