from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import pandas as pd
import datetime
import os
import numpy as np
import warnings
import json


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


def ma5(data):
    close = data['close']
    result = []
    for i in range(5, len(close)):
        result.append(sum(close[i-5: i]) / 5)
    return result


def paint0(root_path, csv_name, out_path, index, label=None):
    """
    将指定的csv文件包含数据绘制成图片
    :param root_path 需要绘制的文件的根路径
    :param csv_name: csv文件名称
    :param out_path: 图片的输出路径
    :param index:  图片的名称索引
    :param label: 图片对应的分类
    :return:  None
    """
    df = pd.read_csv(os.path.join(root_path, csv_name))
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    quotes = []
    times = []
    for idx in range(5, len(df)):  # 舍弃前边五天的数据
        line = df.iloc[idx]
        time = date2num(datetime.datetime.strptime(line['date'], '%Y-%m-%d'))
        # time open high low close
        times.append(time)
        quotes.append((time, line['open'], line['high'], line['low'], line['close']))
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
    fig.subplots_adjust(top=1, bottom=0.05, right=1, left=0.05, hspace=0.01)
    # ma5
    axs[0].plot(times, ma5(df), color='#4169E1', linewidth=1)
    # K线图
    candlestick_ohlc(axs[0], quotes, colorup='r', colordown='g', width=0.5)
    # 成交量
    axs[1].bar(times, df['volume'][5:])
    # diff,dea和macd
    line_datas = get_MACD(df)[5:]
    axs[2].plot(line_datas['diff'], color='r', linewidth=1)
    axs[2].plot(line_datas['dea'], color='g', linewidth=1)
    axs[2].plot(line_datas['macd'], color='b', linewidth=1)
    for ax in axs:
        ax.axis('off')
    if not label:
        label = int(csv_name.split("_")[0])
    fig.savefig(os.path.join(os.path.join(out_path, label), '%d_img%d.png' % (label, index)))
    # plt.show()
    plt.close()


def main():
    conf1 = json.load(open('conf/paint_preprocess.json', 'r', encoding='UTF-8'))['paint']
    # 绘图部分
    paint_funcs = {0: paint0}
    pfunc = paint_funcs[conf1['img_type']]  # 使用的绘图函数
    src_path = conf1['src_path']  # 需要绘制的csv文件的路径
    out_path = conf1['out_path']  # 绘好的图片输出的根路径
    for i, csvname in enumerate(os.listdir(src_path)):
        pfunc(src_path, csvname, out_path, i)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
