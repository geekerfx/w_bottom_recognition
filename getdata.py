from jqdatasdk import auth, get_price
import pandas as pd
import os
import json

def get_stock_data(stock_code, end_date, count, out_path, label):
    """
    根据股票代码，结束日期，时长下载对应的股票数据，以csv格式保存，输出到指定路径
    :param stock_code: 股票代码
    :param end_date: 结束日期
    :param count: 时长
    :param out_path: 输出路径
    :param label:  股票数据对应的标签
    """
    df = get_price(stock_code, end_date=end_date, count=count)
    df.to_csv(os.path.join(out_path, '%d_%s.csv' % (label, (stock_code+end_date))))


def main():
    conf = json.load(open('conf/getdata.json'))
    auth(conf['username'], conf['password'])
    out_path = conf['out_path']
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    src = pd.read_csv(conf['src'], names=['stock_code', 'count', 'end_date', 'label'])
    print("Downloading...")
    for i in src.index:
        line = src.iloc[i]
        get_stock_data(line['stock_code'], line['end_date'], int(line['count']), out_path, int(line['label']))
    print("Done!")
if __name__ == '__main__':
    main()
