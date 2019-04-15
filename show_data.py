import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

value_list = {
    'A': 0.934,
    'L': 0.825,
    'R': 0.962,
    'K': 1.04,
    'N': 0.986,
    'M': 0.804,
    'D': 0.994,
    'F': 0.773,
    'C': 0.9,
    'P': 1.047,
    'Q': 1.047,
    'S': 1.056,
    'E': 0.986,
    'T': 1.008,
    'G': 1.015,
    'W': 0.848,
    'H': 0.882,
    'Y': 0.931,
    'I': 0.766,
    'V': 0.825,
    'O': 0,
}

df = pd.read_csv('data/test_data.csv')

data1 = df[0:200][list(value_list.keys())]
data2 = df[200:400][list(value_list.keys())]

# print(data1.apply(sum).tolist())
# print(list(value_list.keys()))
# print(df.columns)
data3 = df[0:200][['vs1', 'vs2', 'vs3', 'vs4', 'vs5']]
data4 = df[200:400][['vs1',  'vs2', 'vs3', 'vs4', 'vs5']]
print(data3.mean().tolist())
print(data4.mean().tolist())


def show_acid_nums_img():
    # 构建数据
    is_able = data1.apply(sum).tolist()
    un_able = data2.apply(sum).tolist()
    labels = list(value_list.keys())
    bar_width = 0.35

    # 中文乱码的处理
    my_font = font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    # 绘图
    plt.bar(np.arange(21), is_able,label = '可磷酸化蛋白质序列',color='steelblue', alpha=0.8,width=bar_width)
    plt.bar(np.arange(21)+bar_width,un_able,label = '不可磷酸化蛋白质序列', color='indianred',alpha=0.8,width=bar_width)
    # 添加轴标签
    plt.xlabel('Different amino acids')
    plt.ylabel('sum')
    # 添加标题
    plt.title('Comparison of amino acids in two proteins')
    # 添加刻度标签
    plt.xticks(np.arange(21) + bar_width,labels)
    # 设置Y轴的刻度范围
    plt.ylim([0, 650])

    # 为每个条形图添加数值标签
    for k,is_able in enumerate(is_able):
        plt.text(k, is_able+20, '%s' %int(is_able))

    for k,un_able in enumerate(un_able):
        plt.text(k+bar_width, un_able+10, '%s' %int(un_able))
    # 显示图例
    plt.legend(prop=my_font)
    # 显示图形
    plt.show()
    plt.savefig('imgs/acid_nums.svg')

def show_acid_value_img():
    # 构建数据
    is_able = data3.mean().tolist()
    un_able = data4.mean().tolist()
    labels = ['vs1',  'vs2', 'vs3', 'vs4', 'vs5']
    bar_width = 0.35

    # 中文乱码的处理
    my_font = font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    # 绘图
    plt.bar(np.arange(5), is_able,label = '可磷酸化蛋白质序列',color='steelblue', alpha=0.8,width=bar_width)
    plt.bar(np.arange(5)+bar_width,un_able,label = '不可磷酸化蛋白质序列', color='indianred',alpha=0.8,width=bar_width)
    # 添加轴标签
    plt.xlabel('Different distance')
    plt.ylabel('mean value')
    # 添加标题
    plt.title('Comparison of amino acids in two proteins')
    # 添加刻度标签
    plt.xticks(np.arange(5) + bar_width,labels)
    # 设置Y轴的刻度范围
    plt.ylim([0.9, 1])

    # 为每个条形图添加数值标签
    for k,is_able in enumerate(is_able):
        plt.text(k, is_able+20, '%s' %(is_able))

    for k,un_able in enumerate(un_able):
        plt.text(k+bar_width, un_able+10, '%s' %(un_able))
    # 显示图例
    plt.legend(prop=my_font)
    # 显示图形
    plt.show()
    plt.savefig('imgs/acid_value.svg')

show_acid_value_img()