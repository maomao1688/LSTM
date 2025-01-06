import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model



# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集，并对其中的缺失值进行填充
dataset = pd.read_csv('load.csv', index_col=0)
dataset = dataset.fillna(method='pad')


dataset = np.array(dataset)
# print(dataset)

# 将所有的数据放到一个列表里面，方便后续的训练集和测试集的制作
a = []
for item in dataset:
    for i in item:
        a.append(i)
dataset = pd.DataFrame(a)
# print(dataset)


# 获取前80%作为数据的训练集80%到90%作为验证集
# 划分训练集和验证集
test = dataset.iloc[int(len(a)*0.98):, [0]]


# 进行数据归一化，将数据归一化到0-1之间
scaler = MinMaxScaler(feature_range=(0, 1))
test = scaler.fit_transform(test)



"""
进行训练集数据特征和对应标签的划分，其中前面96个负荷特征
来预测第97个点的电力负荷值。
"""


# 设置测试集集的特征列表和对应标签列表
x_test = []
y_test = []


# 将前96个采样点的负荷特征作为训练特征添加到列表中
# 按照上述规律不断滑动取值
for i in np.arange(96, len(test)):
    x_test.append(test[i - 96:i, :])
    y_test.append(test[i])

# 将训练集由list格式变为array格式
x_test, y_test = np.array(x_test), np.array(y_test)
# 导入训练好的模型参数
model = load_model("LSTM_model.h5")


# 测试集输入模型进行预测
predicted = model.predict(x_test)
# print(predicted.shape)
# print(test.shape)

# 将真实值标签进行反归一化操作，方便后续和真实值进行对比
real = scaler.inverse_transform(y_test)


# 将模型预测出的值进行反归一化操作
prediction = scaler.inverse_transform(predicted)


# 绘制真实值和预测值对比图
# 创建一个大小为（12，8）的画布
plt.figure(figsize=(12, 8))

# 传入预测值和真实值
plt.plot(prediction, label='预测值')
plt.plot(real, label='真实值')

# 设置xy轴的刻度值大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# 设置xy轴的标签
plt.legend(loc='best', fontsize=15)
plt.ylabel('负荷值', fontsize=15)
plt.xlabel('采样点', fontsize=15)

# 设置图的参数，设置图的名字
plt.title("基于LSTM神经网络负荷预测", fontsize=15)
plt.show()




# 调用模型评价指标
# R2
from sklearn.metrics import r2_score
# MSE
from sklearn.metrics import mean_squared_error
# MAE
from sklearn.metrics import mean_absolute_error

# 计算模型的评价指标
R2 = r2_score(real, prediction)
MAE = mean_absolute_error(real, prediction)
RMSE = np.sqrt(mean_squared_error(real, prediction))
MAPE = np.mean(np.abs((real-prediction) / prediction))

# 打印模型的评价指标
print('R2:', R2)
print('MAE:', MAE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)














