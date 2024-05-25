import os
import time

import mindspore.ops.operations as P
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
import pandas as pd
from mindspore import nn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 因为运行的时候好像有什么包冲突所以加一句这个，和模型本身无关

from mindspore import dataset as ds
from mindspore.nn import LSTM, Flatten, Dense
from mindspore.ops import mean

# 定义LSTM的结构
timesteps = 1
features = 11
embedding_dim = 8
n_hidden = 8



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]  # 变量的个数
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):  # n_in代表输入往下退的次数，相当于把后n个挤掉了
        cols.append(df.shift(i))  # 将数据向下移动i格但是索引值不变
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]  # 存放变量名
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):  # 把后n+1个作为输出
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    # 然后将输入输出合并 Var1（t-1）那一列代表的就是前t-1时刻var1的时间序列之后作为输入
    # var1(t)那一列代表后t时刻（t，t+1...t+n）的var1的时间序列，是之后的输出序列
    agg = pd.concat(cols, axis=1)
    agg.columns = names  # 为序列命名
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)  # 是否过滤空数据
    return agg


def attention_3d_block2(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    a = ms.Tensor(inputs[0], ms.float16)
    input_dim = 8  # 8
    a = a.permute((0, 2, 1))  # a=None,8,1,第三维和第二维转置
    a = Dense(in_channels=1, out_channels=timesteps, activation='softmax', dtype=ms.float32)(ms.Tensor(a, ms.float32))  # 经过一层全连接层
    if SINGLE_ATTENTION_VECTOR:  # SINGLE_ATTENTION_VECTOR=True，则共享一个注意力权重，如果=False则每维特征会单独有一个权重，换而言之，注意力权重也变成多维的了。
        a = mean(a, axis=1)
        a = a.repeat(input_dim)
    a_probs = a.permute((0, 2, 1))
    output_attention_mul = ms.ops.mul(a, a_probs)  # 两个矩阵元素分别相乘
    return output_attention_mul


data1 = pd.read_csv('../Data/bid5_label_dataNew_outlierRemoved.csv')
# del data1['baterryNo']
del data1['recordtime']

# data = data1[['V', 'A','T','R','SOC']]
# 对date,direction编码an
values = data1.values
encoder = preprocessing.LabelEncoder()  # LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码
values[:, 0] = encoder.fit_transform(values[:, 0])
values[:, 1] = encoder.fit_transform(values[:, 1])
# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
values_minmax = min_max_scaler.fit_transform(values)

agg = series_to_supervised(values_minmax, timesteps, 1)  # 构造成前timesteps个小时作为输入，下一个时刻作为输出
# 预测值只有var3（t），去掉不需要的列
agg.drop(agg.columns[[-2, -3]], axis=1, inplace=True)
# temp=agg[agg.columns[[5,6,11]]]
agg.head()
values = agg.values

print('数据量', values.shape[0])
n = int(values.shape[0]/4*3)
# 分割训练集与测试集（3/4，1/4）
X_train = values[0:n, 0:-1]
print(X_train.shape)
y_train = values[0:n, -1]
X_test = values[n:, 0:-1]
y_test = values[n:, -1]

# 构造成LSTM需要的输入格式
print(X_train.shape[1])
X_train = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))


# 构造mindspore模型能够使用的数据集
class IterableDataset():
    # 可迭代数据集
    # 可以通过迭代的方式逐步获取数据样本
    def __init__(self, data, label):
        '''init the class object to hold the data'''
        self.data = data
        self.label = label

        self.start = 0
        self.end = len(data)

    def __iter__(self):
        self.start = 0
        self.end = len(self.data)
        return self

    def __next__(self):
        if self.start >= self.end:
            raise StopIteration
        data = self.data[self.start]
        label = self.label[self.start]
        self.start += 1
        return data, label


def datapipe(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    return dataset


train_dataset = IterableDataset(X_train, y_train)
train_dataset = ds.GeneratorDataset(train_dataset, column_names=["data", "label"])
train_dataset = datapipe(train_dataset, 72)

train_dataset1 = IterableDataset(X_train, y_train)
train_dataset1 = ds.GeneratorDataset(train_dataset1, column_names=["data", "label"])
train_dataset1 = datapipe(train_dataset1, 72)

test_dataset = IterableDataset(X_test, y_test)
test_dataset = ds.GeneratorDataset(test_dataset, column_names=["data", "label"])
test_dataset = datapipe(test_dataset, 72)


print('\nX_train.shape, y_train.shape, X_test.shape, y_test.shape:')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')


class myModel(nn.Cell):
    def __init__(self, timesteps, features):
        super(myModel, self).__init__()
        # lstm layer
        self.lstm = LSTM(input_size=timesteps * features, hidden_size=n_hidden, dropout=0.2, num_layers=1)
        self.attention = ms.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.flatten = Flatten()
        self.dense = Dense(8, 1)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute((0, 2, 1))
        hidden = final_state.view(-1, n_hidden, 1)  # hidden : [batch_size, n_hidden, 1(=n_layer)]
        # print(lstm_output.shape)
        # print(hidden.shape)
        attn_weights = ms.ops.matmul(lstm_output.squeeze(2), hidden.squeeze(0))  # attn_weights : [batch_size, n_step]
        soft_attn_weights = ms.ops.Softmax(1)(attn_weights)
        # print(lstm_output.shape)
        # print(soft_attn_weights.shape)
        # [batch_size, n_hidden * num_directions(=1), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = ms.ops.matmul(lstm_output, soft_attn_weights.expand_dims(2))
        context = context.squeeze(2)
        return context, soft_attn_weights  # context : [batch_size, n_hidden * num_directions(=1)]

    def construct(self, x):
        x = ms.Tensor(x, ms.float32)
        x, (hn, cn) = self.lstm(x)
        # x, attention = self.attention_net(x, hn)
        # x, weight = self.attention(x, x, x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits

class myModel1(nn.Cell):
    def __init__(self, timesteps, features):
        super(myModel1, self).__init__()
        # lstm layer
        self.lstm = LSTM(input_size=timesteps * features, hidden_size=n_hidden, dropout=0.2, num_layers=1)
        self.attention = ms.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.flatten = Flatten()
        self.dense = Dense(8, 1)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute((0, 2, 1))
        hidden = final_state.view(-1, n_hidden, 1)  # hidden : [batch_size, n_hidden, 1(=n_layer)]
        # print(lstm_output.shape)
        # print(hidden.shape)
        attn_weights = ms.ops.matmul(lstm_output.squeeze(2), hidden.squeeze(0))  # attn_weights : [batch_size, n_step]
        soft_attn_weights = ms.ops.Softmax(1)(attn_weights)
        # print(lstm_output.shape)
        # print(soft_attn_weights.shape)
        # [batch_size, n_hidden * num_directions(=1), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = ms.ops.matmul(lstm_output, soft_attn_weights.expand_dims(2))
        context = context.squeeze(2)
        return context, soft_attn_weights  # context : [batch_size, n_hidden * num_directions(=1)]

    def construct(self, x):
        x = ms.Tensor(x, ms.float32)
        x, (hn, cn) = self.lstm(x)
        x, attention = self.attention_net(x, hn)
        # x, weight = self.attention(x, x, x)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


SINGLE_ATTENTION_VECTOR = False

net = myModel(timesteps, features)
net1 = myModel1(timesteps, features)

opt = nn.Adam(params=net.trainable_params())
opt1 = nn.Adam(params=net1.trainable_params())

steps_per_epoch = train_dataset.get_dataset_size()
steps_per_epoch1 = train_dataset1.get_dataset_size()

config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
config1 = CheckpointConfig(save_checkpoint_steps=steps_per_epoch1)

ckpt_callback = ModelCheckpoint(prefix="test", directory="./checkpoint", config=config)
loss_callback = LossMonitor(steps_per_epoch)

ckpt_callback1 = ModelCheckpoint(prefix="test", directory="./checkpoint1", config=config1)
loss_callback1 = LossMonitor(steps_per_epoch1)

model = Model(network=net, loss_fn=nn.MAELoss(), optimizer=opt, metrics={"mae"})
model1 = Model(network=net1, loss_fn=nn.MAELoss(), optimizer=opt1, metrics={"mae"})


# start_time = time.time()
model.fit(100, train_dataset=train_dataset, callbacks=[ckpt_callback, loss_callback])
model1.fit(100, train_dataset=train_dataset1, callbacks=[ckpt_callback1, loss_callback1])
# end_time = time.time()



# 模型评估
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# make a prediction
X_test = X_test.reshape((X_test.shape[0], timesteps, features))
# start_time1 = time.time()
yhat = model.predict(ms.Tensor(X_test, ms.float32))  # 预测值也是[0,1]之间的，因此要转换回原来的值
yhat1 = model1.predict(ms.Tensor(X_test, ms.float32))  # 预测值也是[0,1]之间的，因此要转换回原来的值
# end_time1 = time.time()
X_test = X_test.reshape((X_test.shape[0], timesteps * features))

# invert scaling for forecast
inv_yhat = np.concatenate((X_test[:, 0:features - 1], yhat), axis=1)  # 按列的方式进行组合
inv_yhat1 = np.concatenate((X_test[:, 0:features - 1], yhat1), axis=1)  # 按列的方式进行组合

min_max_scaler = MinMaxScaler(feature_range=(0, 1)).fit(inv_yhat)
min_max_scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(inv_yhat1)

inv_yhat = min_max_scaler.inverse_transform(inv_yhat)  # 从0~1反变换为真实数据
inv_yhat = inv_yhat[:, -1]  # 得到预测值

inv_yhat1 = min_max_scaler.inverse_transform(inv_yhat1)  # 从0~1反变换为真实数据
inv_yhat1 = inv_yhat1[:, -1]  # 得到预测值

# invert scaling for actual
y = y_test.reshape(y_test.shape[0], 1)
inv_y = np.concatenate((X_test[:, 0:features - 1], y), axis=1)
inv_y = min_max_scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]




# 画出真实数据和预测数据
plt.figure()
plt.plot(inv_yhat[2:], label='LSTM-prediction')
plt.plot(inv_y[2:], label='true')
#plt.ylim(0.95, 1.0)
#plt.legend()

#plt.figure()
plt.plot(inv_yhat1[2:], label='LSTM+attention-prediction')
#plt.plot(inv_y[2:], label='true1')
plt.ylim(0.95, 1.0)
plt.legend()
plt.show()

# calculate RMSE、MAE
# h = data. label.values[9000:-1]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))  # 均方差
mae = mean_absolute_error(inv_y, inv_yhat)  # 均绝对差
mape = 100 * np.mean(np.abs((inv_y - inv_yhat) / inv_y))  # 平均绝对百分误差： 差异值=发货量减去备货量的绝对值差异率=差异值/备货量准确率=1-差异率
print('Test RMSE: %.10f' % rmse)
print('Test MAE: %.10f' % mae)
print('Test MAPE: %.10f' % mape, '%')

# duration = end_time-start_time
# duration1 = end_time1-start_time1
# print('模型训练时间为：', duration, 'S')
# print('模型预测时间为：', duration1, 'S')


# 存储预测结果
final = pd.DataFrame(inv_yhat, columns=['one_step'])  # 151;5.5
final['nine_step'] = pd.DataFrame(inv_yhat)  # 153;5.6
final['Att'] = pd.DataFrame(inv_yhat)  # 138;5.0
final["label"] = pd.DataFrame(inv_y)
final.to_csv('./preResult.csv')
final = pd.read_csv('preResult.csv')