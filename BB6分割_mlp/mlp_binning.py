import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.layers.recurrent import LSTM
from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low

data_path = './daily_data/USD_JPY.D.csv'
bb_period = 20 #ボリンジャーバンドの期間
interval = 4 # 分割数

def create_dataset():
	# csv読み込みといらない特徴量を削除
	rate_df = pd.read_csv(data_path)
	del rate_df['time']
	del rate_df['comp']
	#del rate_df['volume']
	print(rate_df.tail(5))

	# ボリンジャーバンド計算
	temp = list(rate_df['close'].values)
	upper = bb_up(temp, bb_period)
	middle = bb_mid(temp, bb_period)
	lower = bb_low(temp, bb_period)
	rate_df['bb_up'] = upper
	rate_df['bb_mid'] = middle
	rate_df['bb_low'] = lower
	print(rate_df.tail(5))

	# NaNが1つでもある行を削除する(ボリバン作成過程でNaNができてしまうから)
	rate_df = rate_df.dropna(how='any')
	rate_df = rate_df.reset_index(drop=True)

	# 0.01pips以下はノイズとして消す
	rate_df = rate_df.round(3)
	print(rate_df.tail(5))

	# 教師ラベルを作成
	# N日目の終値時のボリバンの区間内で、N+1日目の終値がどこの区間にラベル付けされるかを計算
	labels = []
	# leakageしないようにshiftして終値をずらし、NaNになったところを消すために[:-1]で全体をずらす。
	close_s = rate_df['close'].shift(-1)[:-1]
	bb_up_s = rate_df['bb_up'][:-1]
	bb_low_s = rate_df['bb_low'][:-1]
	for i in range(len(close_s)):
		label = binning_data(bb_up_s[i], bb_low_s[i], close_s[i])
		labels.append(label)

	# one hot encording
	a = np.array([i for i in range(interval+2)])
	lb = LabelBinarizer()
	lb.fit(a)
	labels = lb.transform(labels) # (4323, 7)

	# ラベルに合わせて長さを調節してから、dataframeをnumpyに変換
	rate_df = rate_df[:-1]
	rate_np = rate_df.values

	print(rate_np.shape)
	print(labels.shape)

	# trainとtestに分割
	X_train, X_test, y_train, y_test = train_test_split(rate_np, labels, train_size=0.8)

	return X_train, X_test, y_train, y_test

# 間隔
def binning_data(upper, lower, rate):
	temp = []
	if lower <= rate <= upper:
		# ボリバン内に収まっているとき
		temp.append(upper)
		temp.append(lower)
		temp.append(rate)
		temp = pd.Series(temp)
		s_cut, bins = pd.cut(temp, interval, retbins=True)

		# 要素数6個でlen=6, 0,1,2,3,4,5
		# rateの区間ラベルを返す(これが教師ラベルになる)
		count = None
		for i in range(len(bins)-1):
			count = 0
			if bins[i] < rate <= bins[i+1]:
				count = i
				break
		
		return count

	elif rate > upper:
		# ボリバンを超えている場合は、最大分割位置+1(つまり、interval)を返す
		return interval
	else:
		# ボリバンを下回っている場合は、最大分割位置+2(つまり、interval+1)を返す
		return interval + 1

def build_model(inputs_size, outputs_size):
	inputs = Input(shape=(inputs_size,))
	x = Dense(1024, activation='relu')(inputs)
	x = Dense(512, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	#x = Dense(128, activation='relu')(x)
	x = Dense(outputs_size, activation='softmax')(x)
	model = Model(input=inputs, output=x)
	optimizer = Adam()
	model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	return model


def train_test():
	# 学習データとテストデータ読み込み
	X_train, X_test, y_train, y_test = create_dataset()

	print(X_train.shape)
	print(y_train.shape)

	# モデル構築
	model = build_model(X_train.shape[1], y_train.shape[1])

	history = model.fit(X_train, y_train, 
						batch_size=32, 
						epochs=50, 
						validation_split=0.2,
						validation_data=(X_test,y_test))
	score = model.evaluate(X_test,y_test,verbose=0)
	print('Test loss',score[0])
	print('Test accuracy',score[1])

if __name__ == '__main__':
	create_dataset()
	train_test()