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
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low

data_path = './daily_data/USD_JPY.D.csv'
ema_13_period = 13
ema_21_period = 21 #emaの期間
bb_period = 20
interval = 20 # 分割数
predict_period = 5 #何日後のレートを予測するか
prev_period = 10  #何日前分のレートを予測に使うか
all_predict = prev_period + predict_period
ema_upper = 2  #emaより2円上を分割位置のmax
ema_lower = -2  #emaより2円下を分割位置のmin
input_dim = 6 # lstmに入力する特徴量の次元(wavenetでいう256。つまりチャネル)
use_tech = True

def create_dataset():
	rate_df = pd.read_csv(data_path)
	del rate_df['time']
	del rate_df['comp']

	# テクニカル指標を使う場合
	if use_tech == True:
		rate_df = create_tech(rate_df)
		close_s = rate_df['close']
		upper = create_sequence(rate_df['upper'].values)[:, :prev_period]
		middle = create_sequence(rate_df['middle'].values)[:, :prev_period]
		lower = create_sequence(rate_df['lower'].values)[:, :prev_period]
		ema_13 = create_sequence(rate_df['ema_13'].values)[:, :prev_period]
		ema_21 = create_sequence(rate_df['ema_21'].values)[:, :prev_period]

	# テクニカル指標を使使わない場合
	else:
		close_s = rate_df['close']

	# prev_periodで指定された期間分データをとってきて時系列データを作成する。
	closes = close_s.values
	#feature = [] # 特徴量用のリスト
	labels_pre = closes[all_predict:] # 教師ラベルを格納するための準備の変数
	labels = [] #ラベル格納用
	feature = create_sequence(closes)
	print(feature.shape)

	labels_pre = feature[:, all_predict-1]
	feature = feature[:, :prev_period]

	print(feature.shape)
	print(labels_pre.shape)

	# ラベルを作成
	for i in range(len(labels_pre)):
		if feature[i, prev_period-1] < labels_pre[i]:
			labels.append(1)
		elif feature[i, prev_period-1] > labels_pre[i]:
			labels.append(2)
		else:
			labels.append(0)
	labels = np.asarray(labels)

	# one hot encording
	a = np.array([i for i in range(3)])
	lb = LabelBinarizer()
	lb.fit(a)
	labels = lb.transform(labels)

	print(labels.shape)

	# lstmに合わせて次元を調整
	if use_tech == True:
		feature = np.expand_dims(feature, axis=2)
		upper = np.expand_dims(upper, axis=2)
		middle = np.expand_dims(middle, axis=2)
		lower = np.expand_dims(lower, axis=2)
		ema_13 = np.expand_dims(ema_13, axis=2)
		ema_21 = np.expand_dims(ema_21, axis=2)
		print(feature.shape)
		print(upper.shape)
		feature = np.concatenate([feature, upper, middle, lower, ema_13, ema_21], axis=2)
		print(feature.shape)
	else:
		feature = np.expand_dims(feature, axis=2)

	X_train, X_test, y_train, y_test = train_test_split(feature, labels, train_size=0.8, shuffle=False)

	return X_train, X_test, y_train, y_test

# テクニカル指標を作り出す関数
def create_tech(rate_df):
	temp = list(rate_df['close'].values)
	rate_df['upper'] = bb_up(temp, bb_period)
	rate_df['middle'] = bb_mid(temp, bb_period)
	rate_df['lower'] = bb_low(temp, bb_period)
	rate_df['ema_13'] = ema(temp, ema_13_period)
	rate_df['ema_21'] = ema(temp, ema_21_period)

	# NaNが1つでもある行を削除する(ボリバン作成過程でNaNができてしまうから)
	rate_df = rate_df.dropna(how='any')
	rate_df = rate_df.reset_index(drop=True)

	return rate_df

# prev_periodで指定された期間分データをとってきて時系列データを作成する。
# closeでもボリバンでもなんでも指定された時系列に変換してくれる
# 引数は、numpyで。
def create_sequence(temp):
	feature = [] # 特徴量用のリスト
	for i in range(len(temp)-all_predict+1):
		feature.append(list(temp[i:i+all_predict]))
	feature = np.asarray(feature)
	print(feature.shape)

	return feature


def build_model(input_size, ouput_size):
	inputs = Input(shape=(input_size, input_dim))
	x = LSTM(20)(inputs)
	x = Dropout(0.1)(x)
	x = Dense(ouput_size, activation='softmax')(x)
	model = Model(input=inputs, output=x)
	optimizer = Adam()
	model.compile(loss="categorical_crossentropy",
					optimizer=optimizer, metrics=['accuracy'])
	model.summary()

	return model

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = create_dataset()
	#print(X_train.shape)
	#print(y_test.shape)
	
	model = build_model(X_train.shape[1], y_train.shape[1])
	history = model.fit(X_train, y_train, 
						batch_size=32, 
						epochs=50, 
						validation_split=0.2,
						validation_data=(X_test,y_test))
	score = model.evaluate(X_test,y_test,verbose=0)
	print('Test loss',score[0])
	print('Test accuracy',score[1])