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

data_path = './daily_data/USD_JPY.D.csv'
ema_period = 21 #emaの期間
interval = 20 # 分割数
predict_period = 5 #何日後のレートを予測するか
prev_period = 10  #何日前分のレートを予測に使うか
all_predict = prev_period + predict_period
ema_upper = 2  #emaより2円上を分割位置のmax
ema_lower = -2  #emaより2円下を分割位置のmin
input_dim = 1 # lstmに入力する特徴量の次元(wavenetでいう256。つまりチャネル)

'''
方針
step1. 時系列データを作る (predict_period + prev_period 分をとってくる)
step2. 特徴量(prev_period 分をとれば学習用時系列データになる)　と　ラベル用(predict_period + prev_periodの末尾が計測値の5日後を表すので、そこをとってくる)
※とってくるときのfor文のrangeは、step1のpredict_period + prev_periodを指定すれば、out of indexにならないと思う。

binningには、emaのupperとemaのlowerと予測する部分を渡せばおっけー

'''

def create_dataset():
	rate_df = pd.read_csv(data_path)
	del rate_df['time']
	del rate_df['comp']
	close_s = rate_df['close']

	# prev_periodで指定された期間分データをとってきて時系列データを作成する。
	closes = close_s.values
	feature = [] # 特徴量用のリスト
	labels_pre = closes[all_predict:] # 教師ラベルを格納するための準備の変数
	labels = [] #ラベル格納用
	for i in range(len(closes)-all_predict+1):
		feature.append(list(closes[i:i+all_predict]))
	feature = np.asarray(feature)

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
	feature = np.expand_dims(feature, axis=2)

	X_train, X_test, y_train, y_test = train_test_split(feature, labels, train_size=0.8, shuffle=False)

	return X_train, X_test, y_train, y_test


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
	print(X_train.shape)
	print(y_test.shape)

	model = build_model(X_train.shape[1], y_train.shape[1])
	history = model.fit(X_train, y_train, 
						batch_size=32, 
						epochs=50, 
						validation_split=0.2,
						validation_data=(X_test,y_test))
	score = model.evaluate(X_test,y_test,verbose=0)
	print('Test loss',score[0])
	print('Test accuracy',score[1])