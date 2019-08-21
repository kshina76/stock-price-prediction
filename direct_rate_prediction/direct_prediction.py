import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, SGD

def preprocess(df, scaler):
	#終値を1日分移動させる(時系列データの学習ではやらないといけない)
	df_shift = df.copy()
	df_shift.close = df_shift.close.shift(-1)

	# 最後の行を除外
	df_shift = df_shift[:-1]

	# 念のためデータをdf_2として新しいデータフレームへコピ−
	df_2 = df_shift.copy()

	# time（時間）とcompを削除
	del df_2['time']
	del df_2['comp']

	# データセットの行数と列数を格納
	n = df_2.shape[0]
	p = df_2.shape[1]
 
	# 訓練データとテストデータへ切り分け
	train_start = 0
	train_end = int(np.floor(0.8*n))
	test_start = train_end + 1
	test_end = n
	data_train = df_2.loc[np.arange(train_start, train_end), :]
	data_test = df_2.loc[np.arange(test_start, test_end), :]

	# データの正規化
	scaler.fit(data_train)
	data_train_norm = scaler.transform(data_train)
	data_test_norm = scaler.transform(data_test)

	# 「close以外を特徴量とする」と「closeをターゲットとする」と分けるためにこのような処理をする
	X_train = data_train_norm[:,:-1]
	y_train = data_train_norm[:,-1]
	X_test = data_test_norm[:,:-1]
	y_test = data_test_norm[:,-1]

	# (debug)concatenateでaxis=1を指定してるから1次元ベクトルから1次元行列に変換しなければいけない。
	y_test = y_test.reshape(len(y_test),1)
	test_inv = np.concatenate((X_test, y_test), axis=1)
	test_inv = scaler.inverse_transform(test_inv)

	return X_train, y_train, X_test, y_test

def DNN_model():
	inputs = Input(shape=(4,))
	x = Dense(256,activation='relu')(inputs)
	x = Dense(128,activation='relu')(x)
	x = Dense(64,activation='relu')(x)
	prediction = Dense(1,activation='linear')(x)
	model = Model(input=inputs, output=prediction)

	optimizer = Adam()
	model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
	model.summary()

	return model

# csvからデータ取得
feature = ['open','high','low','volume']
df = pd.read_csv('./USD_JPY.D.csv')
print(df.tail(5))

# 前処理(X_train:正規化トレーニングデータ, y_train:正規化ターゲット, X_test:正規化テストデータ, y_test:正規化ターゲット)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train, y_train, X_test, y_test= preprocess(df,scaler)

# 重要な特徴量を見てみる
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
importance = rf.feature_importances_

print(importance.shape)

for i in range(len(feature)):
	print('{} : {}'.format(feature[i],importance[i]))

#学習

model = DNN_model()
history = model.fit(X_train, y_train, batch_size=600, epochs=50, validation_split=0.2)

#予測
predicted = model.predict(X_test)

#予測結果の正規化を直す
predicted_inv = np.concatenate((X_test,predicted),axis=1)
predicted_inv = scaler.inverse_transform(predicted_inv)

#元データと出力データの比較
y_test = y_test.reshape(len(y_test),1)
a = np.concatenate((X_test,y_test),axis=1)
a = scaler.inverse_transform(a)
print(a)
print(a.shape)
print(predicted_inv)
print(predicted_inv.shape)

#結果表示
fig = plt.figure()
plt.plot(a[700:868,4])
plt.plot(predicted_inv[700:868,4])
plt.show()