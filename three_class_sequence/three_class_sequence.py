import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import mnist
import keras.backend as K
import tensorflow as tf

np.random.seed(seed=0)

# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(0)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

frame_size = 40  #何点を見るかのサイズ
use_tech = False #テクニカル指標を使うかどうか
data_path = './USD_JPY.D.csv'

if use_tech == True:
	ema_num = 3  #ema特徴量の数
	feature_num = frame_size + ema_num
else:
	feature_num = frame_size


def create_dataset():
	rate_df = pd.read_csv(data_path)
	del rate_df['time']
	del rate_df['comp']
	close_s = rate_df['close']

	# 40点のcloseと40点目のemaたちを合わせたもの。
	# または、40点のcloseのみ
	# ちなみに、一番最後の行は入れていない。なぜなら一番最後の行を入れてしまうと、正解ラベルが入ってしまうから。
	closes = close_s.values
	feature = [] # 特徴量用のリスト
	labels_pre = closes[frame_size:] # 教師ラベルを格納するための準備の変数
	labels = [] #ラベル格納用
	if use_tech == True:
		feature = []
		ema13 = close_s.ewm(span=13).mean().values
		ema21 = close_s.ewm(span=21).mean().values
		ema34 = close_s.ewm(span=34).mean().values

		for i in range(len(closes)-frame_size):
			temp = list(closes[i:i+frame_size])
			temp.append(ema13[i+frame_size-1])
			temp.append(ema21[i+frame_size-1])
			temp.append(ema34[i+frame_size-1])
			feature.append(np.asarray(temp))
		feature = np.asarray(feature)
	else:
		for i in range(len(closes)-frame_size):
			feature.append(list(closes[i:i+frame_size]))
		feature = np.asarray(feature)

	# ラベルを作成
	for i in range(len(labels_pre)):
		if feature[i, frame_size-1] < labels_pre[i]:
			labels.append(1)
		elif feature[i, frame_size-1] > labels_pre[i]:
			labels.append(2)
		else:
			labels.append(0)
	labels = np.asarray(labels)

	X_train, X_test, y_train, y_test = train_test_split(feature, labels, train_size=0.8, shuffle=False)

	return X_train, X_test, y_train, y_test


def dnn_model():
	inputs = Input(shape=(feature_num,))
	x = Dense(256,activation='relu')(inputs)
	x = Dense(128,activation='relu')(x)
	x = Dense(64,activation='relu')(x)
	prediction = Dense(3,activation='softmax')(x)
	model = Model(input=inputs,output=prediction)
	optimizer = Adam()
	model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
	model.summary()

	return model

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = create_dataset()

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)

	model = dnn_model()
	history = model.fit(X_train, y_train, 
						batch_size=50, 
						epochs=50, 
						validation_split=0.2,
						validation_data=(X_test,y_test))
	score = model.evaluate(X_test,y_test,verbose=0)
	print('Test loss',score[0])
	print('Test accuracy',score[1])