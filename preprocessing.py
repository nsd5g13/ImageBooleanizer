import numpy as np # Version: 2.4.2
import pathlib
import os, random
import tensorflow as tf # Version: 2.17.0
import pathlib, librosa
import cv2

# -- Encode raw features into one-hot codes -------------------------------------------------
def onehot_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	onehot = []

	for i in range(no_bins):
		onehot.append([])
		for j in range(no_bins):
			if i != j:
				onehot[i].append(0)
			else:
				onehot[i].append(1)
	onehot=onehot[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(onehot[X_index])
	
	return X_bool, thres

# -- Encode raw features into thermometer codes -------------------------------------------------
def thermo_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	thermo = []

	for i in range(no_bins):
        	thermo.append([])
        	for j in range(no_bins-1):
                	if i > j:
                        	thermo[i].append(0)
                	else:
                        	thermo[i].append(1)
	thermo=thermo[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(thermo[X_index])
	
	return X_bool, thres
							
# -- Split dataset into training/test set
def DatasetSplit(X, Y, TrainingSetRatio):
	random.seed(1)
	n = random.sample(range(len(Y)), len(Y))
			
	X_train = []
	Y_train = []
	for each in n[0:int(len(Y)*TrainingSetRatio)]:
		X_train.append(X[each])
		Y_train.append(Y[each])

	X_test = []
	Y_test = []
	for each in n[int(len(Y)*TrainingSetRatio):]:
		X_test.append(X[each])
		Y_test.append(Y[each])

	X_train=np.array(X_train)
	Y_train=np.array(Y_train)
	X_test=np.array(X_test)
	Y_test=np.array(Y_test)

	return X_train, Y_train, X_test, Y_test

# -- preprocessing, specifically for KWS ---------------------------------------------------------
# refer: https://www.tensorflow.org/tutorials/audio/simple_audio
# ------------------------------------------------------------------------------------------------
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
	# Convert the waveform to a spectrogram via a STFT.
	stfts = tf.signal.stft(waveform, frame_length=2560, frame_step=480)
	# Obtain the magnitude of the STFT.
	spectrograms = tf.abs(stfts)
	# Add a `channels` dimension, so that the spectrogram can be used
	# as image-like input data with convolution layers (which expect
	# shape (`batch_size`, `height`, `width`, `channels`).
	spectrograms = spectrograms[..., tf.newaxis]
	return spectrograms

def make_spec_ds(ds):
  return ds.map(map_func=lambda audio,label: (get_spectrogram(audio), label),num_parallel_calls=tf.data.AUTOTUNE)

def kws_dataset(dataset_path):
	data_dir = pathlib.Path(dataset_path)
	if not data_dir.exists():
		# properly change the origin
		tf.keras.utils.get_file(origin='file:///mnt/d/design/Tsetlin/Image Dataset Booleanizer/raw_datasets/mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir='data')
	train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(directory=data_dir, batch_size=1, validation_split=0.2, seed=0, output_sequence_length=16000, subset='both', sampling_rate=16000)
	label_names = np.array(train_ds.class_names)
	print("KWS label names:", label_names)

	train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
	val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

	train_spectrogram_ds = make_spec_ds(train_ds)
	train_spectrogram_ds = train_spectrogram_ds.unbatch()
	all_train_x = []
	all_train_y = []
	for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(6400):
		train_x = example_spectrograms.numpy()
		mels = librosa.feature.melspectrogram(S=train_x, sr=8000)
		log_mels = librosa.core.amplitude_to_db(mels, ref=np.max)
		mfcc = librosa.feature.mfcc(S=log_mels, sr=8000, n_mfcc=13)	
		train_x = mfcc.flatten()
		all_train_x.append(train_x.tolist())
		train_y = example_spect_labels.numpy()
		train_y = train_y.flatten()
		all_train_y.extend(train_y.tolist())
	train_x = np.array(all_train_x)
	train_y = np.array(all_train_y)

	val_spectrogram_ds = make_spec_ds(val_ds)
	val_spectrogram_ds = val_spectrogram_ds.unbatch()
	all_test_x = []
	all_test_y = []
	for example_spectrograms, example_spect_labels in val_spectrogram_ds.take(1600):
		test_x = example_spectrograms.numpy()
		mels = librosa.feature.melspectrogram(S=test_x, sr=8000)
		log_mels = librosa.core.amplitude_to_db(mels, ref=np.max)
		mfcc = librosa.feature.mfcc(S=log_mels, sr=8000, n_mfcc=13)
		test_x = mfcc.flatten()
		all_test_x.append(test_x.tolist())
		test_y = example_spect_labels.numpy()
		test_y = test_y.flatten()
		all_test_y.extend(test_y.tolist())
	test_x = np.array(all_test_x)
	test_y = np.array(all_test_y)

	return train_x, train_y, test_x, test_y

# -- preprocessing, specifically for CIFAR ---------------------------------------------------------
def CIFAR_HOG(X_raw_features):
	patch_size = 0

	imageSize = 32  #The size of the original image - in pixels - assuming this is a square image
	channels = 3    #The number of channels of the image. A RBG color image, has 3 channels

	winSize = imageSize
	blockSize = 24
	blockStride = 8
	cellSize = 8
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = True
	nlevels = 64
	signedGradient = True
	hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)

	fd = hog.compute(X_raw_features[0])
	fds = []
	X_bool = np.empty((X_raw_features.shape[0], fd.shape[0]), dtype=np.uint32)
	for i in range(X_raw_features.shape[0]):
		fd = hog.compute(X_raw_features[i])
		fds.append(fd)

	fds = np.array(fds)
	fds = np.transpose(fds)
	fd_quantiles = []
	for fd in fds:
		fd_quantiles.append(np.quantile(fd, 0.5))

	for i in range(X_raw_features.shape[0]):
		fd = hog.compute(X_raw_features[i])
		for j, each_fd in enumerate(fd):
			if each_fd >= fd_quantiles[j]:
				X_bool[i][j] = 1
			else:
				X_bool[i][j] = 0

	return X_bool