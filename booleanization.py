# python==3.12.6

import os, sys, random
import numpy as np
import preprocessing

from sklearn import datasets
from keras.datasets import mnist, fashion_mnist, cifar10

# dataset name as the argument
if sys.argv[1].lower() == 'all':
	all_datasets = ['digits','mnist','kws', 'kmnist', 'fmnist', 'cifar2', 'cifar10']
else:
	all_datasets = [sys.argv[1].lower()]

# make directory to store the booleanized dataset(s)
if not os.path.exists(r"bool_datasets"):
	os.makedirs(r"bool_datasets")

for dataset in all_datasets:
	match dataset:
		# ------------- MNIST --------------------------------------------------
		case "mnist":
			(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# ------------- FMNIST ------------------------------------------------
		case "fmnist":
			(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# -------------- KMNIST ------------------------------------------------
		case "kmnist":
			X_train = np.load(r"raw_datasets/kmnist/kmnist-train-imgs.npz")['arr_0']
			X_test = np.load(r"raw_datasets/kmnist/kmnist-test-imgs.npz")['arr_0']
			Y_train = np.load(r"raw_datasets/kmnist/kmnist-train-labels.npz")['arr_0']
			Y_test = np.load(r"raw_datasets/kmnist/kmnist-test-labels.npz")['arr_0']
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)
					
		# ------------- Digits ----------------------------------------------------
		case "digits":
			digits = datasets.load_digits()
			X = np.where(digits.data > 7.5, 1, 0)
			Y = digits.target
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)		

		# --------------- Keyword spotting -----------------------------------------------------------
		case "kws":
			[train_x, Y_train, test_x, Y_test] = preprocessing.kws_dataset(r"raw_datasets/mini_speech_commands")
			X_raw_features = np.concatenate((train_x, test_x), axis=0)
			X, _ = preprocessing.thermo_encoding(X_raw_features, 3)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)

		# --------------- CIFAR2 --------------------------------------------------------------------
		case "cifar2":
			(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
			Y_train=Y_train.reshape(Y_train.shape[0])
			Y_test=Y_test.reshape(Y_test.shape[0])
			animals = np.array([2, 3, 4, 5, 6, 7])
			Y_train = np.where(np.isin(Y_train, animals), 1, 0)
			Y_test = np.where(np.isin(Y_test, animals), 1, 0)
			X_raw_features = np.concatenate((X_train_org, X_test_org), axis=0)
			X = preprocessing.CIFAR_HOG(X_raw_features)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)

		# --------------- CIFAR10 --------------------------------------------------------------------
		case "cifar10":
			(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
			Y_train=Y_train.reshape(Y_train.shape[0])
			Y_test=Y_test.reshape(Y_test.shape[0])
			X_raw_features = np.concatenate((X_train_org, X_test_org), axis=0)
			X = preprocessing.CIFAR_HOG(X_raw_features)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)							
														
		case _:
			print("The given dataset %s is not recognized." %sys.argv[1])

	# store datasets in given directory
	if not os.path.exists(r"bool_datasets/"+dataset):
		os.makedirs(r"bool_datasets/"+dataset)
	np.save(r"bool_datasets/"+dataset+'/X_train.npy', X_train)
	np.savetxt(r"bool_datasets/"+dataset+'/X_train.txt', X_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/Y_train.npy', Y_train)
	np.savetxt(r"bool_datasets/"+dataset+'/Y_train.txt', Y_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/X_test.npy', X_test)
	np.savetxt(r"bool_datasets/"+dataset+'/X_test.txt', X_test, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/Y_test.npy', Y_test)
	np.savetxt(r"bool_datasets/"+dataset+'/Y_test.txt', Y_test, fmt='%d')