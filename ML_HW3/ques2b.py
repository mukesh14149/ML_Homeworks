from matplotlib import pyplot as plt
import os
import os.path
import argparse
import h5py
import pickle
import numpy as np
import struct
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str )
args = parser.parse_args()


############################This part taken from internet############################
def loadimg(imgfilename):
    with open(imgfilename, 'rb') as imgfile:
        datastr = imgfile.read()
    
    index = 0
    mgc_num, img_num, row_num, col_num = struct.unpack_from('>IIII', datastr, index)
    index += struct.calcsize('>IIII')
    
    image_array = np.zeros((img_num, row_num, col_num))
    for img_idx in xrange(img_num):
        img = struct.unpack_from('>784B', datastr, index)
        index += struct.calcsize('>784B')
        image_array[img_idx,:,:] = np.reshape(np.array(img), (28,28))
    image_array = image_array/255.0
    np.save(imgfilename[:6]+'image-py', image_array)
    return None


def loadlabel(labelfilename):
    with open(labelfilename, 'rb') as labelfile:
        datastr = labelfile.read()
    
    index = 0
    mgc_num, label_num = struct.unpack_from('>II', datastr, index)
    index += struct.calcsize('>II')
    
    label = struct.unpack_from('{}B'.format(label_num), datastr, index)
    index += struct.calcsize('{}B'.format(label_num))
    
    label_array = np.array(label)
    
    np.save(labelfilename[:5]+'label-py', label_array)
    return None


loadimg('train-images.idx3-ubyte')
loadimg('t10k-images.idx3-ubyte')
loadlabel('train-labels.idx1-ubyte')
loadlabel('t10k-labels.idx1-ubyte')

train_image = np.load('train-image-py.npy')
train_label = np.load('trainlabel-py.npy')
test_image = np.load('t10k-iimage-py.npy')
test_label = np.load('t10k-label-py.npy')
########################Uptothistakenfrominternet##########################################33



X_new = []
for i in train_image:
	X_new.append(i.reshape(784))

X_train = np.array(X_new)

Y_new = []
for i in train_label:
	temp = [0] * 10
	temp[i] = 1
	Y_new.append(temp)

y_train = np.array(Y_new)


X_new = []
for i in test_image:
	X_new.append(i.reshape(784))

X_test = np.array(X_new)

Y_new = []
for i in test_label:
	temp = [0] * 10
	temp[i] = 1
	Y_new.append(temp)

y_test = np.array(Y_new)

print X_train.shape,y_train.shape, X_test.shape, y_test.shape

output = []
max_iter = [100,150]
for i in max_iter:
	parameters = {'learning_rate_init':[0.0001,0.001,0.01]}
	clf = MLPClassifier(max_iter=i, hidden_layer_sizes=(100,50), activation='logistic')
	clf = GridSearchCV(clf, parameters)
	clf = clf.fit(X_train,y_train)
	print clf
	ttt= clf.predict_proba(X_test)
	count = 0
	for s,i in enumerate(ttt):
		t = np.argmax(i)
		if(y_test[s][t]==1):		
			count=count+1
	temp = count/float(len(X_test))
	output.append(temp)
	print temp


fig = plt.figure()
plt.plot(max_iter,output)
plt.ylabel('Accuracy in %')
plt.xlabel('no. of epoch')
plt.savefig('ques2b.png')

