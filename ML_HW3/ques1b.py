from matplotlib import pyplot as plt
import os
import os.path
import argparse
import h5py
import pickle
import numpy as np
import struct
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str )
args = parser.parse_args()



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

# print(train_image.shape)
# print(train_label.shape)
# print(test_image.shape)
# print(test_label.shape)
print train_label[1:100]

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



#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def derivatives_softmax(x):
	return softmax(x) * (1 - softmax(x))

#Variable initialization
epoch=150 #Setting training iterations
lr=0.001 #Setting learning rate
shpo = [784,100,50,10]


#weight and bias initialization
weights = [np.array([0])] + [np.random.randn(x, y) for y, x in zip(shpo[1:], shpo[:-1])]
#print len(weights),weights[0].shape,weights[1].shape,weights[2].shape,weights[3].shape

biases = [np.random.randn(1, y) for y in shpo[0:]]
#print len(biases),biases[0].shape,biases[1].shape,biases[2].shape,biases[3].shape

hidden_layer_input = [np.zeros(bias.shape) for bias in biases]
hiddenlayer_activations = [np.zeros(bias.shape) for bias in biases]
#print "raa",hiddenlayer_activations[0].shape


output = []
max_acc = 0
final_weight=[]
for i in range(epoch):
	print i
	#Forward Propogation
	for rg, rrr in enumerate(X_train):
		hiddenlayer_activations[0] = rrr.reshape(1,len(rrr))
		for j in range(1,len(shpo)):
			hidden_layer_input[j]=np.dot(hiddenlayer_activations[j-1],weights[j]) + biases[j]
			hiddenlayer_activations[j] = sigmoid(hidden_layer_input[j])
			#print j, hiddenlayer_activations[j].shape
			
		#print hiddenlayer_activations[-1].shape, derivatives_sigmoid(hidden_layer_input[-1]).shape
		
		#Backpropagation
		E = (y_train[rg]-softmax(hidden_layer_input[-1])) * derivatives_softmax(hidden_layer_input[-1])
		#print E.shape

		b3 = E
		w3 = (hiddenlayer_activations[-2].T).dot(E)
		E =E.T
		ww = []
		bb = []
		for gh in [2,1]:
			A=derivatives_sigmoid(hidden_layer_input[gh]).T			
			B= (weights[gh+1]).dot(E)
			#print B.shape,A.shape
			E = np.multiply(B,A)
			#print (hiddenlayer_activations[1].T).shape, Error_at_hidden_layer2.shape 
			ww.append(hiddenlayer_activations[gh-1].T.dot(E.T)) 
			bb.append(E.T)  	
	


		weights[3] += w3 *lr
		biases[3] += b3 *lr
		
		weights[2] += ww[0] *lr
		biases[2] += bb[0] *lr

		weights[1] += ww[1] *lr
		biases[1] += bb[1] *lr






#	print hiddenlayer_activations[-1]


	hiddenlayer_activations[0] = X_test
	for j in range(1,len(shpo)):

		hidden_layer_input[j]=np.dot(hiddenlayer_activations[j-1],weights[j]) + biases[j]
		hiddenlayer_activations[j] = sigmoid(hidden_layer_input[j])
#		print j, hidden_layer_input[j].shape

#	print hiddenlayer_activations[-1]
	count = 0
	for s,i in enumerate(softmax(hidden_layer_input[-1])):
		t = np.argmax(i)
		if(y_test[s][t]==1):		
			count=count+1
	temp = count/float(len(X_test))
	output.append(temp)
	if temp>max_acc:
		max_acc = temp
		final_weight = [weights[1],weights[2],weights[3]] 
		final_bais = [biases[1],biases[2],biases[3]]
	print max_acc
f = open('largettb_weight.pkl','wb')
pickle.dump(final_weight,f)

f1 = open('largettb_bais.pkl','wb')
pickle.dump(final_bais,f1)

print max_acc




fig = plt.figure()
plt.plot(np.arange(150),output)
plt.ylabel('Accuracy in %')
plt.xlabel('no. of epoch')
plt.savefig('largettb.png')