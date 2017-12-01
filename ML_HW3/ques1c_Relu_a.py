from matplotlib import pyplot as plt
import os
import os.path
import argparse
import h5py
import pickle
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str )
args = parser.parse_args()

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


X,Y = load_h5py(args.data)
X_new = []
for i in X:
	X_new.append(i.reshape(784))

X = np.array(X_new)

Y_new = []
for i in Y:
	if i==9:
		Y_new.append([1,0])
	else:
		Y_new.append([0,1])

y = np.array(Y_new)
print y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print y_test.shape

#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
	return (x > 0)

#Variable initialization
epoch=100 #Setting training iterations
lr=0.001 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
shpo = [784,100,50,2]




#weight and bias initialization
weights = [np.array([0])] + [np.random.normal(0, 0.02,[x,y]) for y, x in zip(shpo[1:], shpo[:-1])]
#print len(weights),weights[0].shape,weights[1].shape,weights[2].shape,weights[3].shape

biases = [np.random.normal(0, 0.02,[1,y]) for y in shpo[0:]]
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
			hiddenlayer_activations[j] = ReLU(hidden_layer_input[j])
			print j, hidden_layer_input[j].shape,hiddenlayer_activations[j].shape

			
		#print hiddenlayer_activations[-1].shape, derivatives_sigmoid(hidden_layer_input[-1]).shape
		
		#Backpropagation
		E = (y_train[rg]-sigmoid(hidden_layer_input[-1])) * derivatives_sigmoid(hidden_layer_input[-1])
		#print E.shape

		b3 = E
		w3 = (hiddenlayer_activations[-2].T).dot(E)
		E =E.T
		ww = []
		bb = []
		for gh in [2,1]:
			A=dReLU(hidden_layer_input[gh]).T			
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
		hiddenlayer_activations[j] = ReLU(hidden_layer_input[j])
#		print j, hidden_layer_input[j].shape

#	print hiddenlayer_activations[-1]
	count = 0
	for s,i in enumerate(sigmoid(hidden_layer_input[-1])):
		
		t=0
		if i[0] > i[1]:
			t=0
		else:
			t=1

		if(y_test[s][t]==1):		
			count=count+1
	temp = count/float(len(X_test))
	output.append(temp)
	if temp>max_acc:
		max_acc = temp
		final_weight = [weights[1],weights[2],weights[3]] 
		final_bais = [biases[1],biases[2],biases[3]]
	print max_acc

f = open('ttcRelua_weight.pkl','wb')
pickle.dump(final_weight,f)

f1 = open('ttcRelua_bais.pkl','wb')
pickle.dump(final_bais,f1)

print max_acc
plt.plot(np.arange(100),output)
plt.ylabel('Accuracy in %')
plt.xlabel('no. of epoch')
plt.savefig('ttcRelua.png')
