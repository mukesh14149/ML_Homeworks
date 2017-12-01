from matplotlib import pyplot as plt
import os
import os.path
import argparse
import h5py
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle =False)
print y_test.shape

output=[]
max_iter = [50]
for i in max_iter:
	parameters = {'learning_rate_init':[0.01]}
	clf = MLPClassifier(hidden_layer_sizes=(200,100,50), activation='relu',max_iter = i)
	clf = GridSearchCV(clf, parameters)
	clf = clf.fit(X,y)
	print clf
	#ttt= clf.predict_proba(X_test)
	ttt= clf.predict(X_test)
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
plt.savefig('ques2a.png')	

