from GaussianNB import GaussianNB
from DecisionTreeClassifier import DecisionTreeClassifier
from LogisticRegression import LogisticRegression
import h5py


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

# Preprocess data and split it

# Train the models
X,Y = load_h5py('../Data/part_A_train.h5')
Y_R = []

for i in range(0,len(Y)):
	Y_R.append(Y[i].tolist().index(1.0))

data_size = int(0.7*len(X)) 
clf = LogisticRegression()

for j in range(0,10):
	temp = []
	for i in range(0,len(Y_R)):
		if(Y_R[i]==j):
			temp.append(0)
		else:
			temp.append(1)


	clf.fit(X[0:data_size], temp[0:data_size])
	predictarr = clf.predict(X[data_size:len(X)])
	count = 0;
	t = 0;
	for i in range(data_size,len(X)):
		if(temp[i] == predictarr[t]):
			count=count+1
		t = t+1

	print (float(count)/(0.3*len(X)))*100
