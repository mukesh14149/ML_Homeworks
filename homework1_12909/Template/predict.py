import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
print args.test_data
print args.weights_path
X,Y = load_h5py(args.test_data)
Y_R = []
for i in range(0,len(Y)):
	Y_R.append(Y[i].tolist().index(1.0))

if args.model_name == 'GaussianNB':
	clf = joblib.load(args.weights_path+"/GaussianNB.pkl") 
	predictarr = clf.predict(X[0:len(X)])
	count = 0;
	t = 0;
	for i in range(0,len(X)):		
		if(Y_R[i] == predictarr[t]):
			count=count+1
		t = t+1
	print "Accuracy of LogisticRegression::"
	print (count/(len(X)))*100
	pass
elif args.model_name == 'LogisticRegression':
	clf = joblib.load(args.weights_path+"/LogisticRegression.pkl") 
	predictarr = clf.predict(X[0:len(X)])
	count = 0;
	t = 0;
	for i in range(0,len(X)):		
		if(Y_R[i] == predictarr[t]):
			count=count+1
		t = t+1
	print "Accuracy of LogisticRegression::"
	print (count/(len(X)))*100
	pass
elif args.model_name == 'DecisionTreeClassifier':
	clf = joblib.load(args.weights_path+"/DecisionTreeClassifier.pkl") 
	predictarr = clf.predict(X[0:len(X)])
	count = 0;
	t = 0;
	for i in range(0,len(X)):		
		if(Y_R[i] == predictarr[t]):
			count=count+1
		t = t+1
	print "Accuracy of LogisticRegression::"
	print (count/(len(X)))*100
	
	# load the model

	# model = DecisionTreeClassifier(  ...  )

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
else:
	raise Exception("Invald Model name")
