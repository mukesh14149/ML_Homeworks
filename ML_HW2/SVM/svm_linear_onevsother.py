import os
import os.path
import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()



# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X,Y1 = load_h5py(args.data)
Y=Y1
data_size = int(0.7*len(X));

def linear_(x,y):
	return np.dot(x, y)

def kernel_(X1, X2, K=linear_):
    grammat = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            grammat[i,j] = K(x1,x2)
    return grammat


def predict(x,theta,intercept):
	predict=[]
	classification = np.dot(x,theta)+intercept
	for i in classification:
		if i<0:		
			predict.append(0)
		else:
			predict.append(1)
	return predict
				
unique = np.unique(Y)
num = len(unique)
print unique
if num>2:
	final_res = []
	for i in range(0,num):
		#print i
		Y = []
		for j in Y1:
			if(i==j):
				Y.append(1)
			else:	
				Y.append(0)
		clf= SVC(kernel =kernel_)		
		clt= clf.fit(X[0:data_size],Y[0:data_size])

		support_vectors=[]
		for i in clf.support_:
			support_vectors.append(X[i])

		support_vectors = np.array(support_vectors)	
		coef = np.matmul(clf.dual_coef_,support_vectors)
		final_res.append(predict(X[data_size:len(X)],coef.T,clf.intercept_))
			
	pred = []
	for i in range(0,len(final_res[0])):
		if(final_res[0][i]==1):
			pred.append(0)
		if(final_res[1][i]==1):
			pred.append(1)
		if(final_res[2][i]==1):
			pred.append(2)
		
	k=0
	count=0
	for i in Y1[data_size:len(Y1)]:
		if  i==pred[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	print 100*(count/len(Y1[data_size:len(Y1)]))
		
		
	
else:
	clf= SVC(kernel = kernel_)		
	clt = clf.fit(X[0:data_size],Y1[0:data_size])
	support_vectors=[]
	support_vectors_labels=[]
	for i in clf.support_:
		support_vectors.append(X[i])
		support_vectors_labels.append(Y[i])

	support_vectors = np.array(support_vectors)	
	coef = np.matmul(clf.dual_coef_,support_vectors)
	# print coef
	temp = clf.score(X[data_size:len(X)],Y1[data_size:len(Y1)])
	print temp
	temp = predict(X[data_size:len(X)],coef.T,clf.intercept_)

	k=0
	count=0
	for i in Y1[data_size:len(Y1)]:
		if  i==temp[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	#print len(Y1[data_size:len(Y1)])	
	print 100*(count/float(len(Y1[data_size:len(Y1)])))
			
