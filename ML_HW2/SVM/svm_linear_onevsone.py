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
	final_res = {}
	for i in range(0,num):
		for j in range(0,num):
			if i!=j:
				Y = []
				X_temp = []
				jj=0
				for a in Y1[0:data_size]:
					if(i==a):
						Y.append(1)
						X_temp.append(X[jj])
						
					elif(j==a):	
						Y.append(0)
						X_temp.append(X[jj])
					
					jj = jj+1
				print len(Y)
				clf= SVC(kernel =kernel_)		
				print X_temp[0:data_size]
				print Y[0:data_size]
				clt= clf.fit(X_temp[0:data_size],Y[0:data_size])
				print clt.support_
				support_vectors=[]
				for hh in clf.support_:
					support_vectors.append(X_temp[hh])

				support_vectors = np.array(support_vectors)	
				coef = np.matmul(clf.dual_coef_,support_vectors)
				print coef
				final_res[(i,j)]=(predict(X[data_size:len(X)],coef.T,clf.intercept_))
			
	print final_res
	pred = []
	for s in range(0,len(X[data_size:len(X)])):
		for i in range(0,num):
			flag=0
			for j in range(i+1,num):
				if(final_res[i,j][s] ==0):
					flag=1
			if flag==0:
				pred.append(i)

	k=0
	count=0
	for i in Y1[data_size:len(Y1)]:
		if  i==pred[k]:
			count=count+1
		k=k+1	
	print count,len(Y1[data_size:len(Y1)])
	print 100*(count/float(len(Y1[data_size:len(Y1)])))
	
	
	
	
		

