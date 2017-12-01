import json
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

connection_file = open('train.json','r')
train = json.load(connection_file)
connection_file = open('test.json','r')
test = json.load(connection_file)

# one=[]
# two=[]
# three=[]
# four=[]
# five=[]
# print train[0]
# for i in range(0,len(train)):
# 	if train[i]['Y'] == 1:
# 		one.append(train[i])
# 	elif train[i]['Y'] == 2:
# 	 	two.append(train[i])
# 	elif train[i]['Y'] == 3:
# 	 	three.append(train[i])
# 	elif train[i]['Y'] == 4:
# 	 	four.append(train[i])
# 	elif train[i]['Y'] == 5:
# 	 	five.append(train[i])
	
# #print two,three,four,five
# print len(one),len(two),len(three),len(four),len(five)

train_data=[]
train_output=[]
max1=0
count=0
for i in range (0,len(train)):
	# if max1<len(train[i]['X']):
	# 	max1 = len(train[i]['X'])
	# if(len(train[i]['X'])==189):
	# 	count+=1
	train_data.append(train[i]['X'])
	train_output.append(train[i]['Y'])



Data = []
for i in range(len(train_data)):
	result = [str(x) for x in train_data[i]]
	temp = " ".join(result)
	Data.append(temp)
train_data = np.array(train_data)
print train_data[0]
print Data[0]
# unique, counts = np.unique(train_data, return_counts=True)
# print np.shape(train_data)

tfid = TfidfVectorizer(ngram_range=(1,2))
train_data= tfid.fit_transform(Data)

test_data = []
for i in range (0,len(test)):
	test_data.append(test[i]['X'])

Data = []
for i in range(len(test_data)):
	result = [str(x) for x in test_data[i]]
	temp = " ".join(result)
	Data.append(temp)

test_data = tfid.transform(Data)


# # datasetX=[]
# # datasetY=[]
# # print np.shape(train_data)
# # for h in range(1,6):	
# # 	t=0
# # 	tempX=[]
# # 	tempY=[]
# # 	for i,j in train_data:
# # 		if train_output[t]==h:
# # 			tempX.append([i,j])
# # 			tempY.append(train_output[t])
# # 		t=t+1

# # 	tempX = np.array(tempX)	
# # 	#print np.shape(tempX)
# # 	mean_1  = np.mean(tempX[:,0])
# # 	mean_2  = np.mean(tempX[:,1])
# # 	std_1 = np.std(tempX[:,0])
# # 	std_2 = np.std(tempX[:,1])
	
	
# # 	print mean_1,mean_2,std_1,std_2
# # 	t=0
# # 	for i,j in tempX:
# # 		x1,y1=drop_outliers(i,j,mean_1,mean_2,std_1,std_2)
# # 		if x1!=10 and y1!=10:
# # 			datasetX.append([x1,y1])
# # 			datasetY.append(tempY[t])
# # 		t=t+1
		
	
# # datasetX = np.array(datasetX)
# # #write_h5py(args.data,datasetX,datasetY)
# # # print np.shape(datasetX)
# # # color=["#000000","#FF0000","#FFF000","#FFF001","#FF4000","#100000"]
# # # plt.scatter(datasetX[:, 0], datasetX[:, 1], color=[color[i] for i in datasetY])
# # # plt.show()
# # # plt.close()



clf = LinearSVC(C=0.5)
print "fit starting",np.shape(train_data), np.shape(train_output)
clt = clf.fit(train_data,train_output)

joblib.dump(clf, "kaggle.pkl")

output = clf.predict(test_data)

f = open('kaggle.csv', 'w')
l=1
f.write('Id,Expected\n')
for i in output:
	f.write(str(l))
	f.write(",")
	f.write(str(i))
	f.write('\n') 
	l=l+1
f.close()