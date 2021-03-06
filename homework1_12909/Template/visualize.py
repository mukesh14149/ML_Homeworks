from matplotlib import pyplot as plt
import os
import os.path
import argparse
import h5py
from sklearn.manifold import TSNE
parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str )
args = parser.parse_args()

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


X,Y = load_h5py(args.data)
X_tsne = TSNE(learning_rate=100,verbose=1,n_components=2,perplexity=50.0).fit_transform(X)
# Change Y dataset from ndarray to simple list
Y_tsne = []
for i in range(0,len(Y)):
	Y_tsne.append(Y[i].tolist().index(1.0))
print Y_tsne


fig=plt.figure()
color=["#000000","#FF0000","#FFFF00","#808000", "#008000","#00FFFF","#0000FF","#800080","#FF00FF","#008080"]
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=[color[i] for i in Y_tsne])
plt.savefig(args.plots_save_dir)	


