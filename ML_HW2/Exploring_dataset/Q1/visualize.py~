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
def load_h5py('./../../'+filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y


X,Y = load_h5py(args.data)
color=["#000000","#FF0000","#FFF000"]

plt.scatter(X[:, 0], X[:, 1], color=[color[i] for i in Y])
plt.show()
plt.close()


