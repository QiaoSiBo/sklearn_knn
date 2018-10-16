#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


###choose k value
k = 15
###load iris datasets
iris = datasets.load_iris()
###get first two features
X = iris.data[:,:2]
X_test = iris.data[:,2:4]
y = iris.target
###step size in the mesh
h = .01
###create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
for weight in ['uniform','distance']:
    ###we create an instance of Neighbours Classifier and fit the data
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights=weight)
    clf.fit(X,y)
    ### Plot the decision boundary. For that, we will assign a color to each
    ###point in the mesh [x_min, x_max] * [y_min, y_max]
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    Z = clf.predict(np.stack((xx.ravel(),yy.ravel()),axis=1))
    
    ###Put the result into a color plot
     
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
     
    ###Plot also the training points
    plt.scatter(X[:,0],X[:,1], c=y,cmap=cmap_bold, edgecolors='k',s=20)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("3-class classification(k = %i, weights=%s)" %(k, weight))
     
plt.show()
    



