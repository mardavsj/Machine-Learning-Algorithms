import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1) # size of tumor 
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0 for no-cancer and 1 for cancer

logr = linear_model.LogisticRegression()
logr.fit(X,y)

predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)