
###########################     Logistic Regression
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
# define dataset
import pandas as pd
import numpy as np

#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# summarize the dataset

#print(X.shape, y.shape)


# test regression dataset


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
# define dataset
data = pd.read_csv('C:\\Users\\JOHN\\Downloads\\train11.csv')
X = data.iloc[:,0:15]  #independent columns #### 19 for im apps
y = data.iloc[:,-1]    #target column i.e price range
# define the model
model = LogisticRegression()
# fit the model
model.fit(X, y)
# get importance
importance = pd.Series(model.coef_[0], index=X.columns)
importance.nlargest(10).plot(kind='barh')
# summarize feature importance
#for i,v in enumerate(importance):
	#print('Feature: %0d, Score: %.5f' % (i,v))
	#print()
# plot feature importance
#pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()

features = [15]
features.append(importance.nlargest(15))
print('My print')
imp_features=[]
i=0
while i<15 :
    imp_features.append(features[1].index[i])
    print(i,imp_features[i])
    i += 1

