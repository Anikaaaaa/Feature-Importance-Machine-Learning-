from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import pandas as pd

# define dataset
data = pd.read_csv('C:\\Users\\JOHN\\Downloads\\train11.csv')
X = data.iloc[:,0:14]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

# define the model
model = RandomForestClassifier()
#model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = pd.Series(model.feature_importances_,index=X.columns)
importance.nlargest(10).plot(kind='barh')
# summarize feature importance
#for i,v in enumerate(importance):
	#print('Feature: %0d, Score: %.5f' % (i,v))
	#print()
# plot feature importance
#pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()

features = [10]
features.append(importance.nlargest(10))
print('My print')
imp_features=[]
i=0
while i<10 :
    imp_features.append(features[1].index[i])
    print(i,imp_features[i])
    i += 1

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))