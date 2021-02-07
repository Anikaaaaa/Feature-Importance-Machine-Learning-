
############### Permutation Feature Importance for Classification
############### KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
from matplotlib import pyplot
# define dataset
data = pd.read_csv('C:\\Users\\JOHN\\Downloads\\train11.csv')
X = data.iloc[:,0:14]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(X, y)
# perform permutation importance
results = permutation_importance(model, X, y, scoring='accuracy')
# get importance
importance = pd.Series(results.importances_mean, index=X.columns)
# summarize feature importance
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