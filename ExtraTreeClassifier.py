import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# UNIVARIATE SELECTION

data = pd.read_csv('C:\\Users\\JOHN\\Downloads\\train1.csv')
X = data.iloc[:,0:14]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# FEATURE IMPORTANCE
data = pd.read_csv('C:\\Users\\JOHN\\Downloads\\train11.csv')
X = data.iloc[:,0:14]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
################# Im doing
features = [10]
features.append(feat_importances.nlargest(10))
print('My print')
imp_features=[]
i=0
while i<10 :
    imp_features.append(features[1].index[i])
    print(i,imp_features[i])
    i += 1
plt.show()