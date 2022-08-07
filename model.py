import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pickle
from sklearn.metrics import accuracy_score,confusion_matrix
# %matplotlib inline

def data_clean():
    dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':0})
    dataset['Outcome'] = dataset['Outcome'].map({'Positive':1,'Negative':0})
    dataset['Polyuria'] = dataset['Polyuria'].map({'Yes':1,'No':0})
    dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes':1,'No':0})
    dataset['Sudden_weight_loss'] = dataset['Sudden_weight_loss'].map({'Yes':1,'No':0})
    dataset['Weakness'] = dataset['Weakness'].map({'Yes':1,'No':0})
    dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes':1,'No':0})
    dataset['Genital_thrush'] = dataset['Genital_thrush'].map({'Yes':1,'No':0})
    dataset['Visual_blurring'] = dataset['Visual_blurring'].map({'Yes':1,'No':0})
    dataset['Itching'] = dataset['Itching'].map({'Yes':1,'No':0})
    dataset['Irritability'] = dataset['Irritability'].map({'Yes':1,'No':0})
    dataset['Delayed_healing'] = dataset['Delayed_healing'].map({'Yes':1,'No':0})
    dataset['Partial_paresis'] = dataset['Partial_paresis'].map({'Yes':1,'No':0})
    dataset['Muscle_stiffness'] = dataset['Muscle_stiffness'].map({'Yes':1,'No':0})
    dataset['Alopecia'] = dataset['Alopecia'].map({'Yes':1,'No':0})
    dataset['Obesity'] = dataset['Obesity'].map({'Yes':1,'No':0})

    return dataset




dataset = pd.read_csv('diabetes_data.csv')
#dataset.head()

dataset = data_clean(dataset)



print(dataset['Outcome'].value_counts())
print(dataset.columns)

'''
dataset['Gender'] = dataset['Gender'].map({'Male':1,'Female':0})
dataset['Outcome'] = dataset['Outcome'].map({'Positive':1,'Negative':0})
dataset['Polyuria'] = dataset['Polyuria'].map({'Yes':1,'No':0})
dataset['Polydipsia'] = dataset['Polydipsia'].map({'Yes':1,'No':0})
dataset['Sudden_weight_loss'] = dataset['Sudden_weight_loss'].map({'Yes':1,'No':0})
dataset['Weakness'] = dataset['Weakness'].map({'Yes':1,'No':0})
dataset['Polyphagia'] = dataset['Polyphagia'].map({'Yes':1,'No':0})
dataset['Genital_thrush'] = dataset['Genital_thrush'].map({'Yes':1,'No':0})
dataset['Visual_blurring'] = dataset['Visual_blurring'].map({'Yes':1,'No':0})
dataset['Itching'] = dataset['Itching'].map({'Yes':1,'No':0})
dataset['Irritability'] = dataset['Irritability'].map({'Yes':1,'No':0})
dataset['Delayed_healing'] = dataset['Delayed_healing'].map({'Yes':1,'No':0})
dataset['Partial_paresis'] = dataset['Partial_paresis'].map({'Yes':1,'No':0})
dataset['Muscle_stiffness'] = dataset['Muscle_stiffness'].map({'Yes':1,'No':0})
dataset['Alopecia'] = dataset['Alopecia'].map({'Yes':1,'No':0})
dataset['Obesity'] = dataset['Obesity'].map({'Yes':1,'No':0})

'''

#corrdata = dataset.corr()



X1 = dataset.iloc[:,0:-1]
y1 = dataset.iloc[:,-1]


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
best_feature = SelectKBest(score_func=chi2,k=10)
fit = best_feature.fit(X1,y1)
dataset_scores = pd.DataFrame(fit.scores_)
dataset_cols = pd.DataFrame(X1.columns)
featurescores = pd.concat([dataset_cols,dataset_scores],axis=1)
featurescores.columns=['column','scores']



featureview=pd.Series(fit.scores_, index=X1.columns)
featureview.plot(kind='barh')

from sklearn.feature_selection import VarianceThreshold
feature_high_variance = VarianceThreshold(threshold=(0.5*(1-0.5)))
falls=feature_high_variance.fit(X1)
dataset_scores1 = pd.DataFrame(falls.variances_)
dat1 = pd.DataFrame(X1.columns)
high_variance = pd.concat([dataset_scores1,dat1],axis=1)
high_variance.columns=['variance','cols']
high_variance[high_variance['variance']>0.2]

X = dataset[['Polydipsia','Sudden_weight_loss','Partial_paresis','Irritability','Polyphagia','Age','Visual_blurring']]
y = dataset['Outcome']

X.to_csv('diabetes_data_processed.csv')



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=lg, X=X_train ,y=y_train,cv=10)
#print("accuracy is {:.2f} %".format(accuracies.mean()*100))
#print("std is {:.2f} %".format(accuracies.std()*100))

pre=lg.predict(X_test)

logistic_regression=accuracy_score(pre,y_test)
#print(accuracy_score(pre,y_test))
#print(confusion_matrix(pre,y_test))

from sklearn.metrics import classification_report
#print(classification_report(pre,y_test))

from sklearn.svm import SVC
sv=SVC(kernel='linear',random_state=0)
sv.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=sv, X=X_train ,y=y_train,cv=10)
#print("accuracy is {:.2f} %".format(accuracies.mean()*100))
#print("std is {:.2f} %".format(accuracies.std()*100))

pre1=sv.predict(X_test)
svm_linear=accuracy_score(pre1,y_test)
#print(accuracy_score(pre1,y_test))
#print(confusion_matrix(pre1,y_test))

from sklearn.metrics import classification_report
#print(classification_report(pre1,y_test))

from sklearn.neighbors import KNeighborsClassifier
score=[]

for i in range(1,10):
    
    
    knn=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
    knn.fit(X_train,y_train)
    pre3=knn.predict(X_test)
    ans=accuracy_score(pre3,y_test)
    score.append(round(100*ans,2))

#print(sorted(score,reverse=True)[:5])
knn=sorted(score,reverse=True)[:1]

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion='gini')
dc.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=dc, X=X_train ,y=y_train,cv=10)
#print("accuracy is {:.2f} %".format(accuracies.mean()*100))
#print("std is {:.2f} %".format(accuracies.std()*100))

pre5=dc.predict(X_test)
Decisiontress_classifier=accuracy_score(pre5,y_test)
#print(accuracy_score(pre5,y_test))
#print(confusion_matrix(pre5,y_test))

#print('Logistic regression:',logistic_regression)
#print('svmlinear:',svm_linear)
#print('knn:',knn)
print('Decision tress:',Decisiontress_classifier)


with open('model.pkl', 'wb') as files:
    pickle.dump(knn, files)



