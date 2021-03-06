#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 18:07:52 2018

@author: shivam
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################################
data = pd.read_csv('/home/shivam/Desktop/Iris/Iris.csv')
data.head()


data.drop('Id',axis=1,inplace=True)


##################################################################################

sns.pairplot(data, hue='Species', size=3)


plt.savefig('/home/shivam/Desktop/Iris/B07887_01_14.png', format='png', dpi=300)
#####################################################################

# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sns.pairplot(data, hue="Species", size=3, diag_kind="kde")



plt.savefig('/home/shivam/Desktop/Iris/B07887_01_15.png', format='png', dpi=300)
########################################

# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
data.boxplot(by="Species", figsize=(12, 6))


plt.savefig('/home/shivam/Desktop/Iris/B07887_01_16.png', format='png', dpi=500)

#########################################



# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(data, "Species")

plt.savefig('/home/shivam/Desktop/Iris/B07887_01_117.png', format='png', dpi=500)



############################################

# Another multivariate visualization technique pandas has is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(data, "Species")

plt.savefig('/home/shivam/Desktop/Iris/B07887_01_120.png', format='png', dpi=500)




############################################


# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(data, "Species")
plt.savefig('/home/shivam/Desktop/Iris/B07887_01_118.png', format='png', dpi=200)

######################################################################################################

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data['Species'] = LabelEncoder().fit_transform(data['Species'])
data.iloc[[0,1,-2,-1],:]

############################ Pipe Lineing



pipeline = Pipeline([
    ('normalizer', StandardScaler()), #Step1 - normalize data
    ('clf', LogisticRegression()) #step2 - classifier
])
pipeline.steps



##########################


##################################################
###############################################################from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key,' mean ', values.mean())
            print(key,' std ', values.std())



#Seperate train and test data
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1].values,
                                                   data['Species'],
                                                   test_size = 0.3,
                                                   random_state = 10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
scores

scores['test_score'].mean()

###################################################################################################3
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())

for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key,' mean ', values.mean())
            print(key,' std ', values.std())


############################# Cross _Validation

from sklearn.model_selection import GridSearchCV
pipeline.set_params(clf= SVC())
pipeline.steps




cv_grid = GridSearchCV(pipeline, param_grid = {
    'clf__kernel' : ['linear', 'rbf'],
    'clf__C' : np.linspace(0.1,1.2,12)
})

cv_grid.fit(X_train, y_train)





cv_grid.best_params_


cv_grid.best_estimator_


cv_grid.best_score_


y_predict = cv_grid.predict(X_test)
accuracy = accuracy_score(y_test,y_predict)
print('Accuracy of the best classifier after CV is %.3f%%' % (accuracy*100))

############### Ploting of Figure ####################

