import numpy as np
import pandas as pd

training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()

x = training_data.iloc[:, :-1].values
y = training_data.iloc[:,-1].values


#split the date with scikit learn

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
y_prob = classifier.predict_proba(x_test)[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report


new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))
new_prediction1 = classifier.predict(sc.transform(np.array([[40,60000]])))
new_prediction3 = classifier.predict(sc.transform(np.array([[32,92000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[32,82000]])))[:,1]


import pickle 

model_file = 'classifier.pickle'

pickle.dump(classifier, open(model_file,'wb'))

scaler_file = 'sc.pickle'

pickle.dump(sc, open(scaler_file,'wb'))

