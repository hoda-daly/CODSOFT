import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.shape
data.describe().T
fraud = data[data.Class == 1]
valid = data[data.Class == 0]
outlierFraction = len(fraud) / float(len(valid))
outlierFraction 
print(f'Fraud Cases: {len(fraud)}')
print(f'Valid Transactions: {len(valid)}')
fraud.Amount.describe()
valid.Amount.describe()
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()
X = data.drop(['Class'], axis = 1)
y = data.Class
X.shape, y.shape
X_data = X.values
y_data = y.values
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = .2,
                                                   random_state = 42)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
acc = accuracy_score(y_test, pred)
acc
prec = precision_score(y_test, pred)
prec
rec = recall_score(y_test, pred)
rec
f1 = f1_score(y_test, pred)
f1
mcc = matthews_corrcoef(y_test, pred)
mcc
labels = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, pred)
plt.figure(figsize = (12, 12))
sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, 
           fmt = 'd')
plt.title('confusion matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()