#!/usr/bin/env python
# coding: utf-8
%matplotlib inline
# In[12]:


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns  


# # WEEK 1
Data Exploration:

1. Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:

• Glucose

• BloodPressure

• SkinThickness

• Insulin

• BMI

2. Visually explore these variables using histograms. Treat the missing values accordingly.

3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables. 
# In[13]:


data = pd.read_csv('health care diabetes.csv')


# In[14]:


data.head()


# In[15]:


data.isnull().any()


# In[10]:


data.info()


# In[33]:


Positive = data[data['Outcome']==1]
Positive.head(15)


# In[21]:


data['Glucose'].value_counts().head(15)


# In[22]:


plt.hist(data['Glucose'])


# In[23]:


data['BloodPressure'].value_counts().head(15)


# In[24]:


plt.hist(data['BloodPressure'])


# In[25]:


data['SkinThickness'].value_counts().head(15)


# In[26]:


plt.hist(data['SkinThickness'])


# In[27]:


data['Insulin'].value_counts().head(15)


# In[28]:


plt.hist(data['Insulin'])


# In[29]:


data['BMI'].value_counts().head(15)


# In[30]:


plt.hist(data['BMI'])


# In[31]:


data.describe().transpose()


# # WEEK 2
Data Exploration:

1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.

2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.

3. Perform correlation analysis. Visually explore it using a heat map.Lets check the positive cases, for detailed graph, increase the bins for histogram
# In[32]:


plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)


# In[34]:


Positive['BMI'].value_counts().head(15)


# In[35]:


plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)


# In[36]:


Positive['Glucose'].value_counts().head(15)


# In[37]:


plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)


# In[38]:


Positive['BloodPressure'].value_counts().head(15)


# In[39]:


plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)


# In[40]:


Positive['SkinThickness'].value_counts().head(15)


# In[41]:


plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)


# In[42]:


Positive['Insulin'].value_counts().head(15)

Scatter plot
# In[43]:


BloodPressure = Positive['BloodPressure']
Glucose = Positive['Glucose']
SkinThickness = Positive['SkinThickness']
Insulin = Positive['Insulin']
BMI = Positive['BMI']


# In[47]:


plt.scatter(BloodPressure, Glucose, color=['g'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('Blood Pressure-Glucose Relation')
plt.show()

Differentiate the relation based on Outcome variable
# In[48]:


g =sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
              hue="Outcome",
              data=data);


# In[49]:


B =sns.scatterplot(x= "BMI" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[50]:


S =sns.scatterplot(x= "SkinThickness" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[51]:


### correlation matrix to find out the relation among each parameter with the others. This analysis gives us the dependency of these.
data.corr()


# In[54]:


### create correlation heat map
sns.heatmap(data.corr())
#as expected Glucose has highest value for Outcome as 0.46 - diabetes is caused by sugar mostly


# In[61]:


plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='plasma') #heatmap with values


# In[113]:


# Logistic Regression and model building
#since we have 2 categorical varaibles to build a model by, we would use Logitical Regression algo
#and the Outcome variable can have 2 values 0/1 - it is a Binomial Logistic Regression Classification Algo


# In[62]:


data.head(5)


# In[66]:


features = data.iloc[:,[0,1,2,3,4,5,6,7]].values #all the feature values - used to decide the Outcome
label = data.iloc[:,8].values #all the outcome values, used for Labelling the patient as 0 or 1
print(label)
print(features)


# In[68]:


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.2,
                                                random_state =10)
print(X_train)


# In[86]:


#Create model
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train) 


# In[87]:


#find the scores in Train and test data
print(model_lr.score(X_train,y_train))
print(model_lr.score(X_test,y_test))
#seems close enough


# In[88]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,model_lr.predict(features))
cm
#problem here is 122 - where the patient is actually diabeteic but our model said No - Type 2 error


# In[89]:


#lets find the accuracy of the model
from sklearn.metrics import classification_report
print(classification_report(label,model_lr.predict(features)))


# In[90]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve) to predict the probablities and thresholds
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model_lr.predict_proba(features)
print(probs)

# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)

# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')

Lets apply various algos to compare the outcome with KNN
# In[91]:


#Apply Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(max_depth=5)
model_dt.fit(X_train,y_train)


# In[92]:


model_dt.score(X_train,y_train)


# In[93]:


model_dt.score(X_test,y_test)


# In[94]:


#Apply Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=11)
model_rf.fit(X_train,y_train)


# In[95]:


model_rf.score(X_train,y_train)


# In[96]:


model_rf.score(X_test,y_test)


# In[100]:


#Support Vector Classifier

from sklearn.svm import SVC 
model_sv = SVC(kernel='rbf',
           gamma='auto')
model_sv.fit(X_train,y_train)


# In[106]:


model_sv.score(X_train,y_train)


# In[107]:


model_sv.score(X_test,y_test)


# In[108]:


#Apply K-NN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=7,
                             metric='minkowski',
                             p = 2)
model_knn.fit(X_train,y_train)


# In[120]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model_knn.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
print("True Positive Rate - {}, False Positive Rate - {} Thresholds - {}".format(tpr,fpr,thresholds))
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

We can use Precision Recall Curves which gives better results than the ROC
# In[119]:


#Precision Recall Curve for Logistic Regression

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model_lr.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='*')


# In[118]:


#Precision Recall Curve for KNN

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model_knn.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model_knn.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='*')


# In[117]:


#Precision Recall Curve for Decission Tree Classifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model_dt.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model_dt.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='*')


# In[115]:


#Precision Recall Curve for Random Forest

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model_rf.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model_rf.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='*')

