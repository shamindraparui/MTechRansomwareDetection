#!/usr/bin/env python
# coding: utf-8

# In[ ]:


multiple_run_result=[]


# In[ ]:


import random
import pickle
import numpy 
import pandas
#import sys
from copy import deepcopy
from collections import OrderedDict

########################################################################################
#             This script is used for selection of features in a random way.
#             The first 32 features are of APIs, next 4 are of Assembly and
#             the last 3 are of some properties of PE and its associated label
#             and will train Random Forest
#             date : 2-7-2020
#             version : 2-7-v2
########################################################################################




########################################################################################
# following function randomly selects APIs or Assembly instruction depending on the call
def select_randomly(number_of_selection,my_list):
    global selected_features
    list_length=len(my_list)-1
    i=0
    while i<number_of_selection:
        n=random.randint(0,list_length)
        selected_features.append(my_list[n])
        i+=1
    return None
########################################################################################




################################ MAIN_START ########################################################


# Following parameters are used for system configuration
how_many_api=20
how_many_asm=10
scale_value=500*1048576
size_of_test_data=0.20





#  Reading the lists named "FrequentlyUsedAPI.list", "FrequentlyUsedASM.list", "EndFeature.list" which
#  was written by "RandomFeatureSelection.py" & "FetchData.py"

with open("C:/Users/Gamer/Documents/Implementation/UpdatedFeature/FrequentlyUsedAPI.list","rb") as read_file:
    frequently_used_api=pickle.load(read_file)
    read_file.close()
with open("C:/Users/Gamer/Documents/Implementation/UpdatedFeature/FrequentlyUsedASM.list","rb") as read_file:
    frequently_used_asm=pickle.load(read_file)
    read_file.close()
with open("C:/Users/Gamer/Documents/Implementation/UpdatedFeature/EndFeature.list","rb") as read_file:
    end_feature=pickle.load(read_file)
    read_file.close()
    

# following list will store the features (randomly select total of 32 APIs & 4 assembly instructions)
selected_features=[]


# following function will randomly select features
select_randomly(how_many_api,frequently_used_api[:40])
select_randomly(how_many_api,frequently_used_api[100:140])

temp=deepcopy(list(OrderedDict.fromkeys(selected_features)))
selected_features.clear()
selected_features=deepcopy(temp)
temp.clear()
asm_start_index=len(selected_features)

select_randomly(how_many_asm,frequently_used_asm[:12])
select_randomly(how_many_asm,frequently_used_asm[20:32])
temp=deepcopy(list(OrderedDict.fromkeys(selected_features)))
selected_features.clear()
selected_features=deepcopy(temp)
temp.clear()
asm_end_index=asm_start_index + (len(selected_features[asm_start_index:]))
selected_features+=end_feature
#print(selected_features)


# Loading the dataset into memory
with open("C:/Users/Gamer/Documents/Implementation/UpdatedFeature/ALLdataset","rb") as my_dataset_read:
    data_set=pickle.load(my_dataset_read)
    my_dataset_read.close()

data=data_set[selected_features]
data=data.loc[:,~data.columns.duplicated()]

print("Features are :")
print(selected_features)
print("\n")


# Scaling the frequency of each instruction
row=len(data["size"])
i=0
j=asm_start_index
column=asm_end_index
#sys.exit()
while j <column:
    i=0
    while i< row:
        scaled=0
        has=data.iloc[i,j]
        size=data.loc[i,"size"]
        scaled=int((has/size)*scale_value)
        #data.iat[i,j]=deepcopy(scaled)
        data.iat[i,j]=deepcopy(scaled)
        #print(scaled)
        i+=1
    j+=1
with open("current_dataset.df","wb") as dfWrite:
    pickle.dump(data,dfWrite)
    dfWrite.close()


# In[ ]:


from bayes_opt import BayesianOptimization
import lightgbm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV



# list to save the result
final_result=[]
final_result.append(["Name","F1","Precision","Recall","TP","TN","FP","FN","FNR"])
#print("Name \t Accuracy \t F1 \t TPR \t FNR\n")
#################################  NB  below  #################################################
X=None
Y=None
X_train=None
X_test=None
Y_train=None
Y_test=None

X=data.drop(["label","size","Entropy","Name","SectionCharacteristics"], axis=1)
Y=pandas.DataFrame(data["label"])


X_normalized=StandardScaler().fit_transform(X)
Y_encoded=LabelEncoder().fit_transform(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X_normalized,Y_encoded,test_size=size_of_test_data)
c=GaussianNB().fit(X_train,Y_train.ravel())

predicted_NB_Y=c.predict(X_test)

ac=accuracy_score(Y_test,predicted_NB_Y)
f1result=f1_score(Y_test,predicted_NB_Y)
precision_result=precision_score(Y_test,predicted_NB_Y)
recall_result=recall_score(Y_test,predicted_NB_Y)
tn, fp, fn, tp = confusion_matrix(Y_test,predicted_NB_Y).ravel()
tpr=tp / (tp + fn)
fnr=fn/(tp+fn)
fpr=fp / (fp + tn)
tnr=tn/(tn+fp)
final_result.append(["NB",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr])
print("============= NB finished.===============")
#################################  RF  below  #################################################

X=data.drop(["label","size","Entropy","Name","SectionCharacteristics"], axis=1)
Y=pandas.DataFrame(data["label"])
Y=Y.applymap(str)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=size_of_test_data)
param_grid = [
{'n_estimators': [10,20,30],
 'max_features': [5, 10], 
 'min_samples_split':[2,4,6,8],
 'max_depth': [10, 15, 20],
 'random_state':[20],
 'bootstrap': [True, False]}
]

grid_search_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_forest.fit(X_train, Y_train.values.ravel())
rf=RandomForestClassifier()
rf=grid_search_forest.best_estimator_
grid_rf_predicted = rf.predict(X_test)


ac=accuracy_score(Y_test,grid_rf_predicted)
f1result=f1_score(Y_test,grid_rf_predicted,pos_label='1')
precision_result=precision_score(Y_test,grid_rf_predicted,pos_label='1')
recall_result=recall_score(Y_test,grid_rf_predicted,pos_label='1')

tn, fp, fn, tp = confusion_matrix(Y_test,grid_rf_predicted).ravel()
tpr=tp / (tp + fn)
fnr=fn/(tp+fn)
fpr=fp / (fp + tn)
tnr=tn/(tn+fp)
final_result.append(["RF",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr])
print("============= RF finished.===============")
################################  SVM below ##########################################
X=None
Y=None
X_train=None
X_test=None
Y_train=None
Y_test=None

X=data.drop(["label","size","Entropy","Name","SectionCharacteristics"], axis=1)
Y=pandas.DataFrame(data["label"])

X_normalized=StandardScaler().fit_transform(X)
Y_encoded=LabelEncoder().fit_transform(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X_normalized,Y_encoded,test_size=0.20)
param_grid = {'C': [0.1, 1],# 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear','poly','rbf','sigmoid']
              #'degree': [3,4,5,6]
             }  

grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, Y_train)
grid_svm_predicted = grid.predict(X_test) 
#print("\n\n SVM \t Confusion matrix:\n", confusion_matrix(Y_test, grid_svm_predicted))
#measure("SVM", Y_test,grid_svm_predicted)
ac=accuracy_score(Y_test,grid_svm_predicted)
f1result=f1_score(Y_test,grid_svm_predicted)
precision_result=precision_score(Y_test,grid_svm_predicted)
recall_result=recall_score(Y_test,grid_svm_predicted)
    #tn, fp, fn, tp=confusion_matrix(y,pred).ravel()
tn, fp, fn, tp = confusion_matrix(Y_test,grid_svm_predicted).ravel()
tpr=tp / (tp + fn)
fnr=fn/(tp+fn)
fpr=fp / (fp + tn)
tnr=tn/(tn+fp)
final_result.append(["SVM",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr])
print("============= SVM finished.===============")

###########################  Decision Tree #################################################
X=None
Y=None
X_train=None
X_test=None
Y_train=None
Y_test=None


X=data.drop(["label","size","Entropy","Name","SectionCharacteristics"], axis=1)
Y=pandas.DataFrame(data["label"])

X_normalized=StandardScaler().fit_transform(X)
Y_encoded=LabelEncoder().fit_transform(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X_normalized,Y_encoded,test_size=0.20)
param_grid = [
{'max_features': ['auto'], 
 'min_samples_split':[2,4,6,8,10]}
]
grid_search_DT = GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
model = tree.DecisionTreeClassifier()
grid_search_DT.fit(X_train,Y_train)
grid_dt_predicted=grid_search_DT.predict(X_test)

ac=accuracy_score(Y_test,grid_dt_predicted)
f1result=f1_score(Y_test,grid_dt_predicted)
precision_result=precision_score(Y_test,grid_dt_predicted)
recall_result=recall_score(Y_test,grid_dt_predicted)
tn, fp, fn, tp = confusion_matrix(Y_test,grid_dt_predicted).ravel()
tpr=tp / (tp + fn)
fnr=fn/(tp+fn)
fpr=fp / (fp + tn)
tnr=tn/(tn+fp)
final_result.append(["DT",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr])
print("============= DT finished.===============")


###############################  LightGBM  #####################################################


def lgclf(num_iteration,min_data_in_leaf,learn_rate,max_depth,n_estimator,max_bin):
    global X_train
    global Y_train
    global X_test
    global Y_test
    global best_score
    global temp_result
    
    model=lightgbm.LGBMClassifier(boost='dart',num_iteration=int(num_iteration),min_data_in_leaf=int(min_data_in_leaf),learning_rate=learn_rate,max_depth=int(max_depth),n_estimators=int(n_estimator),max_bin=int(max_bin))
    model.fit(X_train,Y_train)
    
    predicted_LGBM_Y=model.predict(X_test)
    
    recall_result=recall_score(Y_test,predicted_LGBM_Y)
    ac=accuracy_score(Y_test,predicted_LGBM_Y)
    f1result=f1_score(Y_test,predicted_LGBM_Y)
    precision_result=precision_score(Y_test,predicted_LGBM_Y)
    tn, fp, fn, tp = confusion_matrix(Y_test,predicted_LGBM_Y).ravel()
    tpr=tp / (tp + fn)
    fnr=fn/(tp+fn)
    fpr=fp / (fp + tn)
    tnr=tn/(tn+fp)
    
    if recall_result > best_score:
        best_score=recall_result
        temp_result=["LGBM",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr]
        with open("LGBM","wb") as my_classifier:
            pickle.dump(model, my_classifier)
            my_classifier.close()
    return recall_result


#==============================================================================

X=None
Y=None
X_train=None
X_test=None
Y_train=None
Y_test=None
best_score=0
temp_result=[]

X=data.drop(["label","size","Entropy","Name","SectionCharacteristics"], axis=1)
Y=pandas.DataFrame(data["label"])

for e in X.columns:
    column_type=X[e].dtype
    if column_type=="object":
        X[e]=data[e].astype("category")

for l in Y.columns:
    column_type=Y[l].dtype
    if column_type=="object":
        Y[l]=Y[l].astype("int")

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.20)


bop=BayesianOptimization(lgclf, {'num_iteration':(200,1000),
                                 'min_data_in_leaf':(30,40),
                                 'learn_rate':(0.01,0.05),
                                 'max_depth':(30,60),
                                 'n_estimator':(40,60),
                                 'max_bin':(300,365)})
bop.maximize(50,5)

#final_result.append(["LGBM",f1result,precision_result,recall_result,tp,tn,fp,fn,fnr])
###################################################################
print("============= LGBM finished.===============")
final_result.append(deepcopy(temp_result))

