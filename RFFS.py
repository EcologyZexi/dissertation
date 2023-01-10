#!/usr/bin/env python
# coding: utf-8

# In[3]:


# suppress warning 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
myfont = matplotlib.font_manager.FontProperties(fname='heiti.ttf',size=40)
plt.rcParams['axes.unicode_minus']=False


#MLP
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import time
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from typing import List
from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score


# In[16]:


df_sample = pd.read_csv('class/tree1520_trainv30.csv',encoding='unicode_escape')
df_sample


# In[17]:


# replace inf/-inf with np.nan
df_sample.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sample


# In[18]:


df_sample=df_sample.replace(np.nan,0)
df_sample


# In[19]:


# delete all inf/-inf rows (totally-119)
df_sample=df_sample.dropna(axis=0,how='any') #drop all rows that have any NaN values
df_sample


# In[28]:


df_samplev1= df_sample.drop('Typev6', axis=1)
df_samplev1


# In[32]:


# select all features
x1 = df_samplev1.iloc[:,1:126]
x1


# In[23]:


label = df_sample['Typev6']
label


# In[24]:


from pandas.core.frame import DataFrame
label=DataFrame(label)
label


# In[25]:


# split the data into train, test set respectively
train_data1,test_data1,train_label1,test_label1 = train_test_split(x1,label,random_state=1, train_size=0.75,test_size=0.25)


# In[33]:


rf1 = RandomForestClassifier(n_estimators=570,
#                               max_depth = None,
#                               bootstrap = True,
#                               max_features = 0.3,
#                               oob_score=True,
#                               class_weight={'R': 100,'P':20,'E':100,'B':40},
#                             class_weight='balanced',   
                             random_state=1)
#                               min_samples_leaf=3,
#                               min_samples_split=2)
#                               criterion='entropy')
#classifier.fit(train_data, train_label.ravel())
rf1.fit(train_data1, train_label1)
# Predicting the Test set results
test_pred1 = rf1.predict(test_data1)


# In[34]:


def show_reportII(model, X_train, y_train, X_test, y_test, *args):
    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test)
    print("train set：")
    print(classification_report(y_train, y_train_predicted))
    print("___________________________________________________")
    print("test set：")
    print(classification_report(y_test, y_test_predicted))
    print("Accuracy: %.2f"% accuracy_score(test_label1, test_pred1))
    print("F-score: %.2f"% f1_score(test_label1, test_pred1, average='macro'))
    print("Kappa: %.2f"% cohen_kappa_score(test_label1, test_pred1))  # 0.8
    return (y_train_predicted, y_test_predicted)

y_train_predicted, y_test_predicted=show_reportII(rf1,train_data1,train_label1,test_data1,test_label1)


# In[ ]:


# Sequential Feature Selector
sfs_feature = 10
sfs_direction = 'backward'
sfs_cv = 2
sfs_scoring = 'r2'
################################ Functions #############################################################
def sfs_feature_selection(data, train_target,sfs_feature,sfs_direction,sfs_cv,sfs_scoring):
    
    #Inputs
    # data - Input feature data 
    # train_target - Target variable training data
    # sfs_feature - no. of features to select
    # sfs_direction -  forward and backward selection
    # sfs_cv - cross-validation splitting strategy
    # sfs_scoring - CV performance scoring metric
    rf = RandomForestClassifier()
    sfs=SequentialFeatureSelector(estimator = rf,
                                  n_features_to_select=sfs_feature,   
                                  direction = sfs_direction,
#                                  k_features= (0,data.shape[1]+1),
#                                  direction=sfs_direction,
                                  cv = sfs_cv,
                                  scoring = sfs_scoring)
    sfs.fit(data,train_target)
    print(1)
    sfs.get_support()
    sfs_df = pd.DataFrame(columns = ['Feature', 'SFS_filter'])
    sfs_df['Feature'] = data.columns
    sfs_df['SFS_filter'] = sfs.get_support().tolist()
    sfs_df_v2 = sfs_df[sfs_df['SFS_filter']==True]
    sfs_top_features = sfs_df_v2['Feature'].tolist()
    print(sfs_top_features)

    return sfs_top_features_df,sfs
################################ Calculate RFECV #############################################################
sfs_top_features_df,sfs = sfs_feature_selection(train_data1,train_label1,sfs_feature,sfs_direction,sfs_cv,sfs_scoring)
sfs_top_features_df.head(n=20)


# In[ ]:


#  Recursive feature elimination
# 1.read data
data = pd.read_csv(r"WFs1.csv")
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
print(X.shape)

# 2. standardise
scaler = StandardScaler()
X_train = scaler.fit_transform(X)

# 2.built model
RFC_ = RFC()                               
c = RFC_.fit(X, Y).feature_importances_    
print("importance：")
print(c)

# 3. CV-based RFE
selector = RFECV(RFC_, step=1, cv=10)       # select optimal feature
selector = selector.fit(X, Y)
X_wrapper = selector.transform(X)          # optimal feature
score =cross_val_score(RFC_ , X_wrapper, Y, cv=10).mean()   
print(score)
print("Optimal number and order:")
print(selector.support_)                                   
print(selector.n_features_)                                 
print(selector.ranking_)                                   


# 4.Recursive feature elimination
selector1 = RFE(RFC_, n_features_to_select=3, step=1).fit(X, Y)       # n_features_to_select suggests the final feature result
selector1.support_.sum()
print(selector1.ranking_)                                            
print(selector1.n_features_)                                      
X_wrapper1 = selector1.transform(X)                                  
score =cross_val_score(RFC_, X_wrapper1, Y, cv=9).mean()
print(score)

# 5.result
score = []                                                         
for i in range(1, 8, 1):
    X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(X, Y)    
    once = cross_val_score(RFC_, X_wrapper, Y, cv=9).mean()                      
    score.append(once)                                                           
print(max(score), (score.index(max(score))*1)+1)                                
print(score)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 8, 1), score)
plt.xticks(range(1, 8, 1))
plt.show()




# In[186]:


# random forest feature selection
TGMaxAcc=0

TGMeanAcc=[]
TGTLMaxAcc=[]
TGTLMinAcc=[]
FGSort=pd.DataFrame()

df_TLAcc_all=pd.DataFrame()
S=pd.DataFrame()
S=train_data1
S = pd.DataFrame(S, dtype=np.float64)
#print(S)
label=train_label1
label=label.values.tolist()
x_columns = S.columns[0:]
x_columns=x_columns.values.tolist()
x_columns
model=[]

unimportant_ftname=[]
unimportant_ftname_all=[]


for i in range(0,105, 1):
    # split the data into train, test set respectively
    print(i,"runs")
#    print(i)
#    a=0
    FSort=pd.DataFrame()
    S=S.values.tolist()
    

#    print(train_data1)
#    print(train_label1)
    TLMaxAcc=0
    TLMeanAcc=0
    TLAcc=[]

    s=KFold(n_splits=25)
#    print(s)
    j=-1
    for train,valid in s.split(S):
        j=j+1
        print(j,"fold")
       # print(train,valid)
        
        train_set=[]
        train_set_label=[]
        for p in range(len(train)):
          #  print(train[p])   
            valid_set=[]
            valid_set_label=[]       

            train_index=train[p]
            train_set.append(S[train_index])
            train_set_label.append(label[train_index])

        for q in range(len(valid)):
            valid_index=valid[q]
            valid_set.append(S[valid_index])
            valid_set_label.append(label[valid_index])            
        
        df_train_set = pd.DataFrame(train_set) 
        df_valid_set = pd.DataFrame(valid_set)       


        
        RFC1 = RFC(n_estimators=570,random_state=1)
        RFC1.fit(train_set, train_set_label)
        valid_pred1 = RFC1.predict(valid_set)
      #  print(train_set_label)
        TLAcc1=accuracy_score(valid_set_label, valid_pred1)
        TLAcc.append(TLAcc1)
        print("TLAcc:",TLAcc)
        
        if j==24: 
            np_TLAcc=np.array(TLAcc)
            TLMeanAcc=np_TLAcc.mean()
            TGMeanAcc.append(TLMeanAcc)
            
            TGTLMaxAcc.append(TLMaxAcc)
            TLMinAcc=np_TLAcc.min()
            TGTLMinAcc.append(TLMinAcc)
            
            df_TLAcc=pd.DataFrame(TLAcc)
            df_TLAcc_all[i]=df_TLAcc
        
        if TLMaxAcc<=TLAcc[j]:
            TLMaxAcc=TLAcc[j]
            print("TLMaxAcc",TLMaxAcc)
          
            import_level = RFC1.feature_importances_ 
        #    x_columns = train_data1.columns[0:]
  
            index = np.argsort(import_level)[::-1]
            
            FSort=import_level
           # print("FSORT:",FSort)

            for f in range(len(index)):
                        #    print(index[f])
                #print("%2d %-*s %f" % (f + 1, 30, x_columns[index[f]], import_level[index[f]]))
                          #  print("%2d %-*s %f" % (f + 1,30, index[f], import_level[index[f]]))
            
                if import_level[index[f]]==import_level.min():
                    idx_feature=index[f]
                    print("the most unimportant variable of {j} fold:",idx_feature)
                    print("the most unimportant variable name of {j} fold:",x_columns[idx_feature])
                                             
    unimportant_ftname.append(x_columns[idx_feature])
            
    print(' unimportant_ftname:',unimportant_ftname)
            
    if TGMaxAcc<=TLMeanAcc:
        TGMaxAcc=TLMeanAcc
        print("TLMeanAcc:",TLMeanAcc)
        FGSort=FSort
        unimportant_ftname_all=unimportant_ftname


    print("the deleted variable:",x_columns[idx_feature])
    del x_columns[idx_feature]
    S=pd.DataFrame(S)
    S=S.drop(S.columns[idx_feature],axis = 1)
    
    print('***************************************************************************')
    
print("TGMaxAcc:",TGMaxAcc)
print("FGSORT:",FGSort)
print("TGTLMeanAcc:",TGMeanAcc)
print("TGTLMaxAcc:",TGTLMaxAcc)
print("TGTLMinAcc:",TGTLMinAcc) 
df_TLAcc_all


# In[89]:


len(FGSort)
len(unimportant_ftname_all)
df_TLAcc_all


# In[91]:


# Specify the name of the excel file
file_name = 'VI_optimizev2.xlsx'
  
# saving the excelsheet
df_TLAcc_all_T.to_excel(file_name)


# In[95]:


df_TLAcc_all_T=df_TLAcc_all.transpose()
df_TLAcc_all_T['mean']=df_TLAcc_all_T.mean(axis=1)
df_TLAcc_all_T['max']=df_TLAcc_all_T.iloc[:,0:9].max(axis=1)
df_TLAcc_all_T['min']=df_TLAcc_all_T.iloc[:,0:9].min(axis=1)
#df_TLAcc_all_T=df_TLAcc_all_T.sort_values(by=['mean'],na_position='first')
df_TLAcc_all_T


# In[12]:


df2=pd.read_csv('MAP/VI_optimizev2.csv')
df2


# In[49]:


fig, ax = plt.subplots(figsize = (12,5))
#y1=df_TLAcc_all_T.iloc[:,26]
#y2=df_TLAcc_all_T.iloc[:,24]
#y3=df_TLAcc_all_T.iloc[:,25]

y1=df2.iloc[:,16]
y2=df2.iloc[:,15]
y3=df2.iloc[:,14]

ax.fill_between(np.arange(1,106,1), y1, y2, color ='lightsalmon',alpha=.5, linewidth=0)
#plt.plot(range(1,200,5),y3,color='blue',alpha=0.7,marker='o',markersize='4')
plt.plot(np.arange(1,106,1),y3,color='darkorange',alpha=0.7)
#plt.legend()
plt.xlabel("Number of Selected Variables ",size=15,family = 'Times New Roman')
plt.ylabel("Accuracy",size=15,family = 'Times New Roman')

plt.tick_params(labelsize=5)
plt.xticks(np.arange(0, 115, step=10),size = 12,family = 'Times New Roman')
plt.yticks(np.arange(0.65, 1.05, step=0.05),size = 12,family = 'Times New Roman')


# In[420]:


name=pd.DataFrame(VI)
# Specify the name of the excel file
file_name = 'VI.xlsx'
  
# saving the excelsheet
FSort1.to_excel(file_name)


# In[5]:


df4=pd.read_excel('VI4.xlsx')

df4


# In[9]:


# variable importance plot
import seaborn as sns

plt.figure(figsize=(6,12))


sns.boxplot(data=df4,orient="h",color='skyblue', linewidth=1.2,notch=True, width=0.8)

plt.xlabel("Feature Importance",size=15,family = 'Times New Roman')
plt.ylabel("Predictor Variables",size=15,family = 'Times New Roman')

plt.xticks(size = 12,family = 'Times New Roman')
plt.yticks(size = 12,family = 'Times New Roman')

# plt.show()
plt.savefig('变量重要性.png',bbox_inches = 'tight',dpi=600)


# In[214]:


def smote(X:pd.DataFrame, y:pd.Series) -> List[pd.DataFrame]:
    y = pd.DataFrame(y)
#    k_neighbors_val = max(1, y.value_counts()[1] // 2)
#    sm = SMOTE(random_state=0,k_neighbors=k_neighbors_val)
    sm = SMOTE(random_state=0)
 #   X_cols = list(X.columns)
 #   y_cols = list(y.columns)
    X_cols = X.columns
    y_cols = y.columns
    x_resample, y_resample = sm.fit_resample(np.array(X), np.array(y).reshape(-1, 1))
    x_resample = pd.DataFrame(x_resample, columns=X_cols)
    y_resample = pd.DataFrame(y_resample, columns=y_cols)
    return [x_resample, y_resample]

train_data1,train_label1=smote(train_data1,train_label1)


# In[291]:


#CV-based selection of the number of trees

data1=pd.DataFrame()

for j in range(0,25):

    a=[]

    for i in range(0,1000,10): 
        rfc = RandomForestClassifier(n_estimators=i+1,random_state=j+1,n_jobs=-1)
        score = cross_val_score(rfc,train_data1,train_label1,cv=10).mean()
        print(i)
        print("score:",score)
        a.append(score)
        print(a)


        #s.iloc[:,0]=z
    a=pd.DataFrame(a)   

    data1[j]=a
    
data1


# In[293]:


data1


# In[297]:


# Specify the name of the excel file
file_name = 'tree_optimize200.xlsx'
  
# saving the excelsheet
data.to_excel(file_name)


# In[298]:


trees=pd.DataFrame()
trees['mean']=data1.mean(axis=1)
trees['max']=data1.max(axis=1)
trees['min']=data1.min(axis=1)
trees


# In[307]:


fig, ax = plt.subplots(figsize = (12,5))
y1=trees.iloc[:,1]
y2=trees.iloc[:,2]
y3=trees.iloc[:,0]
ax.fill_between(range(1,1000,10), y1, y2, color ='lightblue',alpha=.5, linewidth=0)
#plt.plot(range(1,200,5),y3,color='blue',alpha=0.7,marker='o',markersize='4')
plt.plot(range(1,1000,10),y3,color='darkblue',alpha=0.7)
#plt.legend()
plt.xlabel("Number of trees selected",size=15,family = 'Times New Roman')
plt.ylabel("Accuracy",size=15,family = 'Times New Roman')

plt.tick_params(labelsize=5)
plt.xticks(np.arange(0, 1050, step=50),size = 12,family = 'Times New Roman')
plt.yticks(np.arange(0.70, 0.90, step=0.05),size = 12,family = 'Times New Roman')


# In[314]:


trees['mean'].max()


# In[463]:


df_pred1 = pd.DataFrame(test_pred1)
df_pred1

# plot the confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
#creates a grid to plot on
f, ax = plt.subplots(figsize=(7, 5))
#plots confusion matrix
sns.heatmap(confusion_matrix(test_label1, df_pred1,labels = ['P','R','NV']),
            annot=True, 
            fmt="d", 
            linewidths=.5, 
            cmap="YlGnBu",
            xticklabels=['P','R','NV'], 
            yticklabels=['P','R','NV'])
#            xticklabels=True, 
#            yticklabels=True)

#ax.set_xticklabels(test_label, rotation=50, horizontalalignment='left', family='Times New Roman', fontsize=5)
#ax.set_yticklabels(test_label, rotation=0, family='Times New Roman', fontsize=5)

plt.title('Confusion Matrix for RF Prediction combined with S2' +'\n')

plt.show()



# In[318]:


# Create correlation matrix
corr_matrix = train_data1.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop2 = [idx for idx in upper.index if any(upper[idx] > 0.95)]
to_drop


# In[334]:


data_cor=train_data1
data_cor['Type']=train_label1
#data_cor = pd.concat([train_data1,train_label1], axis=1, ignore_index=True)
corr_matrix1 = data_cor.corr(method='pearson').abs()
corr_matrix1 


# In[332]:


# Specify the name of the excel file
file_name = 'corr.xlsx'
  
# saving the excelsheet
corr_matrix1.to_excel(file_name)


# In[ ]:


# filling plot
plt.plot(x, y1, linewidth=1, color="orange", marker="o",label="Mean value")
yTop = z1.values.tolist()
yBottom = w1.values.tolist()
plt.fill_between(x, yTop, yBottom ,color="lightgreen",label="Standard deviation")

plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,2,4,6,8])
plt.legend(["Mean value","Threshold"],loc="upper left")
plt.grid()  #
plt.title('Value for carbon sequestration')
for i in range(1):
        plt.text(x[i], y1[i],y1[i], fontsize=12, color="black", style="italic", weight="light", verticalalignment='center',horizontalalignment='right', rotation=90)
plt.show()


# In[ ]:


def regression_method(model):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    ResidualSquare = (result - y_test)**2     
    RSS = sum(ResidualSquare)  
#    MSE = np.mean(ResidualSquare)       
    num_regress = len(result)   
    print(f'n={num_regress}')
    print(f'R^2={score}')
#    print(f'MSE={MSE}')
    print(f'RSS={RSS}')
    MSE = mean_squared_error(y_test, result)    
#    MSE=np.sum(np.power((y_test.reshape(-1,1) - y_pred),2))/len(y_test)
    R2=1-MSE/np.var(y_test)
    RSS1=np.sum(np.square(result - y_test))
#    print(MSE)            
    print("MSE:",MSE)
    print("R2:", R2)
    print(f'RSS1={RSS1}')                          
###########line chart###########
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('RandomForestRegression R^2: %f'%score)
    plt.legend()       
    plt.show()
    return result


# In[ ]:


y_pred = regression_method(rf)


# In[ ]:


def scatter_plot(TureValues,PredictValues):

    xxx = [-0.5,1.5]
    yyy = [-0.5,1.5]
    plt.figure()
    plt.plot(xxx , yyy , c='0' , linewidth=1 , linestyle=':' , marker='.' , alpha=0.3)
    plt.scatter(TureValues , PredictValues , s=20 , c='r' , edgecolors='k' , marker='o' , alpha=0.8)
    plt.xlim((0,120))   
    plt.ylim((0,120))
    plt.title('RandomForestRegressionScatterPlot')
    plt.show()


# In[ ]:


#density plot map
from scipy import optimize
x=y_test.values.ravel()
y=y_pred

x2=np.linspace(-1000,1000)
y2=x2

C= round(r2_score(x,y),4)
rmse=round(np.sqrt(mean_squared_error(x,y)),3)

def f_1(x,A,B):
    return A*x+B

A1,B1=optimize.curve_fit(f_1,x,y)[0]
y3=A1*x+B1

fig, ax= plt.subplots(figsize=(7,5),dpi=200)
dian=plt.scatter(x,y,edgecolors=None,c='k',s=16,marker='s')

ax.plot(x2,y2,color='k',linewidth=1.5,linestyle='--')
ax.plot(x,y3,color='r',linewidth=1.5,linestyle='-')

fontdict1={'size':17,'color':'k','family':'Times New Roman'}

ax.set_xlabel('ALS 2020 Canopy Area (m\u00b2)',size = 15,family = 'Times New Roman')
ax.set_ylabel('Predicted 2020 Canopy Area (m\u00b2)',size = 15,family = 'Times New Roman')

#ax.grid(False)
ax.set_xlim((0,900))   
ax.set_ylim((0,900))
ax.set_xticks(np.arange(0,1000,step=100))
ax.set_yticks(np.arange(0,1000,step=100))

for spine in ['top','bottom','left','right']:
    ax.spines[spine].set_color('k')
ax.tick_params(left=True,bottom=True,direction='in',labelsize=14)

ax.text(50,800,r'$y=$'+str(round(A1,3))+'$x$'+'+'+str(round(B1,3)),fontdict=fontdict1)
ax.text(50,720,r'$R^2$='+str(round(C,3)),fontdict=fontdict1)
ax.text(50,640,r'$RMSE$='+str(rmse),fontdict=fontdict1)



plt.style.use('seaborn-darkgrid')
# Estimate the 2D histogramn
nbins = 150
H, xedges, yedges = np.histogram2d(x, y, bins=nbins)

# H needs to be rotated and flipped
H = np.rot90(H)
H = np.flipud(H)
# Mask zerosH
Hmasked = np.ma.masked_where(H==0,H) 
# Mask pixels with a value of zero

plt.pcolormesh(xedges, yedges, Hmasked, cmap=cm.get_cmap('jet'), vmin=0, vmax=5)


cbar = plt.colorbar(ax=ax,ticks=[0,1,2,3,4,5],drawedges=False)
cbar.ax.set_ylabel('Frequency',fontdict=fontdict1)
cbar.ax.set_title('Counts',fontdict=fontdict1,pad=8)
cbar.ax.tick_params(labelsize=12,direction='in')
cbar.ax.set_yticklabels(['0','1','2','3','4','>5'],family='Times New Roman')



plt.show()

