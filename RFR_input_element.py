#coding=utf-8
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

numberOfTrees = 100

sc = SimpleImputer()# 使用sklean将缺失值替换为特定值

# train = pd.read_csv('acsf_train.csv')
# #print(np.isnan(train).any())
# train.dropna(inplace=True) #检查缺失值

### load in data
### train ###
train = np.genfromtxt('acsf_train.csv', delimiter=',', encoding='utf-8')

#train = np.genfromtxt('all_carbon_train.csv', delimiter=',')
train_1=pd.Series(train[:, 0])
train_2=pd.Series(train[:, 1])
train_3=pd.Series(train[:, 2])
train_4=pd.Series(train[:, 3])
train_5=pd.Series(train[:, 4])
train_6=pd.Series(train[:, 5])
train_7=pd.Series(train[:, 6])
train_8=pd.Series(train[:, 7])
train_9=pd.Series(train[:, 8])
train_10=pd.Series(train[:, 9])
train_11=pd.Series(train[:, 10])
train_12=pd.Series(train[:, 11])
train_13=pd.Series(train[:, 12])
train_14=pd.Series(train[:, 13])
train_charge=pd.Series(train[:, 14])
#print(train_1)
dataset_train = pd.DataFrame({"train_1":train_1,"train_2":train_2,"train_3":train_3,"train_4":train_4,"train_5":train_5,"train_6":train_6,"train_7":train_7,"train_8":train_8,"train_9":train_9,"train_10":train_10,"train_11":train_11,"train_12":train_12,"train_13":train_13,"train_14":train_14,"train_charge":train_charge})
#print(dataset_train)
### test ###
test = np.genfromtxt('acsf_test.csv', delimiter=',', encoding='utf-8')
#test = np.genfromtxt('all_carbon_test.csv', delimiter=',')
test_1=pd.Series(test[:, 0])
test_2=pd.Series(test[:, 1])
test_3=pd.Series(test[:, 2])
test_4=pd.Series(test[:, 3])
test_5=pd.Series(test[:, 4])
test_6=pd.Series(test[:, 5])
test_7=pd.Series(test[:, 6])
test_8=pd.Series(test[:, 7])
test_9=pd.Series(test[:, 8])
test_10=pd.Series(test[:, 9])
test_11=pd.Series(test[:, 10])
test_12=pd.Series(test[:, 11])
test_13=pd.Series(test[:, 12])
test_14=pd.Series(test[:, 13])
test_charge=pd.Series(test[:, 14])


dataset_test = pd.DataFrame({"test_1":test_1,"test_2":test_2,"test_3":test_3,"test_4":test_4,"test_5":test_5,"test_6":test_6,"test_7":test_7,"test_8":test_8,"test_9":test_9,"test_10":test_10,"test_11":test_11,"test_12":test_12,"test_13":test_13,"test_14":test_14,"test_charge":test_charge})

### normalize data
train_X = dataset_train.drop('train_charge', axis=1)
test_X  = dataset_test.drop('test_charge', axis=1)
#print(test_X)

sc.fit(train_X)

train_X_normalized = pd.DataFrame(sc.fit_transform(train_X), columns=train_X.columns.values)
test_X_normalized = sc.transform(test_X)
#print(test_X_normalized)
train_Y = dataset_train['train_charge']
test_Y = dataset_test['test_charge']
#print(train_X_normalized)
# print(train_Y)
### RFR regressor fit ###
'''
random_state：指定模型随机状态，确保每次生成的模型是相同的
n_jods:进程个数（-1为用所有的CPU进行计算，默认为None，即为1）
最大特征数为所有特征数一半时为佳
最大深度：当特征较少或数据集较少时可选择默认
min_samples_split：当内部节点样本数少于该参数，不继续生长子树(>=2)
min_samples_leaf：当叶节点（决策树末端节点）样本数少于该参数，该层兄弟节点都被剪枝
'''
regressor = RandomForestRegressor(random_state=6, n_estimators=numberOfTrees, max_depth=None, max_features=7, min_samples_leaf=1, min_samples_split=2, bootstrap=False)# 最大特征数为所有特征数一半时为佳
regressor.fit(train_X_normalized, train_Y)
pred_Y = regressor.predict(test_X_normalized)

np.savetxt('acsf_test_charge.txt', test_Y)
np.savetxt('acsf_pred_charge.txt', pred_Y)


### calculate Pearson Correlation ###
def calcMean(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean, y_mean

def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean, 2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    print(p)
    return p


pred_test_charge_Pearson_carbon = calcPearson(test_Y, pred_Y)
filename = "acsf_pred_test_charge_Pearson.txt"
file=open(filename, 'w')
file.write('%f'%pred_test_charge_Pearson_carbon)
file.close

