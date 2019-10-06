import pandas as pd
import numpy as np
import os
# step1 
# 目标：获取前10天的标准差
# 输入：单只股票的数据集
# 输出：单只股票前10的标准差

#实现v1.0 效率较低
def chang_rank():

    path = '/Users/rui/PycharmProjects/gix-2/df_seperate'
    os.chdir(path)
    file_chdir = os.getcwd()
    after_sort = pd.DataFrame()

    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            print(file)
            if (file != '.DS_Store'):
                df = pd.read_csv(file)
                print (df.shape)
                df['deretnd_value'] = 0
                for i in range(10,df.shape[0]):
                    #print(i)
                    list = [df.iloc[i-1,12],df.iloc[i-2,12],df.iloc[i-3,12],df.iloc[i-4,12],df.iloc[i-5,12],df.iloc[i-6,12],\
                                    df.iloc[i-7, 12],df.iloc[i-8,12],df.iloc[i-9,12],df.iloc[i-10,12]]
                    df.iloc[i,22] = np.std(list)
                after_sort = after_sort.append(df)
                    #df.to_csv(file)
                    #for index, row in df.iterrows():
                     #   after_sort.append(row)

    #result = pd.DataFrame(after_sort)
    #print (result)
    #result.to_csv('change_rank.csv')
    print('end')
    after_sort.to_csv('change_rank.csv')
    print (after_sort.head())

chang_rank()


#实现v1.2 直接更新旧数据集，效率稍微高一点
def chang_rank():

    path = '/Users/rui/PycharmProjects/gix-2/df_new'
    os.chdir(path)
    file_chdir = os.getcwd()
    after_sort = pd.DataFrame()

    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            print(file)
            if (file != '.DS_Store'):
                df = pd.read_csv(file)
                print (df.shape)
                df['deretnd_value'] = 0
                for i in range(10,df.shape[0]):
                    #print(i)
                    list = [df.iloc[i-1,12],df.iloc[i-2,12],df.iloc[i-3,12],df.iloc[i-4,12],df.iloc[i-5,12],df.iloc[i-6,12],\
                                    df.iloc[i-7, 12],df.iloc[i-8,12],df.iloc[i-9,12],df.iloc[i-10,12]]
                    df.iloc[i,22] = np.std(list)
                #after_sort = after_sort.append(df)
                df.to_csv(file)
                    #for index, row in df.iterrows():
                     #   after_sort.append(row)

    #result = pd.DataFrame(after_sort)
    #print (result)
    #result.to_csv('change_rank.csv')
    print('end')


chang_rank()
#然后将分别存的数据合在一起
import pandas as pd
import os

path = '/Users/rui/PycharmProjects/gix-2/df_new'
os.chdir(path)
file_chdir = os.getcwd()
after_sort = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        print(file)
        if (file != '.DS_Store'):
            df = pd.read_csv(file)
            print(df.shape)
            for index, row in df.iterrows():
                 after_sort.append(row)

result = pd.DataFrame(after_sort)

result.to_csv('result_all_data.csv')



# step2: rank 
# rank函数
def v2():

    df = pd.read_csv('/Users/rui/PycharmProjects/gix-2/result_all_data.csv')

    df = df[df['Trdsta'] == 1] #正常交易数量
    df['Trddt'] = pd.to_datetime(df['Trddt'])
    df = df.sort_values('Trddt')
    #df = df[df['Trddt'] > '2016-01-01'] #16年以后的数据
     #大排名，然后groupby之后进行标准化
    #df = df.sort_values('Dretwd')

    print (df.shape)
    print(df)


    #求每日排名
    begin = datetime.date(2005,1,4)
    end = datetime.date(2009,12,31)

    after_sort = []

    for i in range((end - begin).days +1):
        day = begin + datetime.timedelta(days = i)
        df_daily = df[df['Trddt'].isin([str(day)])]
        #print ('df_daily',df_daily)
        df_daily = df_daily.sort_values('Dretwd')
        rank = [i for i in range(0, df_daily.shape[0])]
        df_daily.insert(0, 'Rank', rank) # 排名

        for index, row in df_daily.iterrows():
            after_sort.append(row)


    after_data = pd.DataFrame(after_sort)
    after_data.to_csv('after_sort_v2.csv',index=0)
    print (after_data.head(100))
    
# 把rank 之后的数据集分开
def get_stk_list(df):
    group_df = df.groupby('Stkcd').sum()

    index = group_df.index.values.tolist()  # 获取股票列表
    return (index)

def seperate(df):
    stk_index = get_stk_list(df)
    for i in stk_index:
        df_separate = df[df['Stkcd'].isin([i])]
        #print (df_separate)
        df_separate.to_csv('df_seperate/%s.csv' % i)
df = pd.read_csv('/Users/rui/PycharmProjects/gix-2/after_sort_v2.csv')
seperate(df)

step 3: 使用单个股票的数据，生成101alpha 

step4: 使用svm预测单个股票
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd



def stk_train_predict(X,y,i,j):
    save_result_score = []
    save_result_mse = []

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


    ss_x = StandardScaler()
    ss_y = StandardScaler()

    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)
    y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = ss_y.transform(y_test.values.reshape(-1, 1))

    print (type(y_test))

    linear_svr = SVR(kernel='linear')

    model = linear_svr.fit(x_train, y_train.ravel())
    linear_svr_predict = model.predict(x_test)

    ss = model.support_vectors_
    print (ss_y)
    svs = pd.DataFrame(model.support_vectors_)

    svs.to_csv('/Users/rui/PycharmProjects/gix-2/1_svs.csv')

    #print('The value of default measurement of linear SVR is', linear_svr.score(x_test, y_test))
    #print('R-squared value of linear SVR is', r2_score(y_test, linear_svr_predict))
    print('The mean squared error of linear SVR is',
         mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))
    print (ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict))
    print('The mean absolute error of linear SVR is',
          mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))

    #save_result_score.append(linear_svr.score(x_test, y_test))
    #save_result_mse.append(mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))

    #df_mse = pd.DataFrame(save_result_mse)
    #print (df_mse)

    #df_score = pd.DataFrame(save_result_score)
    #print (df_score)


#df = pd.read_csv('/Users/rui/PycharmProjects/gix-2/df_seperate/1.csv', index_col=0)
df = pd.read_csv('/Users/rui/PycharmProjects/gix-2/alpha_101_result.csv', index_col=0)
print (df.info())
df = df.fillna(0)
del df['Trddt']
del df['Capchgdt']
#del df['Unnamed: 0']
del df['Unnamed: 0.1']
del df['Unnamed: 0.1.1']
del df['Unnamed: 0.1.1.1']
del df['Unnamed: 0.1.1.1.1']
del df['Unnamed: 0.1.1.1.1.1']
del df['alpha061']
del df['alpha075']
del df['alpha095']
change = df['Rank']
del df['Rank']
df['Rank'] = change
print (df.head())

df.to_csv('/Users/rui/PycharmProjects/gix-2/1_preprocess.csv')
X = df.ix[:, :-1]
y = df.ix[:, -1]
print (y)



stk_train_predict(X, y, 'profile_mse', 'profile_score')

# setp4： R 语言，使用sv 生成规则
library(kernlab)
library(randomForest)
#library(C50)
library(inTrees)
library(e1071)
library(foreach)
library(parallel)
library(doParallel)
library(iterators)

select_rui <-function (ruleMetric, X, target,i)
{
  ruleI = sapply(ruleMetric[, "condition"], rule2Table, X, 
                 target)
  coefReg <- 0.95 - 0.01 * as.numeric(ruleMetric[, "len"])/max(as.numeric(ruleMetric[, 
                                                                                     "len"]))
  #coefReg <- 0.95
  #rf <- RRF(ruleI, as.factor(target), flagReg = 1, coefReg = coefReg, 
  #mtry = (ncol(ruleI) * 1/2), ntree = 50, maxnodes = 10, 
  #replace = FALSE)
  #rf <- randomForest(ruleI, as.factor(target))
  cl <- makeCluster(4)
  registerDoParallel(cl)
  rf <- foreach(ntree=rep(25, 4), 
                .combine='c',.packages='randomForest') %dopar% 
    randomForest(ruleI, as.factor(target), ntree=ntree)
  stopCluster(cl)
  imp <- rf$importance/max(rf$importance)
  #sort(imp,decreasing = TRUE)
  
  #feaSet <- which(imp >= imp[length(imp)*0.5])
  feaSet <- which(imp >= 0)
  #feaSet <- which(imp > 0.005)
  
  ruleSetPrunedRRF <- cbind(ruleMetric[feaSet, , drop = FALSE], 
                            impRRF = imp[feaSet])
  ix = order(as.numeric(ruleSetPrunedRRF[, "impRRF"]), -as.numeric(ruleSetPrunedRRF[, 
                                                                                    "err"]), -as.numeric(ruleSetPrunedRRF[, "len"]), decreasing = TRUE)
  
  ruleSelect <- ruleSetPrunedRRF[ix, , drop = FALSE]
  ruleSelect <- ruleSelect[ 0:round(nrow(ruleSelect)*i),]
  return(ruleSelect)
}


readTrees <- function (rf) 
{
  lTree <- NULL
  lTree$ntree <- rf$ntree
  lTree$list <- vector("list", rf$ntree)
  for (i in 1:lTree$ntree) {
    lTree$list[[i]] <- getTree(rf, i, labelVar = FALSE)
  }
  return(lTree)
}

#data <- read.csv(file = "/Users/rui/PycharmProjects/gix-2/2501_svs.csv")
#data <- read.csv(file ='/Users/rui/PycharmProjects/gix-2/1_svs.csv')

#data <- read.csv(file = "/Users/rui/PycharmProjects/gix-2/2501_preprocess.csv")
data <- read.csv(file = "/Users/rui/PycharmProjects/gix-2/1_preprocess.csv")

pp <- length(data[1,])
data.x <- as.matrix(data[,-c((pp-1), pp)])
print (data.x)

data.y <- data$Rank
print (data.y)

l <- length(data.y)
sub <- sample(1:l, l, replace = FALSE, set.seed(100))
ISub <- as.integer(length(sub)/10)
indd <- 1
temp_sub <- sub[(ISub*(indd-1)+1):(ISub*indd)]
t_sub <- sub[-c((ISub*(indd-1)+1):(ISub*indd))]
data.x <- data.x[t_sub, ]
data.y <- data.y[t_sub]


set.seed(100)
i.sigma <- 0.1
i.cost <- 1

cv <- 10
# split the data into training dataset and test set



l <- length(data.y)
sub <- sample(1:l, l, replace = FALSE, set.seed(100))
ISub <- as.integer(length(sub)/cv)
## 5???
for(indd in 1:1){
  temp_sub <- sub[(ISub*(indd-1)+1):(ISub*indd)]
  t_sub <- sub[-c((ISub*(indd-1)+1):(ISub*indd))]
  reg.x <- data.x[temp_sub, ]
  reg.y <- data.y[temp_sub]
  x <- data.x[t_sub, ]
  y <- data.y[t_sub]

  train_data <- as.data.frame(cbind(x,y))
  svm_rf <- randomForest(y~.,  data = train_data, ntree = 100, mtry = 5)
  pred.svm_rf <- predict(svm_rf, reg.x)


  treeList <- readTrees(svm_rf)
  ruleExec <- extractRules(treeList, x)
  # ruleExec <- unique(ruleExec)
  ruleMetric <- getRuleMetric(ruleExec,x,y) # measure rules
  #ruleMetric <- pruneRule(ruleMetric,x,y) # prune each rule
  ruleMetric <- select_rui(ruleMetric,x,y,0.1) # rule selection
  learner <- buildLearner(ruleMetric,x,y, minFreq = 0)
  pred.irrf <- applyLearner(learner,reg.x)
  print(pred.irrf)
  #print (caculate(reg.y , pred.irrf))
  print (length(learner[,1]))
  print(learner)
  
  read <- presentRules(learner,colnames(x))
  print(read)
}







