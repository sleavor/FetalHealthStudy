#libraries
library(tidyverse)
library(ggplot2)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(rsample)

#Data
df = read.csv("D:\\Documents\\UT-Austin\\AI & Public Policy\\Final Project\\Fetal_health\\fetal_health.csv")

#Create new vars
df = df %>% mutate( mean_sub_base = baseline.value - histogram_mean,
                    med_sub_base = baseline.value - histogram_median,
                    hist_t = histogram_mean / ifelse(histogram_variance != 0, histogram_variance, 1)
  )

#Train/test split
x = initial_split(df, prop=0.8)
train = training(x)
test = testing(x)

#Get col names
list_names <- colnames(df)[colnames(df) != "fetal_health"]

#make x/y train
x_train = train[,list_names]
y_train = train[,'fetal_health']

x_test = test[,list_names]
y_test = test[,'fetal_health']

xgb_train = xgb.DMatrix(data = data.matrix(x_train), label = y_train)
xgb_test = xgb.DMatrix(data = data.matrix(x_test), label = y_test)

watchlist = list(train=xgb_train, test=xgb_test)

#Model training
model = xgb.train(data = xgb_train, max.depth = 5, watchlist=watchlist, nrounds = 70)
#Final model based off test-rmse
final = xgboost(data = xgb_train, max.depth = 5, nrounds = 28, verbose = 0)

# predict values in test set
y_pred <- predict(final, data.matrix(x_test))

#Notes: Use classifier instead of regressor?
#Classify
y_pred[(y_pred>3)] = 3
y_pred[(y_pred<1)] = 1
y_pred = round(y_pred)
y_error = y_test - y_pred

absolute_error = sum(abs(y_error))

#Make confusion matrix (fails)
conf_mat = confusionMatrix(as.factor(y_pred), as.factor(y_test))
print(conf_mat)

importance_matrix = xgb.importance(colnames(xgb_train), model = final)
print(importance_matrix)
xgb.plot.importance(importance_matrix[1:15,])
