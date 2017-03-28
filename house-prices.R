##Read in the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

##Check for Missing Values
colSums(is.na(train))
colSums(is.na(test))

#Impute Missing Values.
#We use the missForest package to impute missing values.
#missForest runs a randomForest on each variable using the observed part and predicts the na values.
#We need to make sure our variables are not of type "char" before running missForest.

library(missForest)

fulltrain.mis <- missForest(train,maxiter = 5,ntree = 250)
fulltest.mis <- missForest(test,maxiter = 5,ntree = 250)
fulltrain <- fulltrain.mis$ximp[,1:80]
fulltest <- fulltest.mis$ximp
fullnona <- rbind(fulltrain,fulltest)

#Convert SalePrice into log scale to make it a unifrom distribution.
#Regression models perform well when the data distribution is uniform. 
fulltrain$SalePrice <- log(fulltrain$SalePrice+1)

#Explore the Skewness of various features of the dataset.
library(e1071)

classes <- lapply(fullnona,function(x) class(x))
numeric_feats <- names(classes[classes=="integer" | classes=="numeric"])
factor_feats <- names(classes[classes=="factor"| classes=="character"])

skewed_feats <- sapply(numeric_feats, function(x) skewness(fullnona[[x]]))
skewed_feats <- skewed_feats[abs(skewed_feats) > .75]

##Taking log transformations of features with Skewness more than .75
for (x in names(skewed_feats)) {fullnona[[x]] <- log(fullnona[[x]]+1)}


fullnonanum <- data.frame(lapply(fullnona,as.numeric))

##split into test and train
trainnum <- fullnonanum[1:1460,]
testnum <- fullnonanum[1461:2919,]

##Using Caret package for Cross Validation.
library(caret)
tr.control <- trainControl(method="repeatedcv", number = 5,repeats = 5)

#Ridge regression model

lambdas <- seq(1,0,-.001)

set.seed(123)
ridge_model <- train(SalePrice~., data=trainnum,method="glmnet",metric="RMSE",
                     maximize=FALSE,trControl=tr.control,
                     tuneGrid=expand.grid(alpha=0,lambda=lambdas))

ridge_preds <- exp(predict(ridge_model,newdata = testnum))-1

write.csv(data.frame(Id=test$Id,SalePrice=ridge_preds),"ridge_preds.csv",row.names = F)

##ridge_preds scored 0.13428 on Kaggle Public Leaderboard.


#Lasso Regression Model

set.seed(123)
lasso_model <- train(SalePrice~., data=trainnum,method="glmnet",metric="RMSE",
                      maximize=FALSE,trControl=tr.control,
                      tuneGrid=expand.grid(alpha=1,lambda=c(1,0.1,0.05,0.01,seq(0.009,0.001,-0.001), 0.00075,0.0005,0.0001)))

lassopreds <- exp(predict(lasso_model,newdata = testnum))-1

write.csv(data.frame(Id=test$Id,SalePrice=lassopreds),"lasso_preds.csv",row.names = F)

##lasso_preds scored 0.12991 on Kaggle Public Leaderboard.

