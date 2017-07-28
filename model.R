rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)
library(glmnet)
library(caret)
library(mboost)
library(randomForest)
library(e1071)
library(ROSE)

###########################################################################################################################
##Logistic using original unbalanced data

data = read.csv("data5.csv")
dummies = read.csv("dummies.csv")
numericals = read.csv("numericals.csv")
factors = read.csv("factors.csv")

data$state_median_household_income <- as.numeric(data$state_median_household_income)
numericals$state_median_household_income <- as.numeric(numericals$state_median_household_income)

set.seed(3)
data <- data[sample(nrow(data), nrow(data)), ] #random permutation
rownames(data)<- c(1:nrow(data)) ##row names starting from 1

###########################################################################################################################

Nsimu = 10 #5 fold CV
Nobserv = nrow(data)
Nsub = round(Nobserv/Nsimu, digits = 0)

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
##Logistic using original unbalanced data
CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = train[names(train) != "loan_status"]
  x.test = test[names(test) != "loan_status"]
  
  xx = paste(names(x.train), collapse = " + ")
  f  = as.formula(paste("loan_status ~ ", xx))
  fit.logistic <- glm(formula = f, data=train, family=binomial())
  
  pr = predict(fit.logistic, newdata = test,  type="response") # predicted values
  
  test.pred = (pr > 0.5)
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  
  r2 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))
###########################################################################################################################
###########################################################################################################################
rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)
library(glmnet)
library(caret)
library(mboost)
library(randomForest)
library(e1071)
library(ROSE)
###########################################################################################################################
##SVM

data = read.csv("data5.csv")
dummies = read.csv("dummies.csv")
numericals = read.csv("numericals.csv")
factors = read.csv("factors.csv")

data$state_median_household_income <- as.numeric(data$state_median_household_income)
numericals$state_median_household_income <- as.numeric(numericals$state_median_household_income)

##convert outcome to factor
data$loan_status <- factor(ifelse(data$loan_status, 1, 0)) 


##############################################################################
table(data$loan_status)

#0    1 
#6548 314464 

prop.table(table(data$loan_status))

#0          1 
#0.02039799 0.97960201 

data_unbalanced <- data
##############################################################################
##Balance the data, undersample TRUE, oversample FALSE
##the function of ovun.sample only accepts factor variable as outcome for classification, need to change loan_status from logical to factor
data_balanced <- ovun.sample(loan_status ~ ., data = data_unbalanced, method = "both", p=0.5,  N=100000, seed = 3)$data

table(data_balanced$loan_status)
#1      0 
#149739 150261

prop.table(table(data_balanced$loan_status))
#1       0 
#0.49913 0.50087

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
data <- data_balanced
##convert outcome to logical
#data$loan_status <- data$loan_status == "1"


set.seed(3)
data <- data[sample(nrow(data), nrow(data)), ] #random permutation, ovun.sample sort the data based on outcome classes
rownames(data)<- c(1:nrow(data)) ##row names starting from 1

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

Nsimu = 10 #5 fold CV
Nobserv = nrow(data)
Nsub = round(Nobserv/Nsimu, digits = 0)

###########################################################################################################################
###########################################################################################################################
##SVM with Gaussian kernal

CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  y.train = train$loan_status
  y.test = test$loan_status
  ##svm only accepts factor as outcome variable for classification
  model <- svm(loan_status ~ ., data = train, kernel = "radial")
  
  pred <- predict(model, newdata = test)
  
  test.pred = predict(model, newdata = test)
  
  test.pred <- test.pred == "1"
  y.test <- y.test == "1"
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=y.test, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  r2 <- confusionMatrix(data=test.pred, reference=y.test, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))

###########################################################################################################################
rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)
library(glmnet)
library(caret)
library(mboost)
library(randomForest)
library(e1071)
library(ROSE)
################################################
##Logistic with Gradient Boosting

data = read.csv("data5.csv")
dummies = read.csv("dummies.csv")
numericals = read.csv("numericals.csv")
factors = read.csv("factors.csv")

data$state_median_household_income <- as.numeric(data$state_median_household_income)
numericals$state_median_household_income <- as.numeric(numericals$state_median_household_income)

##convert outcome to factor
data$loan_status <- factor(ifelse(data$loan_status, 1, 0)) 


data_unbalanced <- data
##############################################################################
##Balance the data, undersample TRUE, oversample FALSE
##the function of ovun.sample only accepts factor variable as outcome for classification, need to change loan_status from logical to factor
data_balanced <- ovun.sample(loan_status ~ ., data = data_unbalanced, method = "both", p=0.5,  N=100000, seed = 3)$data

###########################################################################################################################
###########################################################################################################################
data <- data_balanced
##convert outcome to logical
#data$loan_status <- data$loan_status == "1"


set.seed(3)
data <- data[sample(nrow(data), nrow(data)), ] #random permutation, ovun.sample sort the data based on outcome classes
rownames(data)<- c(1:nrow(data)) ##row names starting from 1

###########################################################################################################################
###########################################################################################################################

Nsimu = 10 #5 fold CV
Nobserv = nrow(data)
Nsub = round(Nobserv/Nsimu, digits = 0)

###########################################################################################################################
###########################################################################################################################
##Logistic with Gradient Boosting

CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = train[names(train) != "loan_status"]
  x.test = test[names(test) != "loan_status"]
  
  xxx = paste(names(numericals), collapse = " + ")
  design.matrix <- x.train[names(x.train) %in% names(dummies)]
  
  f2 = as.formula(paste("loan_status ~ ", xxx, "+", "bols(design.matrix, df = 4, intercept = TRUE)"))
  ##gamboost only accepts factor as outcome variable for classification
  model = gamboost(formula = f2, family = Binomial(link = "logit"), data = train, control = boost_control(mstop = 1000))
  
  set.seed(3)
  cv5f = cv(model.weights(model), type = "kfold", B = 5)
  cvm = cvrisk(model, folds = cv5f, papply = mclapply)
  #cvm <- cvrisk(model)
  model = model[mstop(cvm)]
  
  pr = predict(model, newdata = test, type = "response")
  
  test.pred = (pr > 0.5)
  
  y.test <- y.test == "1"
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=y.test, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  r2 <- confusionMatrix(data=test.pred, reference=y.test, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))
#######################################################################################################################

rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)
library(glmnet)
library(caret)
library(mboost)
library(randomForest)
library(e1071)
library(ROSE)
###########################################################################################################################
##Logistic with Ridge, LASSO, and just logistic

data = read.csv("data5.csv")
dummies = read.csv("dummies.csv")
numericals = read.csv("numericals.csv")
factors = read.csv("factors.csv")

data$state_median_household_income <- as.numeric(data$state_median_household_income)
numericals$state_median_household_income <- as.numeric(numericals$state_median_household_income)

##convert outcome to factor
data$loan_status <- factor(ifelse(data$loan_status, 1, 0)) 


data_unbalanced <- data
##############################################################################
##Balance the data, undersample TRUE, oversample FALSE
##the function of ovun.sample only accepts factor variable as outcome for classification, need to change loan_status from logical to factor
data_balanced <- ovun.sample(loan_status ~ ., data = data_unbalanced, method = "both", p=0.5,  N=100000, seed = 3)$data


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
data <- data_balanced
##convert outcome to logical
data$loan_status <- data$loan_status == "1"


set.seed(3)
data <- data[sample(nrow(data), nrow(data)), ] #random permutation, ovun.sample sort the data based on outcome classes
rownames(data)<- c(1:nrow(data)) ##row names starting from 1

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

Nsimu = 10 #5 fold CV
Nobserv = nrow(data)
Nsub = round(Nobserv/Nsimu, digits = 0)

###########################################################################################################################
###########################################################################################################################
##Logistic with Ridge
CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = as.matrix(train[names(train) != "loan_status"])
  x.test = as.matrix(test[names(test) != "loan_status"])
  
  #fit.lasso <- glmnet(x.train, y.train, family="binomial", alpha=1)
  fit.ridge <- glmnet(x.train, y.train, family="binomial", alpha=0)
  
  pr = predict(fit.ridge, newx = x.test, type = "response", s = 0.005)
  
  test.pred = (pr > 0.5)
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  
  r2 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))


###########################################################################################################################
###########################################################################################################################
##Logistic with LASSO
CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = as.matrix(train[names(train) != "loan_status"])
  x.test = as.matrix(test[names(test) != "loan_status"])
  
  fit.lasso <- glmnet(x.train, y.train, family="binomial", alpha=1)
  #fit.ridge <- glmnet(x.train, y.train, family="binomial", alpha=0)
  
  pr = predict(fit.lasso, newx = x.test, type = "response", s=0.005)
  
  test.pred = (pr > 0.5)
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  
  r2 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))


###########################################################################################################################
###########################################################################################################################
##Logistic
CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
   
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = train[names(train) != "loan_status"]
  x.test = test[names(test) != "loan_status"]
  
  xx = paste(names(x.train), collapse = " + ")
  f  = as.formula(paste("loan_status ~ ", xx))
  fit.logistic <- glm(formula = f, data=train, family=binomial())
  
  pr = predict(fit.logistic, newdata = test,  type="response") # predicted values
  
  test.pred = (pr > 0.5)
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  
  r2 <- confusionMatrix(data=test.pred, reference=test$loan_status, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))

###########################################################################################################################
###########################################################################################################################
rm(list = ls())
setwd("C:/Users/xyin/Desktop/Manor Resource/test-loan-data")
getwd()

library(data.table)
library(stringr)
library(glmnet)
library(caret)
library(mboost)
library(randomForest)
library(e1071)
library(ROSE)
###########################################################################################################################
##Random Forest
data = read.csv("data5.csv")
numericals = read.csv("numericals.csv")
factors = read.csv("factors.csv")

data$state_median_household_income <- as.numeric(data$state_median_household_income)
numericals$state_median_household_income <- as.numeric(numericals$state_median_household_income)

##convert outcome to factor
loan_status <- factor(ifelse(data$loan_status, 1, 0)) 

##redefine data as factors + numericals
data = data.frame(factors, numericals, loan_status)

data_unbalanced <- data
##############################################################################
##Balance the data, undersample TRUE, oversample FALSE
##the function of ovun.sample only accepts factor variable as outcome for classification, need to change loan_status from logical to factor
data_balanced <- ovun.sample(loan_status ~ ., data = data_unbalanced, method = "both", p=0.5,  N=100000, seed = 3)$data

###########################################################################################################################
###########################################################################################################################
data <- data_balanced
##convert outcome to logical
#data$loan_status <- data$loan_status == "1"

set.seed(3)
data <- data[sample(nrow(data), nrow(data)), ] #random permutation, ovun.sample sort the data based on outcome classes
rownames(data)<- c(1:nrow(data)) ##row names starting from 1

###########################################################################################################################
###########################################################################################################################

Nsimu = 10 #5 fold CV
Nobserv = nrow(data)
Nsub = round(Nobserv/Nsimu, digits = 0)

###########################################################################################################################
###########################################################################################################################
##Random Forests
CM <- list()
accuracy <- list()
prf <- list()

for (isimu in 1:Nsimu) {
  
  test <- data[((isimu-1)*Nsub+1):(isimu*Nsub), ]
  rownames(test)<- c(1:nrow(test)) ##row names starting from 1
  
  train <- data[setdiff(1:Nobserv,((isimu-1)*Nsub+1):(isimu*Nsub)), ]
  rownames(train)<- c(1:nrow(train)) ##row names starting from 1
  
  y.train = train$loan_status
  y.test = test$loan_status
  
  x.train = train[names(train) != "loan_status"]
  x.test = train[names(test) != "loan_status"]
  
  xx = paste(names(x.train), collapse = " + ")
  f  = as.formula(paste("loan_status ~ ", xx))
  ##randomForest only accepts factor as outcome variable for classification
  test.rf <- randomForest(formula = f, data = train, ntree = 500, importance=FALSE, proximity=FALSE, na.action=na.exclude)
  
  test.pred <- predict(test.rf, test)
  test.pred <- unname(test.pred, force = FALSE)
  
  test.pred <- test.pred == "1"
  y.test <- y.test == "1"
  ##confusionMatrix only accepts logical variables as inputs, need to covert factor outcome to logical
  r1 <- confusionMatrix(data=test.pred, reference=y.test, positive = "TRUE",
                        mode = "everything")
  accuracy[[isimu]] <- r1[3]$overall[1]
  output <- matrix(0,3,2)
  colnames(output) <- c("FALSE","TRUE")
  rownames(output) <- c("presicion","recall","F-score")
  output[3,2] <- r1[4]$byClass[7]
  output[1,2] <- r1[4]$byClass[5]
  output[2,2] <- r1[4]$byClass[6]
  r2 <- confusionMatrix(data=test.pred, reference=y.test, positive = "FALSE",
                        mode = "everything")
  output[3,1] <- r2[4]$byClass[7]
  output[1,1] <- r2[4]$byClass[5]
  output[2,1] <- r2[4]$byClass[6]
  prf[[isimu]] <- output
  
  cm <- table(observed = test$loan_status, predicted = test.pred)
  CM[[isimu]] <- cm
}

print(apply(simplify2array(prf), 1:2, mean))
print(apply(simplify2array(CM), 1:2, mean))
print(mean(simplify2array(accuracy)))