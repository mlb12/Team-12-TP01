
## Chapter 6 - Evaluating Test Error of Model Selection Methods ##

rm(list = ls())
## Load necessary packages and data set
library(ISLR)  #dataset
library(caret) #divide data
library(mltools)  #mse

## Load College data set into a variable
collegedata <- College

## Explore the data set
str(collegedata)
summary(collegedata)
head(collegedata)



## 1. Split data into training and testing set
set.seed(1)
divideData <- createDataPartition(collegedata$Apps, p = .5, list = FALSE)
train <- collegedata[divideData,]
test <- collegedata[-divideData,]



## 2. Fit a linear model
lmmodel <- lm(Apps~., data = train)
summary(lmmodel)

#Create a list of predictions using built-in predict function and test data
predictions <- predict(lmmodel, newdata = test)
lm.mse <- mse(predictions, test$Apps)   #Test error = 1,050,587.32



## 3. Fit a ridge regression model
library(glmnet)

x <- model.matrix(Apps ~ ., data = collegedata)[, -1] ; x
y <- collegedata$Apps ; y

#A range of values for lambda from 10^10 to 10^-2
grid <- 10^seq(10, -2, length = 100)

#Create the model
ridgemodel <- glmnet(x, y, alpha = 0, lambda = grid)

#Split the data into training and test sets
set.seed (1)
train <- sample(c(TRUE, FALSE), nrow(collegedata), replace= TRUE)
test <- (!train)
y.test <- y[test]


#Find the best lambda using 10-fold cross validation
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)
bestlam_ridge <- cv.out$lambda.min ; bestlam_ridge

#Make predictions using ridge model
ridge.pred <- predict(ridgemodel, s = bestlam_ridge, newx = x[test , ])

#Calculate test error of ridge model
ridge.mse <- mean((ridge.pred - y.test)^2) ; ridge.mse   #Test error = 892,906.33

#Calculate R-squared value
tss<- sum((y-mean(y))^2)
rss<- ridge.mse*384
ridge.rsq<- 1 - (rss/tss) ; ridge.rsq



## 4. Fit a lasso model

#Split the data into training and test sets
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(collegedata), replace= TRUE)
test <- (!train)
y.test <- y[test]

#Create the model
lassomodel <- glmnet(x[train , ], y[train], alpha = 1, lambda = grid)
plot(lassomodel)

#Find the best lambda using cross validation
cv.out <- cv.glmnet(x[train , ], y[train], alpha = 1)
plot(cv.out)
bestlam_lasso <- cv.out$lambda.min ; bestlam_lasso


#Make predictions using lasso model
lasso.pred <- predict(lassomodel , s = bestlam_lasso , newx = x[test , ])

#Calculate test error of lasso model
lasso.mse <- mean((lasso.pred - y.test)^2) ; lasso.mse   #Test error = 987,912.63

#Coefficient estimates
lassomodel.all <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(lassomodel.all, type = "coefficients", s = bestlam_lasso)[1:18,] ; lasso.coef
lasso.coef[lasso.coef != 0] #10 out of 17 predictors are non-zero



## 5. Fit a principal components regression (PCR) model 
library(pls)

#Split the data into training and test sets
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(collegedata), replace= TRUE)
test <- (!train)
y.test <- y[test]

#Create the PCR model
pcr.fit <- pcr(Apps~., data = collegedata, subset = train, scale = TRUE, validation = "CV")
#scale standardizes the predictor variables
summary(pcr.fit)

#Plot cross validation MSEs to determine number of principal components
validationplot(pcr.fit, val.type = "MSEP") #M = 10

#Make predictions using PCR model
pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 10)

#Calculate test error of PCR model
pcr.mse <- mean((pcr.pred - y.test)^2) ; pcr.mse   #Test error = 1,682,908.77



## 6. Fit a partial least squares model

#Split the data into training and test sets
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(collegedata), replace= TRUE)
test <- (!train)
y.test <- y[test]

#Create the PLS model
pls.fit <- plsr(Apps~., data = collegedata, subset = train, scale = TRUE, validation = "CV")
summary(pls.fit)

#Plot cross validation MSEs to determine number of principal components
validationplot(pls.fit, val.type = "MSEP") #M = 6

#Make predictions using PLS model
pls.pred <- predict(pls.fit, x[test, ], ncomp = 6)

#Calculate test error of PLS model
pls.mse <- mean((pls.pred - y.test)^2) ; pls.mse  #Test error = 1,011,426.08


