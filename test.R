library(kernlab)
library(drf)
library(Matrix)
library(dplyr)
library(doParallel)
library(doRNG)
library(parallel)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(drf)
library(ggplot2)
library(fastDummies)
library(Hmisc)

load("~/wage_data.Rdata")

set.seed(10)

which = rep(TRUE, nrow(wage))
which = which & (wage$age >= 17)
which = which & (wage$weeks_worked > 48)
which = which & (wage$hours_worked > 16)
which = which & (wage$employment_status == 'employed')
which = which & (wage$employer != 'self-employed')
which[is.na(which)] = FALSE

data = wage[which, ]
sum(is.na(data))
colSums(is.na(data))
rownames(data) = 1:nrow(data)

data$log_wage = log(data$salary / (data$weeks_worked * data$hours_worked))

## Prepare data and fit drf

X = data[,c(
  'age',
  'race',
  'hispanic_origin',
  'citizenship',
  'nativity',
  
  'marital',
  'family_size',
  'children',
  
  'education_level',
  'english_level',
  
  'economic_region'
)]
X$occupation = unlist(lapply(as.character(data$occupation), function(s){return(substr(s, 1, 2))}))
X$occupation = as.factor(X$occupation)
X$industry = unlist(lapply(as.character(data$industry), function(s){return(substr(s, 1, 2))}))
X$industry[X$industry %in% c('32', '33', '3M')] = '31'
X$industry[X$industry %in% c('42')] = '41'
X$industry[X$industry %in% c('45', '4M')] = '44'
X$industry[X$industry %in% c('49')] = '48'
X$industry[X$industry %in% c('92')] = '91'
X$industry = as.factor(X$industry)
X=dummy_cols(X, remove_selected_columns=TRUE)
X = as.matrix(X)

Y = data[,c('log_wage')]
W = data[,c('sex')]

train_idx = sample(1:nrow(data), 5000, replace=FALSE)

## Focus on training data
Ytrain=Y[train_idx]
Wtrain<-W[train_idx]
Wtrain<- matrix(ifelse(Wtrain == 'male', 1, 0), ncol=1)
#as.numeric(W[train_idx])
Xtrain=X[train_idx,]

Xtest <- X[train_idx[2]+1,, drop=F]

fit <- drf(X=Xtrain, Y=Ytrain, W=Wtrain, num.trees = 3000, ci.group.size = 150)

witness <- predict_witness(fit, alpha = 0.05, newdata = Xtest, newtreatment = matrix(1), g = rep(1, N))

data$witness <- witness[1,]
data$lower   <- witness[2, ]
data$upper   <- witness[3, ]