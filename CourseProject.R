# PREDICT 422 Practical Machine Learning
# Course Project
# Winter 2016
# Joshua Peng

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

# Load the data
charity <- read.csv("charity.csv") # load the "charity.csv" file
attach(charity)

# Identify Variables with Missing Values
colnames(charity)[colSums(is.na(charity)) > 0]

# Adding in reference reg5 to analyze correlations
sum(reg1)
sum(reg2)
sum(reg3)
sum(reg4)
8009-(1605+2555+1071+1117)
reg5 = rep(0,nrow(charity))
for(i in 1:nrow(charity)) {
  if (reg1[i]==0 && reg2[i]==0 && reg3[i]==0 && reg4[i]==0) {reg5[i]=1}}
charity$reg5 = reg5
charity.train <- charity[charity$part=="train",]
rm(charity.train)

# Analyzing REGx
library('corrplot') #package corrplot
regx_cor = cor(charity.train[,c(2:5,25,22)])
corrplot(regx_cor, method = "number") #plot matrix

# normality tests
library(nortest)
ad.test(hinc)
ad.test(npro)
ad.test(tdon)

# descriptive summary of variables
library(modeest)
summary(charity)
mfv(hinc)[1]
mfv(wrat)[1]
mfv(avhv)[1]
mfv(tlag)[1]
mfv(agif)[1]

# histograms
hist(reg1) # dichotomous
hist(reg2) # dichotomous
hist(reg3) # dichotomous
hist(reg4) # dichotomous
hist(home) # dichotomous
hist(chld) # zero inflated
hist(hinc) # median spike
hist(genf) # dichotomous
hist(wrat) # left skewed
hist(avhv) # right skewed
hist(incm) # right skewed
hist(inca) # right skewed
hist(plow) # right skewed, poisson
hist(npro) # right skewed
hist(tgif) # right skewed, poisson
hist(lgif) # right skewed, poisson
hist(rgif) # right skewed, poisson
hist(tdon) # slightly right skewed
hist(tlag) # right skewed
hist(agif) # right skewed

# predictor transformations
charity$wrat2 = wrat^2
charity$wrat3 = wrat^3
charity$hinc2 = hinc^2
charity$hinc3 = hinc^3
charity$ln_chld = log(chld)
charity$ln_avhv = log(avhv)
charity$ln_incm = log(incm)
charity$ln_inca = log(inca)
charity$ln_plow = log(plow)
charity$ln_npro = log(npro)
charity$ln_tgif = log(tgif)
charity$ln_lgif = log(lgif)
charity$ln_rgif = log(rgif)
charity$ln_tdon = log(tdon)
charity$ln_tlag = log(tlag)
charity$ln_agif = log(agif)
charity$sr_chld = sqrt(chld)
charity$sr_avhv = sqrt(avhv)
charity$sr_incm = sqrt(incm)
charity$sr_inca = sqrt(inca)
charity$sr_plow = sqrt(plow)
charity$sr_npro = sqrt(npro)
charity$sr_tgif = sqrt(tgif)
charity$sr_lgif = sqrt(lgif)
charity$sr_rgif = sqrt(rgif)
charity$sr_tdon = sqrt(tdon)
charity$sr_tlag = sqrt(tlag)
charity$sr_agif = sqrt(agif)
chld0 = rep(0,nrow(charity))
plow0 = rep(0,nrow(charity))
reg5 = rep(0,nrow(charity))
for(i in 1:nrow(charity)) {
if(chld[i]==0) {chld0[i]=1}
if(plow[i]==0) {plow0[i]=1}
if(reg1[i]==0 && reg2[i]==0 && reg3[i]==0 && reg4[i]==0) {reg5[i]=1}}
charity$reg5 = reg5
charity$chld0 = chld0
charity$plow0 = plow0
charity$ln_plow[charity$ln_plow <= 0] = 0
charity$ln_chld[charity$ln_chld <= 0] = 0
attach(charity)

# histograms after data transformations
hist(hinc2)
hist(hinc3)
hist(wrat2)
hist(wrat3)
hist(ln_chld)
hist(ln_avhv)
hist(ln_incm) 
hist(ln_inca) 
hist(ln_plow) 
hist(ln_npro) 
hist(ln_tgif) 
hist(ln_lgif) 
hist(ln_rgif) 
hist(ln_tdon) 
hist(ln_tlag) 
hist(ln_agif) 
hist(sr_chld)
hist(sr_avhv)
hist(sr_incm) 
hist(sr_inca) 
hist(sr_plow) 
hist(sr_npro) 
hist(sr_tgif) 
hist(sr_lgif) 
hist(sr_rgif) 
hist(sr_tdon) 
hist(sr_tlag) 
hist(sr_agif) 

# Set up data for analysis (initial steps)
data.train <- charity[charity$part=="train",]
x.train <- data.train[,c(2:21, 25:55)]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995
data.valid <- charity[charity$part=="valid",]
x.valid <- data.valid[,c(2:21, 25:55)]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999
data.test <- charity[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,c(2:21, 25:55)]

# Set up data for analysis (data splitting before standardization)
rm(data.train.std.y, data.train.std.c, data.valid.std.y, data.valid.std.c, data.test.std)
x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1
x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1
x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

# Set up data for analysis (Data splitting after standardization)
rm(data.train.std.y, data.train.std.c, data.valid.std.y, data.valid.std.c, data.test.std)
x.charity = charity[,c(2:21, 25:55)]
c.charity = charity[,22] # donr
n.charity.c = length(c.charity) # 8009
y.charity = charity[c.charity==1,23] # damt for observations with donr=1
n.charity.y = length(y.charity) # 5001
x.charity.mean = apply(x.charity, 2, mean)
x.charity.sd = apply(x.charity, 2, sd)
x.charity.std = t((t(x.charity)-x.charity.mean)/x.charity.sd) # standardize to have zero mean and unit sd
apply(x.charity.std, 2, mean) # check zero mean
apply(x.charity.std, 2, sd) # check unit sd
data.charity.std.c = data.frame(x.charity.std, donr=c.charity, part=charity$part) # to classify donr
data.charity.std.y = data.frame(x.charity.std[c.charity==1,], damt=y.charity) # to predict damt when donr=1
data.charity.std.y0 = data.frame(x.charity.std, part=charity$part) # add in part
data.charity.std.y1 = data.frame(data.charity.std.y0[c.charity==1,], damt=y.charity) # to predict damt when donr=1
i.train.c = which(data.charity.std.c$part=="train")
i.train.y = which(data.charity.std.y1$part=="train")
i.valid.c = which(data.charity.std.c$part=="valid")
i.valid.y = which(data.charity.std.y1$part=="valid")
i.test = which(charity$part=="test")
data.train.std.c = data.charity.std.c[i.train.c,]
data.train.std.y = data.charity.std.y[i.train.y,]
data.valid.std.c = data.charity.std.c[i.valid.c,]
data.valid.std.y = data.charity.std.y[i.valid.y,]
data.train.std.c$part = NULL
data.train.std.y$part = NULL
data.valid.std.c$part = NULL
data.valid.std.y$part = NULL
data.test.std = x.charity.std[i.test,]

##### CLASSIFICATION MODELING ######

library(MASS)
library(leaps)
library(gam)
library(ROCR)
library(AUC)
vs1 = regsubsets(donr ~ ., data = data.train.std.c, nvmax = 20)
plot(vs1, scale="adjr2")

# Logistic Regression

# M1c: Trimmed subset of 12 variables
model.log2 = glm(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag, data.train.std.c, family=binomial("logit"))
# M1b: Best 20 variable model from regsubsets
model.log2 = glm(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                   wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                   ln_agif + sr_chld + sr_tdon + sr_tlag, data.train.std.c, family=binomial("logit"))
# M1a: Default Original subset
model.log2 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))

##### My Logistic Regression Model
post.valid.log2 <- predict(model.log2, data.valid.std.c, type="response") # n.valid post probs
profit.log2 <- cumsum(14.5*c.valid[order(post.valid.log2, decreasing=T)]-2)
plot(profit.log2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log2)) # report number of mailings and maximum profit
cutoff.log2 <- sort(post.valid.log2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log2 <- ifelse(post.valid.log2>cutoff.log2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log2, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.log2, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)
#####

model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))
post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# 1291.0 11642.5
cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 709  18
#              1 310 981
# check n.mail.valid = 310+981 = 1291
# check profit = 14.5*981-2*1291 = 11642.5
# Results
# n.mail Profit  Model
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# Linear Discriminant Analysis

# M2c: Trimmed subset of 12 variables
model.lda2 = lda(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag, data.train.std.c)
# M2b: Best 20 variable model from regsubsets
model.lda2 = lda(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                   wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                   ln_agif + sr_chld + sr_tdon + sr_tlag, data.train.std.c)
# M2a: Default Original subset
model.lda2 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()

##### My Linear Discriminant Analysis
post.valid.lda2 <- predict(model.lda2, data.valid.std.c)$posterior[,2] # n.valid.c post probs
profit.lda2 <- cumsum(14.5*c.valid[order(post.valid.lda2, decreasing=T)]-2)
plot(profit.lda2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda2)) # report number of mailings and maximum profit
cutoff.lda2 <- sort(post.valid.lda2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda2 <- ifelse(post.valid.lda2>cutoff.lda2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda2, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.lda2, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)
#####

model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()
# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model
post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs
# calculate ordered profit function using average donation = $14.50 and mailing cost = $2
profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# 1329.0 11624.5
cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#               c.valid
#chat.valid.lda1   0   1
#              0 675  14
#              1 344 985
# check n.mail.valid = 344+985 = 1329
# check profit = 14.5*985-2*1329 = 11624.5

# Quadratic Discriminant Analysis

# M3c: Trimmed subset of 12 variables
model.qda2 = qda(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag, data.train.std.c)
# M3b: Best 20 variable model from regsubsets
model.qda2 = qda(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                   wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                   ln_agif + sr_chld + sr_tdon + sr_tlag, data.train.std.c)
# M3a: Default Original subset
model.qda2 <- qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()

post.valid.qda2 <- predict(model.qda2, data.valid.std.c)$posterior[,2] # n.valid.c post probs
profit.qda2 <- cumsum(14.5*c.valid[order(post.valid.qda2, decreasing=T)]-2)
plot(profit.qda2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda2)) # report number of mailings and maximum profit
cutoff.qda2 <- sort(post.valid.qda2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda2 <- ifelse(post.valid.qda2>cutoff.qda2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda2, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.qda2, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)
#####

# K Nearest Neighbors
library(class)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=1)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=2)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=3)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=4)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=5)
set.seed(1)
model.knn1=knn(data.train.std.c,data.valid.std.c,c.train,k=6)

post.valid.knn1 <- as.integer(as.character(model.knn1))
profit.knn1 <- cumsum(14.5*c.valid[order(post.valid.knn1, decreasing=T)]-2)
plot(profit.knn1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn1)) # report number of mailings and maximum profit
cutoff.knn1 <- sort(post.valid.knn1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.knn1 <- ifelse(post.valid.knn1>cutoff.knn1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.knn1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.knn1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Logistic Regression using GAM with Natural Splines
library(gam)
# M5c: Trimmed subset of 12 variables
model.ns1 = gam(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                ns(ln_incm,4) + ns(ln_tgif,4) + ns(sr_tdon,4) + ns(sr_tlag,4), 
                data.train.std.c, family=binomial("logit"))
# M5b: Best 20 variable model from regsubsets
model.ns1 = gam(donr ~ reg1 + reg2 + home + chld + hinc + ns(inca,4) + ns(tgif,4) + ns(tlag,4) + wrat2 + 
                   wrat3 + hinc2 + ns(ln_incm,4) + ns(ln_tgif,4) + ns(ln_lgif,4) + ns(ln_tdon,4) + 
                   ns(ln_tlag,4) + ns(ln_agif,4) + sr_chld + ns(sr_tdon,4) + ns(sr_tlag,4), 
                   data.train.std.c, family=binomial("logit"))
# M5a: Default Original subset
model.ns1 <- gam(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ns(ln_avhv,4) + ns(incm,4) + ns(inca,4) + ns(plow,4) + ns(npro,4) + ns(tgif,4) + 
                  ns(lgif,4) + ns(rgif,4) + ns(tdon,4) + ns(tlag,4) + ns(agif,4), 
                  data.train.std.c, family=binomial("logit"))

post.valid.ns1 <- predict(model.ns1, data.valid.std.c, type="response") # n.valid post probs
profit.ns1 <- cumsum(14.5*c.valid[order(post.valid.ns1, decreasing=T)]-2)
plot(profit.ns1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.ns1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.ns1)) # report number of mailings and maximum profit
cutoff.ns1 <- sort(post.valid.ns1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.ns1 <- ifelse(post.valid.ns1>cutoff.ns1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.ns1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.ns1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Logistic Regression using GAM with Smoothing Splines
library(gam)
# M6c: Trimmed subset of 12 variables
model.ss1 = gam(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                s(ln_incm,4) + s(ln_tgif,4) + s(sr_tdon,4) + s(sr_tlag,4), 
                data.train.std.c, family=binomial("logit"))
# M6b: Best 20 variable model from regsubsets
model.ss1 = gam(donr ~ reg1 + reg2 + home + chld + hinc + s(inca,4) + s(tgif,4) + s(tlag,4) + wrat2 + 
                wrat3 + hinc2 + s(ln_incm,4) + s(ln_tgif,4) + s(ln_lgif,4) + s(ln_tdon,4) + 
                s(ln_tlag,4) + s(ln_agif,4) + sr_chld + s(sr_tdon,4) + s(sr_tlag,4), 
                data.train.std.c, family=binomial("logit"))
# M6a: Default Original subset
model.ss1 <- gam(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                 s(ln_avhv,4) + s(incm,4) + s(inca,4) + s(plow,4) + s(npro,4) + s(tgif,4) + 
                 s(lgif,4) + s(rgif,4) + s(tdon,4) + s(tlag,4) + s(agif,4), 
                 data.train.std.c, family=binomial("logit"))

post.valid.ss1 <- predict(model.ss1, data.valid.std.c, type="response") # n.valid post probs
profit.ss1 <- cumsum(14.5*c.valid[order(post.valid.ss1, decreasing=T)]-2)
plot(profit.ss1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.ss1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.ss1)) # report number of mailings and maximum profit
cutoff.ss1 <- sort(post.valid.ss1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.ss1 <- ifelse(post.valid.ss1>cutoff.ss1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.ss1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.ss1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Logistic Regression using GAM with Local Regression Smoothing (span=0.2)
library(gam)
# M7c: Trimmed subset of 12 variables
model.lo1 = gam(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                lo(ln_incm,span=0.2) + lo(ln_tgif,span=0.2) + lo(sr_tdon,span=0.2) + lo(sr_tlag,span=0.2), 
                data.train.std.c, family=binomial("logit"))
# M7b: Best 20 variable model from regsubsets
model.lo1 = gam(donr ~ reg1 + reg2 + home + chld + hinc + lo(inca,span=0.2) + lo(tgif,span=0.2) + 
                  lo(tlag,span=0.2) + wrat2 + wrat3 + hinc2 + lo(ln_incm,span=0.2) + lo(ln_tgif,span=0.2) + 
                  lo(ln_lgif,span=0.2) + lo(ln_tdon,span=0.2) + lo(ln_tlag,span=0.2) + lo(ln_agif,span=0.2) + 
                  sr_chld + lo(sr_tdon,span=0.2) + lo(sr_tlag,span=0.2), 
                  data.train.std.c, family=binomial("logit"))
# M7a: Default Original subset
model.lo1 <- gam(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                 lo(ln_avhv,span=0.2) + lo(incm,span=0.2) + lo(inca,span=0.2) + lo(plow,span=0.2) + 
                 lo(npro,span=0.2) + lo(tgif,span=0.2) + lo(lgif,span=0.2) + lo(rgif,span=0.2) + 
                 lo(tdon,span=0.2) + lo(tlag,span=0.2) + lo(agif,span=0.2), 
                 data.train.std.c, family=binomial("logit"))

post.valid.lo1 <- predict(model.lo1, data.valid.std.c, type="response") # n.valid post probs
profit.lo1 <- cumsum(14.5*c.valid[order(post.valid.lo1, decreasing=T)]-2)
plot(profit.lo1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lo1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lo1)) # report number of mailings and maximum profit
cutoff.lo1 <- sort(post.valid.lo1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lo1 <- ifelse(post.valid.lo1>cutoff.lo1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lo1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.lo1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Logistic Regression using GAM with Local Regression Smoothing (span=0.5)
library(gam)
# M8c: Trimmed subset of 12 variables
model.lo2 = gam(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                  lo(ln_incm,span=0.5) + lo(ln_tgif,span=0.5) + lo(sr_tdon,span=0.5) + lo(sr_tlag,span=0.5), 
                  data.train.std.c, family=binomial("logit"))
# M8b: Best 20 variable model from regsubsets
model.lo2 = gam(donr ~ reg1 + reg2 + home + chld + hinc + lo(inca,span=0.5) + lo(tgif,span=0.5) + 
                  lo(tlag,span=0.5) + wrat2 + wrat3 + hinc2 + lo(ln_incm,span=0.5) + lo(ln_tgif,span=0.5) + 
                  lo(ln_lgif,span=0.5) + lo(ln_tdon,span=0.5) + lo(ln_tlag,span=0.5) + lo(ln_agif,span=0.5) + 
                  sr_chld + lo(sr_tdon,span=0.5) + lo(sr_tlag,span=0.5), 
                  data.train.std.c, family=binomial("logit"))
# M8a: Default Original subset
model.lo2 <- gam(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                   lo(ln_avhv,span=0.5) + lo(incm,span=0.5) + lo(inca,span=0.5) + lo(plow,span=0.5) + 
                   lo(npro,span=0.5) + lo(tgif,span=0.5) + lo(lgif,span=0.5) + lo(rgif,span=0.5) + 
                   lo(tdon,span=0.5) + lo(tlag,span=0.5) + lo(agif,span=0.5), 
                   data.train.std.c, family=binomial("logit"))

post.valid.lo2 <- predict(model.lo2, data.valid.std.c, type="response") # n.valid post probs
profit.lo2 <- cumsum(14.5*c.valid[order(post.valid.lo2, decreasing=T)]-2)
plot(profit.lo2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lo2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lo2)) # report number of mailings and maximum profit
cutoff.lo2 <- sort(post.valid.lo2, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lo2 <- ifelse(post.valid.lo2>cutoff.lo2, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lo2, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.lo2, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Decision Tree
library(tree)
# M9c: Trimmed subset of 12 variables
set.seed(1)
model.tr1 = tree(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag, data=data.train.std.c)
# M9b: Best 20 variable model from regsubsets
set.seed(1)
model.tr1 = tree(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                   wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                   ln_agif + sr_chld + sr_tdon + sr_tlag, data=data.train.std.c)
# M9a: Default Original subset
set.seed(1)
model.tr1 <- tree(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                    data=data.train.std.c) # include additional terms on the fly using I()

post.valid.tr1 <- predict(model.tr1, data.valid.std.c) # n.valid post probs
profit.tr1 <- cumsum(14.5*c.valid[order(post.valid.tr1, decreasing=T)]-2)
plot(profit.tr1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.tr1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.tr1)) # report number of mailings and maximum profit
cutoff.tr1 <- sort(post.valid.tr1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tr1 <- ifelse(post.valid.tr1>cutoff.tr1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tr1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.tr1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Bagging
library(randomForest)
# M10c: Trimmed subset of 12 variables
set.seed(1)
model.bag1 = randomForest(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag, data=data.train.std.c, 
                   mtry=12, ntree=500, importance=T)
# M10b: Best 20 variable model from regsubsets
set.seed(1)
model.bag1 = randomForest(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                   wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                   ln_agif + sr_chld + sr_tdon + sr_tlag, data=data.train.std.c, 
                   mtry=20, ntree=500, importance=T)
# M10a: Default Original subset
set.seed(1)
model.bag1 = randomForest(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data=data.train.std.c, mtry=21, ntree=500, importance=T)

post.valid.bag1 <- predict(model.bag1, data.valid.std.c) # n.valid post probs
profit.bag1 <- cumsum(14.5*c.valid[order(post.valid.bag1, decreasing=T)]-2)
plot(profit.bag1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.bag1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.bag1)) # report number of mailings and maximum profit
cutoff.bag1 <- sort(post.valid.bag1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.bag1 <- ifelse(post.valid.bag1>cutoff.bag1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.bag1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.bag1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Random Forests
library(randomForest)
# M11c: Trimmed subset of 12 variables
set.seed(1)
model.rf1 = randomForest(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                          ln_incm + ln_tgif + sr_tdon + sr_tlag, data=data.train.std.c, 
                          mtry=3, ntree=500, importance=T)
# M11b: Best 20 variable model from regsubsets
set.seed(1)
model.rf1 =  randomForest(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                          wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                          ln_agif + sr_chld + sr_tdon + sr_tlag, data=data.train.std.c, 
                          mtry=4, ntree=500, importance=T)
# M11a: Default Original subset
set.seed(1)
model.rf1 <-  randomForest(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                            ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                            data=data.train.std.c, mtry=5, ntree=500, importance=T)

post.valid.rf1 <- predict(model.rf1, data.valid.std.c) # n.valid post probs
profit.rf1 <- cumsum(14.5*c.valid[order(post.valid.rf1, decreasing=T)]-2)
plot(profit.rf1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.rf1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.rf1)) # report number of mailings and maximum profit
cutoff.rf1 <- sort(post.valid.rf1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.rf1 <- ifelse(post.valid.rf1>cutoff.rf1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.rf1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.rf1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Boosting
library(gbm)
# M12c: Trimmed subset of 12 variables
set.seed(1)
model.boo1 = gbm(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                        ln_incm + ln_tgif + sr_tdon + sr_tlag, data=data.train.std.c, 
                        distribution="gaussian", n.trees=5000,
                        interaction.depth=4,shrinkage=0.01,verbose=F)
# M12b: Best 20 variable model from regsubsets
set.seed(1)
model.boo1 =  gbm(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                         wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                         ln_agif + sr_chld + sr_tdon + sr_tlag, data=data.train.std.c, 
                         distribution="gaussian", n.trees=5000,
                         interaction.depth=4,shrinkage=0.01,verbose=F)
# M12a: Default Original subset
set.seed(1)
model.boo1 <-  gbm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                          ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                          data=data.train.std.c, distribution="gaussian", n.trees=5000,
                          interaction.depth=4,shrinkage=0.01,verbose=F)

post.valid.boo1 <- predict(model.boo1, data.valid.std.c, n.trees=5000) # n.valid post probs
profit.boo1 <- cumsum(14.5*c.valid[order(post.valid.boo1, decreasing=T)]-2)
plot(profit.boo1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.boo1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.boo1)) # report number of mailings and maximum profit
cutoff.boo1 <- sort(post.valid.boo1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.boo1 <- ifelse(post.valid.boo1>cutoff.boo1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.boo1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.boo1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Artificial Neural Network
library(nnet)
xc_vars = c("reg1","reg2","home","sr_chld","hinc","hinc2","wrat2","wrat3",
            "ln_incm","ln_tgif","sr_tdon","sr_tlag")
xb_vars = c("reg1","reg2","home","chld","hinc","inca","tgif","tlag","wrat2", 
            "wrat3","hinc2","ln_incm","ln_tgif","ln_lgif","ln_tdon","ln_tlag", 
            "ln_agif","sr_chld","sr_tdon","sr_tlag")
xa_vars = c("reg1","reg2","reg3","reg4","home","chld","hinc","hinc2","genf","wrat", 
            "ln_avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon",
            "tlag","agif")
xc1 = data.train.std.c[,xc_vars]
xb1 = data.train.std.c[,xb_vars]
xa1 = data.train.std.c[,xa_vars]
xvc1 = data.valid.std.c[,xc_vars]
xvb1 = data.valid.std.c[,xb_vars]
xva1 = data.valid.std.c[,xa_vars]

# M12c: Trimmed subset of 12 variables
set.seed(1)
model.nn1 = nnet(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag,
                   data=data.train.std.c, size=20, maxit=100, decay=.001)
newdata=xvc1
# M12b: Best 20 variable model from regsubsets
set.seed(1)
model.nn1 =  nnet(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                    wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                    ln_agif + sr_chld + sr_tdon + sr_tlag, 
                    data=data.train.std.c, size=20, maxit=100, decay=.001)
newdata=xvb1
# M12a: Default Original subset
set.seed(1)
model.nn1 <-  nnet(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + hinc2 + genf + wrat + 
                     ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                     data=data.train.std.c, size=20, maxit=100, decay=.001)
newdata=xva1

post.valid.nn1 <- predict(model.nn1, newdata) # n.valid post probs
profit.nn1 <- cumsum(14.5*c.valid[order(post.valid.nn1, decreasing=T)]-2)
plot(profit.nn1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.nn1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.nn1)) # report number of mailings and maximum profit
cutoff.nn1 <- sort(post.valid.nn1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.nn1 <- ifelse(post.valid.nn1>cutoff.nn1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.nn1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.nn1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Support Vector Machines (Linear kernel)
library(e1071)

# M12c: Trimmed subset of 12 variables
set.seed(1)
model.svm1 = svm(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                 ln_incm + ln_tgif + sr_tdon + sr_tlag,
                 data=data.train.std.c, kernel="linear", cost=0.1, scale=FALSE)
# M12b: Best 20 variable model from regsubsets
set.seed(1)
model.svm1 =  svm(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                  wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                  ln_agif + sr_chld + sr_tdon + sr_tlag, 
                  data=data.train.std.c, kernel="linear", cost=0.1, scale=FALSE)
# M12a: Default Original subset
set.seed(1)
model.svm1 = svm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + hinc2 + genf + wrat + 
              ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
              data=data.train.std.c, kernel="linear", cost=0.1, scale=FALSE)

post.valid.svm1 <- predict(model.svm1, data.valid.std.c) # n.valid post probs
profit.svm1 <- cumsum(14.5*c.valid[order(post.valid.svm1, decreasing=T)]-2)
plot(profit.svm1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm1)) # report number of mailings and maximum profit
cutoff.svm1 <- sort(post.valid.svm1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm1 <- ifelse(post.valid.svm1>cutoff.svm1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.svm1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

# Support Vector Machines (Radial Kernel)
library(e1071)

# M12c: Trimmed subset of 12 variables
set.seed(1)
model.svm1 = svm(donr ~ reg1 + reg2 + home + sr_chld + hinc + hinc2 + wrat2 + wrat3 +
                   ln_incm + ln_tgif + sr_tdon + sr_tlag,
                 data=data.train.std.c, kernel="radial", cost=0.1, scale=FALSE)
# M12b: Best 20 variable model from regsubsets
set.seed(1)
model.svm1 =  svm(donr ~ reg1 + reg2 + home + chld + hinc + inca + tgif + tlag + wrat2 + 
                    wrat3 + hinc2 + ln_incm + ln_tgif + ln_lgif + ln_tdon + ln_tlag + 
                    ln_agif + sr_chld + sr_tdon + sr_tlag, 
                  data=data.train.std.c, kernel="radial", cost=0.1, scale=FALSE)
# M12a: Default Original subset
set.seed(1)
model.svm1 = svm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + hinc2 + genf + wrat + 
                   ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                 data=data.train.std.c, kernel="radial", cost=0.1, scale=FALSE)

post.valid.svm1 <- predict(model.svm1, data.valid.std.c) # n.valid post probs
profit.svm1 <- cumsum(14.5*c.valid[order(post.valid.svm1, decreasing=T)]-2)
plot(profit.svm1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm1)) # report number of mailings and maximum profit
cutoff.svm1 <- sort(post.valid.svm1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm1 <- ifelse(post.valid.svm1>cutoff.svm1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm1, c.valid) # classification table
## computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
library(ROCR)
pred <- prediction(chat.valid.svm1, c.valid)
perf <- performance(pred,"tpr","fpr")
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values
plot(perf)

model.best.c = model.boo1 # model 12a
profit.best.c = profit.boo1
# select best model to have maximum profit in the validation sample
post.test <- predict(model.best.c, data.test.std, n.tree = 5000) # post probs for test data
# Oversampling adjustment for calculating number of mailings for test set
n.mail.valid <- which.max(profit.best.c)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set
cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#    0    1 
# 1676  331
# based on this model we'll mail to the 331 highest posterior probabilities

##### PREDICTION MODELING ######

library(leaps)
library(stats)
vs2 = regsubsets(damt ~ ., data = data.train.std.y, really.big = T, nvmax = 20)
plot(vs2, scale="adjr2")

# Least Squares Regression

# M1c: Trimmed subset of 15 variables
model.ls1 = lm(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
               hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
               data.train.std.y)
# M1b: Best 20 variable model from regsubsets
model.ls1 = lm(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
              wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
              ln_agif + sr_incm + sr_plow + sr_lgif, data.train.std.y)
# M1a: Default Original subset
model.ls1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.867523
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1696615
summary(model.ls1)$r.squared
summary(model.ls1)$adj.r.squared
AIC(model.ls1)
AIC(model.ls1,k=8.290042) # for BIC, k=ln(#obs in training set)

# Ridge Regression

library(glmnet)
grid=10^seq(10,-2,length=10000)
y = as.matrix(data.train.std.y$damt)
xc_vars = c("reg3","reg4","reg5","home","chld","hinc","plow","wrat","wrat2",
            "hinc3","ln_incm","ln_lgif","ln_tgif","ln_rgif","ln_agif")
xb_vars = c("reg3","reg4","reg5","home","chld","hinc","plow","wrat","wrat2", 
            "wrat3","hinc3","ln_incm","ln_plow","ln_tgif","ln_lgif","ln_rgif", 
            "ln_agif","sr_incm","sr_plow","sr_lgif")
xa_vars = c("reg1","reg2","reg3","reg4","home","chld","hinc","genf","wrat", 
            "ln_avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon",
            "tlag","agif")
xc = as.matrix(data.train.std.y)[,xc_vars]
xb = as.matrix(data.train.std.y)[,xb_vars]
xa = as.matrix(data.train.std.y)[,xa_vars]
xvc = as.matrix(data.valid.std.y)[,xc_vars]
xvb = as.matrix(data.valid.std.y)[,xb_vars]
xva = as.matrix(data.valid.std.y)[,xa_vars]

# M2c: Trimmed subset of 15 variables
set.seed(1)
cv.out = cv.glmnet(xc,y,alpha=0,lambda=grid)
bestlam = cv.out$lambda.min
newdata = xvc
model.rr1 = glmnet(xc,y,alpha=0,lambda=bestlam, standardize=FALSE)
# M2b: Best 20 variable model from regsubsets
set.seed(1)
cv.out = cv.glmnet(xb,y,alpha=0,lambda=grid, standardize=FALSE)
bestlam = cv.out$lambda.min
newdata = xvb
model.rr1 = glmnet(xb,y,alpha=0,lambda=bestlam, standardize=FALSE)
# M2a: Default Original subset
set.seed(1)
cv.out = cv.glmnet(xa,y,alpha=0,lambda=grid)
bestlam = cv.out$lambda.min
newdata = xva
model.rr1 = glmnet(xa,y,alpha=0,lambda=bestlam, standardize=FALSE)

pred.valid.rr1 = predict(model.rr1, newx=newdata, s=bestlam) # validation predictions
mean((y.valid - pred.valid.rr1)^2) # mean prediction error
sd((y.valid - pred.valid.rr1)^2)/sqrt(n.valid.y) # std error
bestlam

# Lasso

library(glmnet)
grid=10^seq(10,-2,length=10000)
y = as.matrix(data.train.std.y$damt)
xc_vars = c("reg3","reg4","reg5","home","chld","hinc","plow","wrat","wrat2",
            "hinc3","ln_incm","ln_lgif","ln_tgif","ln_rgif","ln_agif")
xb_vars = c("reg3","reg4","reg5","home","chld","hinc","plow","wrat","wrat2", 
            "wrat3","hinc3","ln_incm","ln_plow","ln_tgif","ln_lgif","ln_rgif", 
            "ln_agif","sr_incm","sr_plow","sr_lgif")
xa_vars = c("reg1","reg2","reg3","reg4","home","chld","hinc","genf","wrat", 
            "ln_avhv","incm","inca","plow","npro","tgif","lgif","rgif","tdon",
            "tlag","agif")
xc = as.matrix(data.train.std.y)[,xc_vars]
xb = as.matrix(data.train.std.y)[,xb_vars]
xa = as.matrix(data.train.std.y)[,xa_vars]
xvc = as.matrix(data.valid.std.y)[,xc_vars]
xvb = as.matrix(data.valid.std.y)[,xb_vars]
xva = as.matrix(data.valid.std.y)[,xa_vars]

# M3c: Trimmed subset of 15 variables
set.seed(1)
cv.out = cv.glmnet(xc,y,alpha=1,lambda=grid)
bestlam = cv.out$lambda.min
newdata = xvc
model.las1 = glmnet(xc,y,alpha=1,lambda=bestlam, standardize=FALSE)
# M3b: Best 20 variable model from regsubsets
set.seed(1)
cv.out = cv.glmnet(xb,y,alpha=1,lambda=grid, standardize=FALSE)
bestlam = cv.out$lambda.min
newdata = xvb
model.las1 = glmnet(xb,y,alpha=1,lambda=bestlam, standardize=FALSE)
# M3a: Default Original subset
set.seed(1)
cv.out = cv.glmnet(xa,y,alpha=1,lambda=grid)
bestlam = cv.out$lambda.min
newdata = xva
model.las1 = glmnet(xa,y,alpha=1,lambda=bestlam, standardize=FALSE)

pred.valid.las1 = predict(model.las1, newx=newdata, alpha=1, s=bestlam) # validation predictions
mean((y.valid - pred.valid.las1)^2) # mean prediction error
sd((y.valid - pred.valid.las1)^2)/sqrt(n.valid.y) # std error
bestlam
lasso.coef = predict(model.las1, newx=newdata, s=bestlam, alpha=1, type="coefficients")
lasso.coef

# Principal Components Regression

# M4c: Trimmed subset of 15 variables
library(pls)
set.seed(1)
model.pcr1 = pcr(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                 hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
                 data=data.train.std.y, validation ="CV", scale = F)
# M4b: Best 20 variable model from regsubsets
set.seed(1)
model.pcr1 = pcr(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                 wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                 ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y, 
                 validation ="CV", scale = F)
# M4a: Default Original subset
set.seed(1)
model.pcr1 <- pcr(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data=data.train.std.y, validation="CV", scale = F)

pred.valid.pcr1 <- predict(model.pcr1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.pcr1)^2) # mean prediction error
sd((y.valid - pred.valid.pcr1)^2)/sqrt(n.valid.y) # std error
summary(model.pcr1)

# Partial Least Squares

# M5c: Trimmed subset of 15 variables
set.seed(1)
model.pls1 = plsr(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                  hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
                  data=data.train.std.y, validation ="CV", scale = F)
# M5b: Best 20 variable model from regsubsets
set.seed(1)
model.pls1 = plsr(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                   wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                   ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y, 
                   validation ="CV", scale = F)
# M5a: Default Original subset
set.seed(1)
model.pls1 <- plsr(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                    ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                    data=data.train.std.y, validation="CV", scale = F)

pred.valid.pls1 <- predict(model.pls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.pls1)^2) # mean prediction error
sd((y.valid - pred.valid.pls1)^2)/sqrt(n.valid.y) # std error
summary(model.pls1)

# Least Squares Regression using GAM with Natural Splines
library(gam)
# M6c: Trimmed subset of 12 variables
model.ns1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + ns(plow,4) + wrat + wrat2 +
                hinc3 + ns(ln_incm,4) + ns(ln_lgif,4) + ns(ln_tgif,4) + ns(ln_rgif,4) + ns(ln_agif,4), 
                data.train.std.y, family=gaussian("identity"))
# M6b: Best 20 variable model from regsubsets
model.ns1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + ns(plow,4) + wrat + wrat2 + 
                  wrat3 + hinc3 + ns(ln_incm,4) + ns(ln_plow,4) + ns(ln_tgif,4) + 
                  ns(ln_lgif,4) + ns(ln_rgif,4) + ns(ln_agif,4) + ns(sr_incm,4) + 
                  ns(sr_plow,4) + ns(sr_lgif,4), 
                  data.train.std.y, family=gaussian("identity"))
# M6a: Default Original subset
model.ns1 <- gam(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   ns(ln_avhv,4) + ns(incm,4) + ns(inca,4) + ns(plow,4) + ns(npro,4) + ns(tgif,4) + 
                   ns(lgif,4) + ns(rgif,4) + ns(tdon,4) + ns(tlag,4) + ns(agif,4), 
                   data.train.std.y, family=gaussian("identity"))

pred.valid.ns1 <- predict(model.ns1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ns1)^2) # mean prediction error
sd((y.valid - pred.valid.ns1)^2)/sqrt(n.valid.y) # std error
AIC(model.ns1)
AIC(model.ns1,k=8.290042) # for BIC, k=ln(#obs in training set)


# Least Squares Regression using GAM with Smoothing Splines
library(gam)
# M7c: Trimmed subset of 12 variables
model.ss1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + s(plow,4) + wrat + wrat2 +
                hinc3 + s(ln_incm,4) + s(ln_lgif,4) + s(ln_tgif,4) + s(ln_rgif,4) + s(ln_agif,4), 
                data.train.std.y, family=gaussian("identity"))
# M7b: Best 20 variable model from regsubsets
model.ss1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + s(plow,4) + wrat + wrat2 + 
                  wrat3 + hinc3 + s(ln_incm,4) + s(ln_plow,4) + s(ln_tgif,4) + 
                  s(ln_lgif,4) + s(ln_rgif,4) + s(ln_agif,4) + s(sr_incm,4) + 
                  s(sr_plow,4) + s(sr_lgif,4), 
                  data.train.std.y, family=gaussian("identity"))
# M7a: Default Original subset
model.ss1 <- gam(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   s(ln_avhv,4) + s(incm,4) + s(inca,4) + s(plow,4) + s(npro,4) + s(tgif,4) + 
                   s(lgif,4) + s(rgif,4) + s(tdon,4) + s(tlag,4) + s(agif,4), 
                   data.train.std.y, family=gaussian("identity"))

pred.valid.ss1 <- predict(model.ss1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ss1)^2) # mean prediction error
sd((y.valid - pred.valid.ss1)^2)/sqrt(n.valid.y) # std error
AIC(model.ss1)
AIC(model.ss1,k=8.290042) # for BIC, k=ln(#obs in training set)

# Least Squares Regression using GAM with Local Regression Smoothing (span=0.2)
library(gam)
# M8c: Trimmed subset of 12 variables
model.lo1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + lo(plow,span=0.2) + wrat + wrat2 +
                  hinc3 + lo(ln_incm,span=0.2) + lo(ln_lgif,span=0.2) + lo(ln_tgif,span=0.2) + 
                  lo(ln_rgif,span=0.2) + lo(ln_agif,span=0.2), 
                  data.train.std.y, family=gaussian("identity"))
# M8b: Best 20 variable model from regsubsets
model.lo1 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + lo(plow,span=0.2) + wrat + wrat2 + 
                  wrat3 + hinc3 + lo(ln_incm,span=0.2) + lo(ln_plow,span=0.2) + lo(ln_tgif,span=0.2) + 
                  lo(ln_lgif,span=0.2) + lo(ln_rgif,span=0.2) + lo(ln_agif,span=0.2) + lo(sr_incm,span=0.2) + 
                  lo(sr_plow,span=0.2) + lo(sr_lgif,span=0.2), 
                  data.train.std.y, family=gaussian("identity"))
# M8a: Default Original subset
model.lo1 <- gam(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   lo(ln_avhv,span=0.2) + lo(incm,span=0.2) + lo(inca,span=0.2) + 
                   lo(plow,span=0.2) + lo(npro,span=0.2) + lo(tgif,span=0.2) + 
                   lo(lgif,span=0.2) + lo(rgif,span=0.2) + lo(tdon,span=0.2) + 
                   lo(tlag,span=0.2) + lo(agif,span=0.2), 
                   data.train.std.y, family=gaussian("identity"))

pred.valid.lo1 <- predict(model.lo1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.lo1)^2) # mean prediction error
sd((y.valid - pred.valid.lo1)^2)/sqrt(n.valid.y) # std error
AIC(model.lo1)
AIC(model.lo1,k=8.290042) # for BIC, k=ln(#obs in training set)

# Least Squares Regression using GAM with Local Regression Smoothing (span=0.5)
library(gam)
# M9c: Trimmed subset of 12 variables
model.lo2 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + lo(plow,span=0.5) + wrat + wrat2 +
                  hinc3 + lo(ln_incm,span=0.5) + lo(ln_lgif,span=0.5) + lo(ln_tgif,span=0.5) + 
                  lo(ln_rgif,span=0.5) + lo(ln_agif,span=0.5), 
                  data.train.std.y, family=gaussian("identity"))
# M9b: Best 20 variable model from regsubsets
model.lo2 = gam(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + lo(plow,span=0.5) + wrat + wrat2 + 
                  wrat3 + hinc3 + lo(ln_incm,span=0.5) + lo(ln_plow,span=0.5) + lo(ln_tgif,span=0.5) + 
                  lo(ln_lgif,span=0.5) + lo(ln_rgif,span=0.5) + lo(ln_agif,span=0.5) + lo(sr_incm,span=0.5) + 
                  lo(sr_plow,span=0.5) + lo(sr_lgif,span=0.5), 
                  data.train.std.y, family=gaussian("identity"))
# M9a: Default Original subset
model.lo2 <- gam(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   lo(ln_avhv,span=0.5) + lo(incm,span=0.5) + lo(inca,span=0.5) + 
                   lo(plow,span=0.5) + lo(npro,span=0.5) + lo(tgif,span=0.5) + 
                   lo(lgif,span=0.5) + lo(rgif,span=0.5) + lo(tdon,span=0.5) + 
                   lo(tlag,span=0.5) + lo(agif,span=0.5), 
                   data.train.std.y, family=gaussian("identity"))

pred.valid.lo2 <- predict(model.lo2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.lo2)^2) # mean prediction error
sd((y.valid - pred.valid.lo2)^2)/sqrt(n.valid.y) # std error
AIC(model.lo2)
AIC(model.lo2,k=8.290042) # for BIC, k=ln(#obs in training set)

# Decision Tree
library(tree)
# M10c: Trimmed subset of 12 variables
set.seed(1)
model.tr1 = tree(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                 hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
                 data=data.train.std.y)
# M10b: Best 20 variable model from regsubsets
set.seed(1)
model.tr1 = tree(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                 wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                 ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y)
# M10a: Default Original subset
set.seed(1)
model.tr1 <- tree(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data=data.train.std.y) # include additional terms on the fly using I()

pred.valid.tr1 <- predict(model.tr1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.tr1)^2) # mean prediction error
sd((y.valid - pred.valid.tr1)^2)/sqrt(n.valid.y) # std error
summary(model.tr1)

# Bagging
library(randomForest)
# M11c: Trimmed subset of 12 variables
set.seed(1)
model.bag1 = randomForest(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                 hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
                 data=data.train.std.y, mtry=15, ntree=500, importance=T)
# M11b: Best 20 variable model from regsubsets
set.seed(1)
model.bag1 = randomForest(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                 wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                 ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y,
                 mtry=20, ntree=500, importance=T)
# M11a: Default Original subset
set.seed(1)
model.bag1 <- randomForest(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data=data.train.std.y, mtry=20, ntree=500, importance=T)

pred.valid.bag1 <- predict(model.bag1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.bag1)^2) # mean prediction error
sd((y.valid - pred.valid.bag1)^2)/sqrt(n.valid.y) # std error
max(model.bag1$rsq)

# Random Forests
library(randomForest)
# M12c: Trimmed subset of 12 variables
set.seed(1)
model.rf1 = randomForest(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                         hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, 
                         data=data.train.std.y, mtry=4, ntree=500, importance=T)
# M12b: Best 20 variable model from regsubsets
set.seed(1)
model.rf1 = randomForest(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                         wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                         ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y,
                         mtry=4, ntree=500, importance=T)
# M12a: Default Original subset
set.seed(1)
model.rf1 <- randomForest(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                          ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                          data=data.train.std.y, mtry=4, ntree=500, importance=T)

pred.valid.rf1 <- predict(model.rf1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.rf1)^2) # mean prediction error
sd((y.valid - pred.valid.rf1)^2)/sqrt(n.valid.y) # std error
max(model.rf1$rsq)

# Boosting
library(gbm)
# M13c: Trimmed subset of 12 variables
set.seed(1)
model.boo1 = gbm(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 +
                hinc3 + ln_incm + ln_lgif + ln_tgif + ln_rgif + ln_agif, data=data.train.std.y, 
                 distribution="gaussian", n.trees=5000,
                 interaction.depth=4,shrinkage=0.01,verbose=F)
# M13b: Best 20 variable model from regsubsets
set.seed(1)
model.boo1 =  gbm(damt ~ reg3 + reg4 + reg5 + home + chld + hinc + plow + wrat + wrat2 + 
                  wrat3 + hinc3 + ln_incm + ln_plow + ln_tgif + ln_lgif + ln_rgif + 
                  ln_agif + sr_incm + sr_plow + sr_lgif, data=data.train.std.y, 
                  distribution="gaussian", n.trees=5000,
                  interaction.depth=4,shrinkage=0.01,verbose=F)
# M13a: Default Original subset
set.seed(1)
model.boo1 <-  gbm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   ln_avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                   data=data.train.std.y, distribution="gaussian", n.trees=5000,
                   interaction.depth=4,shrinkage=0.01,verbose=F)

pred.valid.boo1 <- predict(model.boo1, newdata = data.valid.std.y, n.trees=5000) # validation predictions
mean((y.valid - pred.valid.boo1)^2) # mean prediction error
sd((y.valid - pred.valid.boo1)^2)/sqrt(n.valid.y) # std error


# Colormap for classification models
library(gplots)
library(RColorBrewer)
d1 = read.csv("classmap2.csv", row.names = 1, header=T)
d2 = read.csv("classactual2.csv", row.names = 1, header=T)
rownames = rownames(d1)
#d1 = d1[,-c(16,17)]
#d2 = d2[,-c(16,17)]
d1 = as.matrix(d1)
d2 = as.matrix(d2)
# creates a own color palette from blue to red
my_palette <- colorRampPalette(c("blue","yellow","red"))(n = 299)
# (optional) defines the color breaks manually for a "skewed" color transition
col_breaks = c(seq(0,0.799,length=100),   # for blue
               seq(0.80,0.929,length=100),   # for yellow
               seq(0.93,1.0,length=100))   # for red
# creates a 5 x 5 inch image
png("classheatmap.png",     # create PNG for the heat map        
    width = 11*300,        # 5 x 300 pixels
    height = 17*300,
    res = 300,            # 300 pixels per inch
    pointsize = 10)        # smaller font size
heatmap.2(d1,
          cellnote = d2,        # same data set for cell labels
          main = "Classification Metrics", # heat map title
          notecol="black",      # change font color of cell labels to black
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          margins =c(10,8),     # widens margins around plot
          col=my_palette,       # use on color palette defined earlier
          breaks=col_breaks,    # enable color transition at specified limits
          dendrogram="none",    # only draw a row dendrogram
          Colv=FALSE,           # turn off column clustering
          Rowv=FALSE,           # turn off row clustering
          keysize=25,
          lmat=rbind(c(2),c(3),c(1),c(4)), 
          lhei=c(5,5,18,0), 
          lwid=c(1))            
dev.off()

# Colormap for prediction models
library(gplots)
library(RColorBrewer)
d3 = read.csv("predmap.csv", row.names = 1, header=T)
d4 = read.csv("predactual.csv", row.names = 1, header=T)
rownames = rownames(d3)
d3 = as.matrix(d3)
d4 = as.matrix(d4)
# creates a own color palette from blue to red
my_palette <- colorRampPalette(c("red","yellow","blue"))(n = 299)
# (optional) defines the color breaks manually for a "skewed" color transition
col_breaks = c(seq(0,0.03,length=100),   # for red
               seq(0.031,0.08,length=100),   # for yellow
               seq(0.081,1.0,length=100))   # for blue
# creates a 5 x 5 inch image
png("predheatmap.png",     # create PNG for the heat map        
    width = 2.2*300,        # 5 x 300 pixels
    height = 14*300,
    res = 300,            # 300 pixels per inch
    pointsize = 9)        # smaller font size
heatmap.2(d3,
          cellnote = d4,        # same data set for cell labels
          main = "Classification Metrics", # heat map title
          notecol="black",      # change font color of cell labels to black
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          margins =c(12,10),     # widens margins around plot
          col=my_palette,       # use on color palette defined earlier
          breaks=col_breaks,    # enable color transition at specified limits
          dendrogram="none",    # only draw a row dendrogram
          Colv=FALSE,           # turn off column clustering
          Rowv=FALSE,           # turn off row clustering
          keysize=25,
          lmat=rbind(c(2),c(3),c(1),c(4)), 
          lhei=c(5,5,18,0), 
          lwid=c(1))            
dev.off()

# select model.ls2 since it has minimum mean prediction error in the validation sample
model.best.y = model.ns1
yhat.test <- predict(model.best.y, newdata = data.test.std) # test predictions
# FINAL RESULTS
# Save final results for both classification and regression
length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt
ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="JP.csv", row.names=FALSE) # use your initials for the file name
# submit the csv file in Angel for evaluation based on actual test donr and damt values
