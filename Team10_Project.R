#Team Project - Predicting Repeat Buyers
#Team 10
# Adam Jiang, Sharath Pingula, Antoine Rigaut, Ruisi Xiong
#---------------------------------------------------------
# Set directory
setwd('/Users/homw/Documents/MSDS16/linearModels/project/')
library(glmnet)
library(boot)
library(ggplot2)
library(ROCR)
library(randomForest)
library(dplyr)

# Supporting functions 

plot.ROC.curve <- function(probs, labels){
  preds <- prediction(probs, labels)
  perf <- performance(preds, measure = "tpr", x.measure = "fpr")
  auc <- performance(preds, measure = "auc")
  auc <- auc@y.values[[1]]
  
  roc.data <- data.frame(fpr=unlist(perf@x.values),
                         tpr=unlist(perf@y.values),
                         model="GLM")
  ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
    geom_ribbon(alpha=0.2) +
    geom_line(aes(y=tpr)) +
    ggtitle(paste0("ROC Curve w/ AUC=", auc))
}

# measure accuracy
get.accuracy <- function(pred, labels){
  sum(pred == labels) / length(pred)
}

# measure recall
get.recall <- function(pred, labels){
  sum(pred == 1 & pred == labels) / sum(labels == 1)
}

# get precision
get.precision <- function(pred, labels){
  sum(pred == 1 & pred == labels) / sum(pred == 1)
}

# Load the processed dataset
transaction <- read.csv("project_aggregated_ds.csv", stringsAsFactors = F) 
# Missing values in this dataset are not really missing values. They are zero values. 
transaction[is.na(transaction)] <- 0
# Get rid of negative records
transaction <- transaction[apply(transaction,1, function(x) any(x<0) == F),] # 305 customers with negative values

# We want to add the department of the product coupon as a categorical variable 
# Load offers
offers <- read.csv('offers.csv', header = T)
# Department is the first two digits of category (if category is 4-digit) or the first digit (if category is 3-digit)
offers$dept <- as.factor(sapply(offers$category, function(x)ifelse(nchar(x) > 3, substr(x, 1, 2), substr(x, 1, 1))))

# Load trainHistory
trainHistory <- read.csv("trainHistory.csv")
# Merge to find for each customer id the department of the offer
trainHistory <- merge(trainHistory[,c("id", "offer")], offers[,c("offer", "dept")], by="offer", all.x=T)
names(trainHistory)[2] <- "ID"

# Now merge the transaction dataset with the obtained department variable
transaction <- merge(transaction, trainHistory[,c("ID", "dept")], by="ID", all.x=T)
transaction[, "dept"] <- factor(transaction$dept) # Remove unused factor level by simply calling factor on the var

# Get rid of customer id and offer quantity. We will not make use of these variables in the model
# transaction <- transaction[,names(transaction)[!(names(transaction) %in% c("offer_quantity", "ID"))]]

# Transform into factor variable
transaction$never_bought_brand <- factor(transaction$never_bought_brand)
transaction$never_bought_category <- factor(transaction$never_bought_category)
transaction$never_bought_company <- factor(transaction$never_bought_company)
transaction$has_bought_brand_category <- factor(transaction$has_bought_brand_category)
transaction$has_bought_brand_company <- factor(transaction$has_bought_brand_company)
transaction$has_bought_brand_company_category <- factor(transaction$has_bought_brand_company_category)

# Removal of outliers
boxplot(transaction$total_spend, ylab="total_spend") # % customers with more than 1mil spent
a <- boxplot(transaction$total_spend, ylab="total_spend") 
transaction <- transaction[transaction$total_spend < a$stats[5],] # Remove the outliers



#  Simple linear model w/o log transformation on spend values
# --------------------
set.seed(1)
fit.data <- transaction[,!(names(transaction) %in% c('ID',"offer_quantity"))]
ind <- sample(1:nrow(fit.data), nrow(fit.data) * 0.7)
train <- fit.data[ind,]
test <- fit.data[-ind,]
fit <- glm(label~., data=train, family = "binomial")
summary(fit) # Note that the department variable seems to play a significant role

# Summary performance metrics
prob <- predict(fit, newdata = test, type = 'response')
pred <- ifelse(prob>0.3,1,0)
table(pred, test$label)
get.recall(pred, test$label) # Only 54% recall
get.accuracy(pred, test$label) # 76% accuracy
get.precision(pred, test$label) # 63% precision. It seems the classifier is conservative about classifying an 
# observation as a repeater
plot.ROC.curve(prob, test$label) # 0.703 in AUC (updates in very iteration)

# Lasso regularization w/o log transformations
dm <- model.matrix(label~., data=fit.data)[,-1]
set.seed(1)
ind <- sample(1:nrow(fit.data), nrow(fit.data) * 0.7)
train <- dm[ind,]
test <- dm[-ind,]
cv.lam <- cv.glmnet(train, factor(transaction[ind, "label"]), alpha=1, family="binomial", type.measure = "class")
plot(cv.lam)
bestlam <- cv.lam$lambda.min # best lambda as selected by cross validation
trainll <- glmnet(train, factor(transaction[ind, "label"]), alpha=1, family="binomial")
probs <- predict(trainll, newx = test, s = bestlam, type="response")
plot.ROC.curve(probs, transaction[-ind, "label"]) # 0.710

# Adding chains and markets
cust.offers <- read.csv('trainHistory.csv', header = T)
transaction <- inner_join(transaction,cust.offers, by = c("ID"="id"))
transaction <- transaction[,!(names(transaction) %in% c('ID',
                                                        'repeater',
                                                        'repeattrips',
                                                        'offerdate',
                                                        'offer_quantity',
                                                        'offer'))]

#create chain levels
unique(transaction$chain)
t <- table(transaction$chain)
th <- t[t>1500]
tm <- t[t<=1500 & t>=500]
tl <- t[t<500]

names(th)
transaction$f_chain <- as.factor(transaction$chain)
i=1
for (i in 1:nrow(transaction)){
  if (transaction$f_chain[i] %in% as.factor(names(th))) {transaction$ff_chain[i] = 'high'}
  if (transaction$f_chain[i] %in% as.factor(names(tm))) {transaction$ff_chain[i] = 'medium'}
  if (transaction$f_chain[i] %in% as.factor(names(tl))) {transaction$ff_chain[i] = 'low'}
}
table(transaction$ff_chain)

# There seems to be high collinearity between variables with purchase behaviour in 180, 90, 60 and 30 days 
# Selecting variables related to only recent 30days and total purchase

# Log transformation of the non-binary or factor variables
bin <- c("has_bought_brand_category", 
         "has_bought_brand_company_category", 
         "has_bought_brand_company", 
         "label", 
         "never_bought_category", 
         "never_bought_brand", 
         "never_bought_company", 
         "dept","chain","ff_chain","market","offer_value","f_chain")
transaction.log <- transaction
transaction.log[,!(names(transaction.log) %in% bin)] <- sapply(transaction[,!(names(transaction) %in% bin)], function(x)log(x+1))
fit.data <- transaction[,(names(transaction) %in% 
                            c('has_bought_brand',
                              'has_bought_brand_a','has_bought_brand_a_30',
                              'has_bought_brand_company','has_bought_brand_q',
                              'has_bought_company',
                              'has_bought_company_a',
                              'has_bought_company_a_30','has_bought_company_q',
                              'never_bought_category',
                              'offer_value','total_spend',
                              'has_bought_brand_category',
                              'has_bought_category',
                              'has_bought_category_a','has_bought_category_a_30',
                              'has_bought_category_q',
                              'never_bought_brand','never_bought_company',
                              'chain','market','label','dept'))]

#Try with chain, market, offer value as factors
fit.data$market <- as.factor(fit.data$market) #Convert market to factor type
fit.data$offer_value <- as.factor(fit.data$offer_value)

set.seed(10)
ind <- sample(1:nrow(fit.data), nrow(fit.data) * 0.7)
train <- fit.data[ind,]
test <- fit.data[-ind,]
fit <- glm(label~., data=train, family = "binomial")
summary(fit) # Note that the department variable seems to play a significant role

# Summary performance metrics
prob <- predict(fit, newdata = test, type = 'response')
plot.ROC.curve(prob, test$label) # 0.708 AUC
pred <- predict(fit, newdata = fit.data, type = 'response')

# prediction matrix
pred.bin <- ifelse(pred<0.25,0,1)
table(pred.bin, fit.data$label)

#Factor importance
set.seed(1)
ind <- sample(1:nrow(fit.data), nrow(fit.data) * 0.7)
train <- fit.data[ind,]
test <- fit.data[-ind,]
rf.fit1 <- randomForest(x=train[,-10],y = as.factor(train$label), ntree = 30,mtry = 10,do.trace = TRUE,replace = TRUE) #
#names(train)[10]

pred.rf <- predict(rf.fit,test[,-10], type='prob')
#roc(test$label, pred.rf[,2])
plot.ROC.curve(pred.rf[,2], test$label)

#Variable importance plots
importance(rf.fit)
varImpPlot(rf.fit, main = "Feature Importance", col="dark blue", cex = 0.9, pch = 18)


#---No improvement from below --------
#--- Check if removing outliers is increasing any prediction accuracy
#Handling Outliers and Collinearity
cd <- cooks.distance(fit)
rstud <- rstudent(fit)
rstud[abs(rstud)>3]
outliers <- order(abs(rstud), decreasing = T)[1:10]
high.cd <- order(cd,decreasing = T)[1:50]
cd[cd>10*mean(cd)]

#Remove records with high residual values and high cook distances
train <- train[-outliers,]
train <- train[-high.cd,]
fit <- glm(label~., data=train, family = "binomial")
summary(fit)
prob <- predict(fit, newdata = test, type = 'response')
plot.ROC.curve(prob, test$label) # 0.713 in AUC

#Residual plot check
train$pred <- fit$fitted.values
t <- train[,c('total_spend','label','pred')]
plot(predict(fit),residuals(fit),col=c("blue","red")[1+t$label],
     main = 'Residual Plot')
abline(h=0,lty=2,col="grey")
predict(fit)[1:10]
plot(t$pred,residuals(fit),col=c("blue","red")[1+t$label],
     main = 'Residual Plot', xlab = 'Predicetd Probs')
plot(transaction[,c(21:26)])

#Plots------------------------
ggplot(transaction, aes(x=as.factor(offer), fill = repeater))+
  geom_bar(stat = "bin")+
  xlab("Offers")+
  ylab("No of Customers offered")+
  ggtitle("How many repeaters for Offers?")

#Collinearity plots
t<- transaction[sample(1:nrow(transaction), 5000),]
plot(t[,c(23:27)])

repeater <- as.factor(fit.data$label)
#Dept Vs repeat buyers
ggplot(fit.data, aes(x = log(total_spend+1), y = pred)) + 
  geom_point(aes(colour = repeater),size = 1)+
  ylab('Predicted probability')+
  xlab('log of Total Spend')+
  ggtitle('Predicted probability Vs Total Spend')


#Box plot
ggplot(transaction, 
       aes(y = log(total_spend+1), x = repeater))+
  geom_boxplot(colour = c(blue,red))+
  ylab("log of Total Spend")+
  xlab("Repeat Buyer")+
  ggtitle("Total Spend Vs Repeater")+
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))
  
