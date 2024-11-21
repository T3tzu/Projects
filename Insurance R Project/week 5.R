# Step 1
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(ROCR)
library(ggplot2)
library( MASS )

df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")  

# List the structure of the data (str)
str(df)

# Execute a summary of the data
summary(df)

# Print the first six records
head(df)

###### Step 2: Classification Models ######

set.seed(3) # Ensures reproducibility

# Exclude TARGET_LOSS_AMT from the predictors
df_FLAG <- df
df_FLAG$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_FLAG <- sample(c(TRUE, FALSE), nrow(df_FLAG), replace = TRUE, prob = c(0.7, 0.3))
df_train_FLAG <- df_FLAG[FLAG_FLAG, ]
df_test_FLAG <- df_FLAG[!FLAG_FLAG, ]

# TREE
control_parameters <- rpart.control(maxdepth = 10)
tree_model_FLAG <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, method = "class", control = control_parameters)
rpart.plot(tree_model_FLAG)
print(tree_model_FLAG$variable.importance)

pt_FLAG = predict( tree_model_FLAG, df_test_FLAG, type="prob" )
head( pt_FLAG )
pt2_FLAG = prediction( pt_FLAG[,2], df_test_FLAG$TARGET_BAD_FLAG)
pt3_FLAG = performance( pt2_FLAG, "tpr", "fpr" )

# RF
rf_model_FLAG = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, ntree=500, importance=TRUE )
importance( rf_model_FLAG )
varImpPlot( rf_model_FLAG )

pr_FLAG = predict( rf_model_FLAG, df_test_FLAG )
head( pr_FLAG )
pr2_FLAG = prediction( pr_FLAG, df_test_FLAG$TARGET_BAD_FLAG)
pr3_FLAG = performance( pr2_FLAG, "tpr", "fpr" )

# GRADIENT BOOSTING

gb_model_FLAG = gbm(TARGET_BAD_FLAG ~ ., data = df_train_FLAG, n.trees=500, distribution="bernoulli" )
summary.gbm( gb_model_FLAG, cBars=10 )

pg_FLAG = predict( gb_model_FLAG, df_test_FLAG, type="response" )
head( pg_FLAG )
pg2_FLAG = prediction( pg_FLAG, df_test_FLAG$TARGET_BAD_FLAG)
pg3_FLAG = performance( pg2_FLAG, "tpr", "fpr" )

# LOGISTIC REGRESSION using ALL the variables and FORWARD VARIABLE SELECTION

theUpper_LR_FLAG = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_train_FLAG )
theLower_LR_FLAG = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_train_FLAG )

summary( theUpper_LR_FLAG )
summary( theLower_LR_FLAG )

lr_model_FLAG = stepAIC(theLower_LR_FLAG, direction="forward", scope=list(lower=theLower_LR_FLAG, upper=theUpper_LR_FLAG))
summary( lr_model_FLAG )

plr_FLAG = predict( lr_model_FLAG, df_test_FLAG, type="response" )
plr2_FLAG = prediction( plr_FLAG, df_test_FLAG$TARGET_BAD_FLAG )
plr3_FLAG = performance( plr2_FLAG, "tpr", "fpr" )

plot( plr3_FLAG, col="gold" )
abline(0,1,lty=2)
legend("bottomright",c("LOGISTIC REGRESSION FWD"),col=c("gold"), bty="y", lty=1 )

# LR STEP TREE
treeVars_FLAG = tree_model_FLAG$variable.importance
treeVars_FLAG = names(treeVars_FLAG)
treeVarsPlus_FLAG = paste( treeVars_FLAG, collapse="+")
F = as.formula( paste( "TARGET_BAD_FLAG ~", treeVarsPlus_FLAG ))

tree_LR_FLAG = glm( F, family = "binomial", data=df_train_FLAG )
theLower_LR_FLAG = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_train_FLAG )

summary( tree_LR_FLAG )
summary( theLower_LR_FLAG )

lrt_model_FLAG = stepAIC(theLower_LR_FLAG, direction="both", scope=list(lower=theLower_LR_FLAG, upper=tree_LR_FLAG))
summary( lrt_model_FLAG )

plrt_FLAG = predict( lrt_model_FLAG, df_test_FLAG, type="response" )
plrt2_FLAG = prediction( plrt_FLAG, df_test_FLAG$TARGET_BAD_FLAG )
plrt3_FLAG = performance( plrt2_FLAG, "tpr", "fpr" )

plot( plrt3_FLAG, col="gray" )
abline(0,1,lty=2)
legend("bottomright",c("LOGISTIC REGRESSION TREE"),col=c("gray"), bty="y", lty=1 )

plot( pt3_FLAG, col="green" )
plot( pr3_FLAG, col="red", add=TRUE )
plot( pg3_FLAG, col="blue", add=TRUE )
plot( plr3_FLAG, col="gold", add=TRUE ) 
plot( plrt3_FLAG, col="gray", add=TRUE ) 

abline(0,1,lty=2)
legend("bottomright",c("TREE","RANDOM FOREST", "GRADIENT BOOSTING", "LOGIT REG FWD", "LOGIT REG TREE"),col=c("green","red","blue","gold","gray"), bty="y", lty=1 )

aucT = performance( pt2_FLAG, "auc" )@y.values
aucR = performance( pr2_FLAG, "auc" )@y.values
aucG = performance( pg2_FLAG, "auc" )@y.values
aucLR = performance( plr2_FLAG, "auc")@y.values
aucLRT = performance( plrt2_FLAG, "auc")@y.values

print( paste("TREE AUC=", aucT) )
print( paste("RF AUC=", aucR) )
print( paste("GB AUC=", aucG) )
print( paste("LR AUC=", aucLR) )
print( paste("LRT AUC=", aucLRT) )

# The Random Forest model performed best due to its highest AUC value, indicating superior ability to differentiate between the classes. Despite being less interpretable and more computationally demanding, I would recommend it for scenarios where accuracy is the top priority. If ease of interpretation or computational resources were major concerns, a simpler model like logistic regression could be considered.

###### Step 3: Linear Regression ######

set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_AMT <- df
df_AMT$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_AMT <- sample(c(TRUE, FALSE), nrow(df_AMT), replace = TRUE, prob = c(0.7, 0.3))
df_train_AMT <- df_AMT[FLAG_AMT, ]
df_test_AMT <- df_AMT[!FLAG_AMT, ]

# TREE

tr_model_AMT = rpart( data=df_train_AMT, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model_AMT )
rpart.plot( tr_model_AMT, digits=-3, extra=100 )
tr_model_AMT$variable.importance

pt_AMT = predict( tr_model_AMT, df_test_AMT )
head( pt_AMT )
RMSEt = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pt_AMT )^2 ) )

# RF

rf_model_AMT = randomForest( data=df_train_AMT, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model_AMT )
varImpPlot( rf_model_AMT )

pr_AMT = predict( rf_model_AMT, df_test_AMT )
head( pr_AMT )
RMSEr = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pr_AMT )^2 ) )

# GRADIENT BOOSTING

gb_model_AMT = gbm( data=df_train_AMT, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model_AMT, cBars=10 )

pg_AMT = predict( gb_model_AMT, df_test_AMT, type="response" )
head( pg_AMT )
RMSEg = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - pg_AMT )^2 ) )

# LINEAR REGRESSION using ALL the variable and BACKWARD VARIABLE SELECTION

theUpper_LR_AMT = lm( TARGET_LOSS_AMT ~ ., data=df_train_AMT )
theLower_LR_AMT = lm( TARGET_LOSS_AMT ~ 1, data=df_train_AMT )

summary( theUpper_LR_AMT )
summary( theLower_LR_AMT )

lr_model_AMT = stepAIC(theUpper_LR_AMT, direction="backward", scope=list(lower=theLower_LR_AMT, upper=theUpper_LR_AMT))
summary( lr_model_AMT )

plr_AMT = predict( lr_model_AMT, df_test_AMT )
head( plr_AMT )
RMSElr = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_AMT )^2 ) )

# LR STEP TREE
treeVars_AMT = tr_model_AMT$variable.importance
treeVars_AMT = names(treeVars_AMT)
treeVarsPlus_AMT = paste( treeVars_AMT, collapse="+")
F = as.formula( paste( "TARGET_LOSS_AMT ~", treeVarsPlus_AMT ))

tree_LR_AMT = lm( F, data=df_train_AMT )
theLower_LR_AMT = lm( TARGET_LOSS_AMT ~ 1, data=df_train_AMT )

summary( tree_LR_AMT )
summary( theLower_LR_AMT )

lrt_model_AMT = stepAIC(theLower_LR_AMT, direction="both", scope=list(lower=theLower_LR_AMT, upper=tree_LR_AMT))
summary( lrt_model_AMT )


plr_tree_AMT = predict( tree_LR_AMT, df_test_AMT )
head( plr_tree_AMT )
RMSElr_tree = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_tree_AMT )^2 ) )

plr_tree_step_AMT = predict( lrt_model_AMT, df_test_AMT )
head( plr_tree_step_AMT )
RMSElr_tree_step = sqrt( mean( ( df_test_AMT$TARGET_LOSS_AMT - plr_tree_step_AMT )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

print( paste("LR BACK RMSE=",  RMSElr ))
print( paste("LR TREE RMSE=",  RMSElr_tree ))
print( paste("LR TREE STEP RMSE=", RMSElr_tree_step ))

# The Random Forest model had the lowest RMSE, indicating it was the most accurate for this dataset. I'd recommend it for scenarios where accuracy is crucial. If you need a faster or more interpretable model, Linear Regression could be a simpler but less accurate alternative.

###### Step 4: Probability / Severity Model ######
set.seed(3) # Ensuring reproducibility

# Split the full data into training and testing sets
FLAG <- sample(c(TRUE, FALSE), nrow(df), replace = TRUE, prob = c(0.7, 0.3))
df_train <- df[FLAG, ]
df_test <- df[!FLAG, ]

# Exclude TARGET_LOSS_AMT for the classification model training
df_train_excluded <- df_train
df_train_excluded$TARGET_LOSS_AMT <- NULL

# Filter the datasets to include only the instances with a default
df_train_defaults <- df_train[df_train$TARGET_BAD_FLAG == 1, ]
df_test_defaults <- df_test[df_test$TARGET_BAD_FLAG == 1, ]

# Logistic Regression with Forward Selection
theUpper_LR = glm(TARGET_BAD_FLAG ~ ., family = "binomial", data = df_train_excluded)
theLower_LR = glm(TARGET_BAD_FLAG ~ 1, family = "binomial", data = df_train_excluded)
logit_model = stepAIC(theLower_LR, direction = "forward", scope = list(lower = theLower_LR, upper = theUpper_LR))
summary(logit_model)

# Linear Regression with Backward Selection
theUpper_LM = lm(TARGET_LOSS_AMT ~ ., data = df_train_defaults)
theLower_LM = lm(TARGET_LOSS_AMT ~ 1, data = df_train_defaults)
lm_model = stepAIC(theUpper_LM, direction = "backward", scope = list(lower = theLower_LM, upper = theUpper_LM))
summary(lm_model)

# Predictions
prob_default = predict(logit_model, df_test, type = "response")
loss_given_default = predict(lm_model, df_test_defaults, type = "response")

# Calculate Expected Loss
expected_loss = df_test$TARGET_BAD_FLAG * prob_default * loss_given_default

# Actual Loss - assume TARGET_LOSS_AMT is 0 where TARGET_BAD_FLAG is 0
actual_loss = ifelse(df_test$TARGET_BAD_FLAG == 1, df_test$TARGET_LOSS_AMT, 0)

# RMSE Calculation
RMSE = sqrt(mean((actual_loss - expected_loss)^2))
print(paste("RMSE for Probability / Severity Model =", RMSE))

# I would recommend using the Random Forest model from Step 3 over the Probability / Severity Model from Step 4 for applications where prediction accuracy is paramount. However, if the context requires assessing not only the likelihood but also the severity of outcomes, adjustments or enhancements to the Probability / Severity Model might be necessary to improve its predictive performance.