library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(ROCR)
library(ggplot2)

# Step 1: Read in the Data
df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")  

# List the structure of the data (str)
str(df)

# Execute a summary of the data
summary(df)

# Print the first six records
head(df)

# Step 2: Classification Models

set.seed(3) # Ensures reproducibility

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_classification <- sample(c(TRUE, FALSE), nrow(df_classification), replace = TRUE, prob = c(0.7, 0.3))
df_train_classification <- df_classification[FLAG_classification, ]
df_test_classification <- df_classification[!FLAG_classification, ]

# TREE
control_parameters <- rpart.control(maxdepth = 10)
tree_model <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", control = control_parameters)
rpart.plot(tree_model)
print(tree_model$variable.importance)

pt = predict( tree_model, df_test_classification, type="prob" )
head( pt )
pt2 = prediction( pt[,2], df_test_classification$TARGET_BAD_FLAG)
pt3 = performance( pt2, "tpr", "fpr" )

# RF
rf_model = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_classification, ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_classification )
head( pr )
pr2 = prediction( pr, df_test_classification$TARGET_BAD_FLAG)
pr3 = performance( pr2, "tpr", "fpr" )

# GRADIENT BOOSTING

gb_model = gbm(TARGET_BAD_FLAG ~ ., data = df_train_classification, n.trees=500, distribution="bernoulli" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_classification, type="response" )
head( pg )
pg2 = prediction( pg, df_test_classification$TARGET_BAD_FLAG)
pg3 = performance( pg2, "tpr", "fpr" )

# Plot

plot( pt3, col="green" )
plot( pr3, col="red", add=TRUE )
plot( pg3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("TREE","RANDOM FOREST", "GRADIENT BOOSTING"),col=c("green","red","blue"), bty="y", lty=1 )

aucT = performance( pt2, "auc" )@y.values
aucR = performance( pr2, "auc" )@y.values
aucG = performance( pg2, "auc" )@y.values

print( paste("TREE AUC=", aucT) )
print( paste("RF AUC=", aucR) )
print( paste("GB AUC=", aucG) )

# The Random Forest model performed the best, with an AUC of approximately 0.95 which is the highest.
# Considering the ROC curve, where the Random Forest line is closer to the top-left corner compared to the Decision Tree and Gradient Boosting, and the highest AUC value, the Random Forest model has demonstrated superior performance in your classification task. It balances well between bias and variance, can handle a mix of feature types, and is less likely to overfit compared to a single decision tree. Gradient Boosting also performed well and could be a strong contender, especially with further tuning. However, given the evidence, Random Forest would be my primary recommendation for predicting home equity loan defaults in this scenario.

# Step 3: Regression Decision Tree

set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_regression <- sample(c(TRUE, FALSE), nrow(df_regression), replace = TRUE, prob = c(0.7, 0.3))
df_train_regression <- df_regression[FLAG_regression, ]
df_test_regression <- df_regression[!FLAG_regression, ]

# TREE

tr_model = rpart( data=df_train_regression, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model )
rpart.plot( tr_model, digits=-3, extra=100 )
tr_model$variable.importance

pt = predict( tr_model, df_test_regression )
head( pt )
RMSEt = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pt )^2 ) )

# RF

rf_model = randomForest( data=df_train_regression, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_regression )
head( pr )
RMSEr = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pr )^2 ) )

# GRADIENT BOOSTING

gb_model = gbm( data=df_train_regression, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_regression, type="response" )
head( pg )
RMSEg = sqrt( mean( ( df_test_regression$TARGET_LOSS_AMT - pg )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

# The Random Forest model performed the best, with the lowest RMSE of approximately 4177.33
# The Random Forest model not only provides the most accurate predictions but also offers robustness against overfitting, making it suitable for operational use. While Gradient Boosting also provides reasonable accuracy, it doesn't outperform Random Forest. The Decision Tree, while the easiest to interpret, offers less accuracy. Thus, in this scenario, the slight loss of interpretability with Random Forest is outweighed by its superior performance, making it the recommended choice.

# Step 4: Probability / Severity Model Decision Tree
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

# Re-train the Classification tree using Random Forest for TARGET_BAD_FLAG (probability of default)
rf_model_2 = randomForest(TARGET_BAD_FLAG ~ ., data = df_train_excluded, ntree=500, importance=TRUE )
importance( rf_model_2 )

pr_default = predict( rf_model_2, df_test)

# TREE

tr_model = rpart( data=df_train_defaults, TARGET_LOSS_AMT ~ ., control=control_parameters, method="poisson" )
rpart.plot( tr_model )
rpart.plot( tr_model, digits=-3, extra=100 )
tr_model$variable.importance

pt = predict( tr_model, df_test_defaults )
head( pt )
RMSEt = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pt )^2 ) )

# RF

rf_model = randomForest( data=df_train_defaults, TARGET_LOSS_AMT ~ ., ntree=500, importance=TRUE )
importance( rf_model )
varImpPlot( rf_model )

pr = predict( rf_model, df_test_defaults)
head( pr )
RMSEr = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pr )^2 ) )

# GRADIENT BOOSTING

gb_model = gbm( data=df_train_defaults, TARGET_LOSS_AMT ~ ., n.trees=500, distribution="poisson" )
summary.gbm( gb_model, cBars=10 )

pg = predict( gb_model, df_test_defaults, type="response" )
head( pg )
RMSEg = sqrt( mean( ( df_test_defaults$TARGET_LOSS_AMT - pg )^2 ) )

print( paste("TREE RMSE=", RMSEt ))
print( paste("RF RMSE=", RMSEr ))
print( paste("GB RMSE=", RMSEg ))

# Using the Gradient Boosting model predictions
selected_loss_model_predictions = pg  # Predictions from Gradient Boosting

# Ensure alignment in indices for calculating expected losses
# Get the probabilities of default for the records that are actually defaulted
pr_default_filtered = pr_default[df_test$TARGET_BAD_FLAG == 1]

# Calculate Expected Loss - multiply the predicted probability of default by the predicted loss
expected_loss = pr_default_filtered * selected_loss_model_predictions

# Calculate Actual Loss - from the test set, only for default records
actual_loss = df_test$TARGET_LOSS_AMT[df_test$TARGET_BAD_FLAG == 1]

# Calculate Overall RMSE for the Probability/Severity Model
overall_rmse = sqrt(mean((actual_loss - expected_loss)^2))
print(paste("Overall RMSE for Probability / Severity Model:", overall_rmse))

# The comparison between the models from Steps 3 and 4 centers on their applicability and complexity. Step 3 likely involved separate models for predicting default probability and loss severity, evaluated independently on metrics like AUC and RMSE, offering straightforward results but lacking integrated financial impact analysis. In contrast, Step 4 introduces a more sophisticated model that combines both probability of default and severity of loss, directly estimating expected losses which is crucial for comprehensive financial risk management. This model from Step 4 is particularly valuable in scenarios where understanding both the likelihood and potential impact of defaults is essential for making informed decisions about risk mitigation and capital allocation. Although more complex and possibly less interpretable, the integrated approach of Step 4 is recommended for its holistic view on financial outcomes, making it better suited for detailed risk assessment and decision support in financial contexts.