library(rpart)
library(rpart.plot)
library(ROCR)

# Step 1: Read in the Data
df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")  

# List the structure of the data (str)
str(df)

# Execute a summary of the data
summary(df)

# Print the first six records
head(df)

# Step 2: Classification Decision Tree
set.seed(3) # Ensures reproducibility

# Set the control parameters for the tree depth and method of splitting
control_parameters <- rpart.control(maxdepth = 10)

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Splitting criteria
FLAG_classification <- sample(c(TRUE, FALSE), nrow(df_classification), replace = TRUE, prob = c(0.7, 0.3))
df_train_classification <- df_classification[FLAG_classification, ]
df_test_classification <- df_classification[!FLAG_classification, ]

# Decision tree using Gini
treeGini <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Decision tree using Entropy
treeEntropy <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_classification, method = "class", parms = list(split = "information"), control = control_parameters)
rpart.plot(treeEntropy)
print(treeEntropy$variable.importance)

# Gini model on training data
predictionsGini_train <- predict(treeGini, df_train_classification, type = "prob")[,2]
rocGini_train <- prediction(predictionsGini_train, df_train_classification$TARGET_BAD_FLAG)
perfGini_train <- performance(rocGini_train, "tpr", "fpr")
aucGini_train <- performance(rocGini_train, measure = "auc")@y.values[[1]]
plot(perfGini_train, col = "red")
text(0.6, 0.4, paste("AUC Gini Train =", round(aucGini_train, 4)))
# Entropy model on training data
predictionsEntropy_train <- predict(treeEntropy, df_train_classification, type = "prob")[,2]
rocEntropy_train <- prediction(predictionsEntropy_train, df_train_classification$TARGET_BAD_FLAG)
perfEntropy_train <- performance(rocEntropy_train, "tpr", "fpr")
aucEntropy_train <- performance(rocEntropy_train, measure = "auc")@y.values[[1]]
plot(perfEntropy_train, col = "blue")
text(0.6, 0.4, paste("AUC Entropy Train =", round(aucEntropy_train, 4)))
# Gini model on testing data
predictionsGini_test <- predict(treeGini, df_test_classification, type = "prob")[,2]
rocGini_test <- prediction(predictionsGini_test, df_test_classification$TARGET_BAD_FLAG)
perfGini_test <- performance(rocGini_test, "tpr", "fpr")
aucGini_test <- performance(rocGini_test, measure = "auc")@y.values[[1]]
plot(perfGini_test, col = "red")
text(0.6, 0.4, paste("AUC Gini Test =", round(aucGini_test, 4)))
# Entropy model on testing data
predictionsEntropy_test <- predict(treeEntropy, df_test_classification, type = "prob")[,2]
rocEntropy_test <- prediction(predictionsEntropy_test, df_test_classification$TARGET_BAD_FLAG)
perfEntropy_test <- performance(rocEntropy_test, "tpr", "fpr")
aucEntropy_test <- performance(rocEntropy_test, measure = "auc")@y.values[[1]]
plot(perfEntropy_test, col = "blue")
text(0.6, 0.4, paste("AUC Entropy Test =", round(aucEntropy_test, 4)))
# Summary
# The classification decision trees developed to predict loan default using Gini and Entropy show strong performance, with AUC values for both training and testing datasets exceeding 0.8. This suggests that the models are effective at distinguishing between default and non-default cases and are likely neither overfit nor underfit, as indicated by their consistent AUC scores across training and testing sets. The two models perform similarly well, with no significant difference in AUC values. However, the Gini method has a slight edge with marginally higher AUC scores in both training and testing datasets. This could imply that the Gini model has better predictive capability, though the difference is minimal. 


# Step 3: Regression Decision Tree
set.seed(3) # Ensures reproducibility

# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Splitting criteria
FLAG_regression <- sample(c(TRUE, FALSE), nrow(df_regression), replace = TRUE, prob = c(0.7, 0.3))
df_train_regression <- df_regression[FLAG_regression, ]
df_test_regression <- df_regression[!FLAG_regression, ]

# Decision tree using ANOVA
treeAnova <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_regression, method = "anova", control = control_parameters)
rpart.plot(treeAnova)
print(treeAnova$variable.importance)

# Decision tree using Poisson
treePoisson <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_regression, method = "poisson", control = control_parameters)
rpart.plot(treePoisson)
print(treePoisson$variable.importance)

# Calculate RMSE for the ANOVA model using training data set
predictionsAnova_train <- predict(treeAnova, df_train_regression)
RMSE_Anova_train <- sqrt(mean((df_train_regression$TARGET_LOSS_AMT - predictionsAnova_train)^2))
print(RMSE_Anova_train)

# Calculate RMSE for the Poisson model using training data set
predictionsPoisson_train <- predict(treePoisson, df_train_regression)
RMSE_Poisson_train <- sqrt(mean((df_train_regression$TARGET_LOSS_AMT - predictionsPoisson_train)^2))
print(RMSE_Poisson_train)

# Calculate RMSE for the ANOVA model using testing data set
predictionsAnova_test <- predict(treeAnova, df_test_regression)
RMSE_Anova_test <- sqrt(mean((df_test_regression$TARGET_LOSS_AMT - predictionsAnova_test)^2))
print(RMSE_Anova_test)

# Calculate RMSE for the Poisson model using testing data set
predictionsPoisson_test <- predict(treePoisson, df_test_regression)
RMSE_Poisson_test <- sqrt(mean((df_test_regression$TARGET_LOSS_AMT - predictionsPoisson_test)^2))
print(RMSE_Poisson_test)

# Summary
# The regression decision trees constructed to predict TARGET_LOSS_AMT reveal that the ANOVA model slightly outperforms the Poisson model, as indicated by lower RMSE values on both training and testing data sets. The RMSE for the ANOVA model is 4824.357 for training and 5263.02 for testing, compared to 5339.834 for training and 5410.932 for testing for the Poisson model. The modest increase in RMSE from training to testing in the ANOVA model suggests it is neither significantly overfit nor underfit, and it generalizes better to unseen data than the Poisson model. The differences in RMSE are not substantial, but they lean in favor of the ANOVA model, indicating it may be more accurate for predicting potential losses on defaulted loans.

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

# Re-train the classification tree using Gini for TARGET_BAD_FLAG (probability of default)
treeGini2 <- rpart(TARGET_BAD_FLAG ~ ., data = df_train_excluded, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini2)
print(treeGini2$variable.importance)

# Fit the regression tree for TARGET_LOSS_AMT on df_train_defaults
treeLoss <- rpart(TARGET_LOSS_AMT ~ ., data = df_train_defaults, method = "anova", control = control_parameters)
rpart.plot(treeLoss)
print(treeLoss$variable.importance)

# Use the trained classification model (treeGini2) to predict the probability of default
prob_default_train <- predict(treeGini2, df_train, type = "prob")[,2]
prob_default_test <- predict(treeGini2, df_test, type = "prob")[,2]

# Predict the loss given default using the regression tree
loss_given_default_train <- predict(treeLoss, df_train_defaults)
loss_given_default_test <- predict(treeLoss, df_test_defaults)

# Combine the probability of default with the loss given default to get the expected loss
expected_loss_train <- prob_default_train[df_train$TARGET_BAD_FLAG == 1] * loss_given_default_train
expected_loss_test <- prob_default_test[df_test$TARGET_BAD_FLAG == 1] * loss_given_default_test

# Calculate RMSE for the Probability / Severity model on the training and testing data
RMSE_ProbSeverity_train <- sqrt(mean((df_train_defaults$TARGET_LOSS_AMT - expected_loss_train)^2))
RMSE_ProbSeverity_test <- sqrt(mean((df_test_defaults$TARGET_LOSS_AMT - expected_loss_test)^2))

# Print out the RMSE values
print(paste("RMSE for Probability / Severity model on training data:", RMSE_ProbSeverity_train))
print(paste("RMSE for Probability / Severity model on testing data:", RMSE_ProbSeverity_test))

# Summary
# The Probability/Severity model from Step 4 offers a holistic view of risk by assessing both the likelihood and impact of loan defaults, unlike the direct loss prediction models in Step 3. Despite its higher RMSE, indicating less precision in predicting losses, it provides valuable insights into both default probability and potential loss severity. This comprehensive approach is preferable for risk management, as it balances the incidence and magnitude of risks, making it the recommended choice for a fuller risk assessment.