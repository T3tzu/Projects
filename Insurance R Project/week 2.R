library(rpart)
library(rpart.plot)
library(ROCR)

# Step 1: Read in the Data
# Read the data into R
df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")

# List the structure of the data (str)
str(df)

# Execute a summary of the data
summary(df)

# Print the first six records
head(df)

# Step 2: Classification Decision Tree
# Set the control parameters for the tree depth and method of splitting
control_parameters <- rpart.control(maxdepth = 10)

# Exclude TARGET_LOSS_AMT from the predictors
df_classification <- df
df_classification$TARGET_LOSS_AMT <- NULL

# Decision tree using Gini
treeGini <- rpart(TARGET_BAD_FLAG ~ ., data = df_classification, method = "class", parms = list(split = "gini"), control = control_parameters)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Decision tree using Entropy
treeEntropy <- rpart(TARGET_BAD_FLAG ~ ., data = df_classification, method = "class", parms = list(split = "information"), control = control_parameters)
rpart.plot(treeEntropy)
print(treeEntropy$variable.importance)

# Creating ROC curve and calculating AUC for the Gini model
predictionsGini <- predict(treeGini, df_classification, type = "prob")[,2]
rocGini <- prediction(predictionsGini, df_classification$TARGET_BAD_FLAG)
perfGini <- performance(rocGini, "tpr", "fpr")
aucGini <- performance(rocGini, measure = "auc")@y.values[[1]]
plot(perfGini, col = "red")
abline(0, 1, lty = 2)
text(0.6, 0.4, paste("AUC Gini =", round(aucGini, 4)))

# Creating ROC curve and calculating AUC for the Entropy model
predictionsEntropy <- predict(treeEntropy, df_classification, type = "prob")[,2]
rocEntropy <- prediction(predictionsEntropy, df_classification$TARGET_BAD_FLAG)
perfEntropy <- performance(rocEntropy, "tpr", "fpr")
aucEntropy <- performance(rocEntropy, measure = "auc")@y.values[[1]]
plot(perfEntropy, col = "blue")
abline(0, 1, lty = 2)
text(0.6, 0.4, paste("AUC Entropy =", round(aucEntropy, 4)))

# Summary and recommendation
#The decision trees for predicting loan default were built using Gini and Entropy criteria. The Gini tree, with a slightly higher AUC, is the preferred model as it better differentiates between defaults. Key predictors for default include missing debt-to-income information and past delinquency, suggesting that individuals with incomplete financial profiles or a history of late payments are at higher risk. The Gini tree's better performance and interpretability make it the recommended choice.


# Step 3: Regression Decision Tree
# Exclude TARGET_BAD_FLAG from the predictors for the regression tasks
df_regression <- df
df_regression$TARGET_BAD_FLAG <- NULL

# Decision tree using ANOVA
treeAnova <- rpart(TARGET_LOSS_AMT ~ ., data = df_regression, method = "anova", control = control_parameters)
rpart.plot(treeAnova)
print(treeAnova$variable.importance)

# Decision tree using Poisson
treePoisson <- rpart(TARGET_LOSS_AMT ~ ., data = df_regression, method = "poisson", control = control_parameters)
rpart.plot(treePoisson)
print(treePoisson$variable.importance)

# Calculate RMSE for the ANOVA model
predictionsAnova <- predict(treeAnova, df_regression)
RMSE_Anova <- sqrt(mean((df_regression$TARGET_LOSS_AMT - predictionsAnova)^2))
print(RMSE_Anova)

# Calculate RMSE for the Poisson model
predictionsPoisson <- predict(treePoisson, df_regression)
RMSE_Poisson <- sqrt(mean((df_regression$TARGET_LOSS_AMT - predictionsPoisson)^2))
print(RMSE_Poisson)

# Summary and recommendation
#The regression decision trees, using ANOVA and Poisson methods, aimed to predict the loss amount in the event of a loan default. The ANOVA tree, which achieved a lower RMSE, is favored for its more accurate loss predictions. The loan amount was identified as a key determinant of loss, with larger loans likely incurring greater losses. This is logically consistent with lending practices. Thus, the ANOVA tree is recommended, as it sensibly captures the relationship between loan attributes and potential losses.

# Step 4: Probability / Severity Model Decision Tree
# Re-plotting the classification tree using Gini for TARGET_BAD_FLAG (probability of default)
rpart.plot(treeGini)
print(treeGini$variable.importance)

# Building and plotting the decision tree for TARGET_LOSS_AMT for defaulted records only
df_loss <- df[df$TARGET_BAD_FLAG == 1, ]
treeLoss <- rpart(TARGET_LOSS_AMT ~ ., data = df_loss, method = "anova", control = control_parameters)
rpart.plot(treeLoss)
print(treeLoss$variable.importance)

# Predicting the loss given default
loss_given_default <- predict(treeLoss, df_loss)

# Predicting the probability of default using the classification tree
prob_default <- predict(treeGini, df, type = "prob")[,2] 

# Initialize expected loss with zeros
expected_loss <- rep(0, nrow(df))

# Only calculate expected loss where TARGET_BAD_FLAG is 1
expected_loss[df$TARGET_BAD_FLAG == 1] <- prob_default[df$TARGET_BAD_FLAG == 1] * loss_given_default

# Calculate the RMSE value for the Probability / Severity model
# The RMSE will only be calculated for records with TARGET_BAD_FLAG == 1
RMSE_ProbSeverity <- sqrt(mean((df_loss$TARGET_LOSS_AMT - expected_loss[df$TARGET_BAD_FLAG == 1])^2))
print(RMSE_ProbSeverity)

# Summary and recommendation
# The combined Probability/Severity model, integrating the probability of default with the predicted loss amount, yielded a higher RMSE compared to the standalone regression model from Step 3. Despite this, it provides a more holistic risk assessment by considering both the likelihood and impact of loan default. While the standalone ANOVA model from Step 3 is more accurate for predicting losses among defaults, the comprehensive nature of the Probability/Severity model may offer greater practical value for risk management purposes. Therefore, for strategic decision-making, the Probability/Severity model is recommended, despite its lower precision, because it encapsulates the full spectrum of credit risk.