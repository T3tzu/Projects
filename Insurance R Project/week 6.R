# Step 1

library( rpart )
library( rpart.plot )
library( ROCR )
library( MASS )
library( Rtsne )
library( randomForest )
library( ggplot2 )

df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")  

SEED = 123
set.seed( SEED )

# Step 2: PCA Analysis

df_pca = df
df_pca$TARGET_BAD_FLAG = NULL
df_pca$TARGET_LOSS_AMT = NULL
pca2 = prcomp(df_pca[,c(1,2,4,6,8,10,12,14,16,18)] ,center=TRUE, scale=TRUE)
summary(pca2)
plot(pca2, type = "l")
df_new <- data.frame(predict(pca2, df_pca))
df_new$TARGET_BAD_FLAG <- df$TARGET_BAD_FLAG

# Print the weights of the Principal Components
print(pca2$rotation)

# Plot the first two principal components colored by the Target Flag
ggplot(df_new, aes(x = PC1, y = PC2, color = factor(TARGET_BAD_FLAG))) +
  geom_point() +
  scale_color_manual(values = c("blue", "red"), 
                     labels = c("Non Default", "Default")) +
  theme_minimal() +
  ggtitle("Scatter Plot of the First Two Principal Components") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2")

# The first two Principal Components do not clearly separate the default and non-default classes, as there is significant overlap between them.

# Step 3: tSNE Analysis

dfu = df
dfu$TARGET_LOSS_AMT = NULL
dfu = unique(dfu)

# Conduct tSNE analysis with Perplexity = 30
theTSNE = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
dfu$TS1 = theTSNE$Y[,1]
dfu$TS2 = theTSNE$Y[,2]

# Plotting the results with TARGET_BAD_FLAG
library(ggplot2)
ggplot(dfu, aes(x = TS1, y = TS2, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 30")

# The tSNE plot with a perplexity of 30 shows considerable overlap between defaults and non-defaults, indicating that the tSNE values at this perplexity level are not highly predictive of the Target Flag.

# Conduct tSNE with a higher perplexity, e.g., 50
theTSNE_high = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=50, verbose=TRUE, max_iter = 500)
dfu$TS1_high = theTSNE_high$Y[,1]
dfu$TS2_high = theTSNE_high$Y[,2]
ggplot(dfu, aes(x = TS1_high, y = TS2_high, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 50")

# Conduct tSNE with a lower perplexity, e.g., 10
theTSNE_low = Rtsne(dfu[, c(2,3,5,7,9,11,13,15,17,19)], dims = 2, perplexity=10, verbose=TRUE, max_iter = 500)
dfu$TS1_low = theTSNE_low$Y[,1]
dfu$TS2_low = theTSNE_low$Y[,2]
ggplot(dfu, aes(x = TS1_low, y = TS2_low, color = factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.5) +
  labs(color = "Target Flag") +
  ggtitle("tSNE Plot with Perplexity = 10")

# The plot with perplexity 50 shows a tendency towards more discernible clustering compared to the others, which may suggest a marginally better predictive ability for the Target Flag.

# Train Random Forest Models
P = paste(colnames(dfu)[c(2,3,5,7,9,11,13,15,17,19)], collapse = "+")
F1 = as.formula( paste("TS1 ~", P ) )
F2 = as.formula( paste("TS2 ~", P ) )
print( F1 )
print( F2 )
ts1_model_rf = randomForest( data=dfu, F1, ntree=500, importance=TRUE )
ts2_model_rf = randomForest( data=dfu, F2, ntree=500, importance=TRUE )

# Step 4: Tree and Regression Analysis on the Original Data

df_model = df
df_model$TARGET_LOSS_AMT = NULL

head( df_model )

# Decision Tree
tr_set = rpart.control( maxdepth = 10 )
t1G = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='gini') )
t1E = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='information') )

rpart.plot( t1G )
rpart.plot( t1E )

t1G$variable.importance
t1E$variable.importance

# In both decision tree models (t1G and t1E), M_DEBTINC appears to be the most important predictor, followed by IMP_DEBTINC, IMP_DELINQ, and M_VALUE. 

# Logistic Regression
theUpper_LR = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_model )
theLower_LR = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_model )

summary( theUpper_LR )
summary( theLower_LR )

lr_model = stepAIC(theLower_LR, direction="forward", scope=list(lower=theLower_LR, upper=theUpper_LR))
summary( lr_model )

# In the logistic regression model (lr_model), several variables have statistically significant coefficients: M_DEBTINC, IMP_DELINQ, IMP_DEBTINC, M_VALUE, M_DEROG, IMP_DEROG, IMP_NINQ, FLAG.Job.Office, M_YOJ, FLAG.Job.Sales, M_DELINQ, M_CLNO, IMP_CLNO, FLAG.Job.Other, IMP_VALUE, IMP_YOJ, FLAG.Job.Self, FLAG.Job.Mgr, FLAG.Job.ProfExe, M_MORTDUE, and IMP_MORTDUE.

pG = predict( t1G, df_model )
pG2 = prediction( pG[,2], df_model$TARGET_BAD_FLAG )
pG3 = performance( pG2, "tpr", "fpr" )

pE = predict( t1E, df_model )
pE2 = prediction( pE[,2], df_model$TARGET_BAD_FLAG )
pE3 = performance( pE2, "tpr", "fpr" )

plr = predict( lr_model, df_model, type="response" )
plr2 = prediction( plr, df_model$TARGET_BAD_FLAG )
plr3 = performance( plr2, "tpr", "fpr" )

plot( pG3, col="red" )
plot( pE3, col="green", add=TRUE )
plot( plr3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("GINI","ENTROPY","REGRESSION"),col=c("red","green","blue"), bty="y", lty=1 )

aucG = performance( pG2, "auc" )@y.values
aucE = performance( pE2, "auc" )@y.values
aucR = performance( plr2, "auc" )@y.values

print( aucG )
print( aucE )
print( aucR )

# Step 5: Tree and Regression Analysis on the PCA/tSNE Data

df_model = df
df_model$TARGET_LOSS_AMT = NULL

df_model$PC1 = df_new[,"PC1"]
df_model$PC2 = df_new[,"PC2"]
df_model$PC3 = df_new[,"PC3"]
df_model$PC4 = df_new[,"PC4"]

df_model$TS1M_RF = predict( ts1_model_rf, df_model )
df_model$TS2M_RF = predict( ts2_model_rf, df_model )

df_model$LOAN = NULL
df_model$IMP_MORTDUE = NULL
df_model$IMP_VALUE = NULL
df_model$IMP_YOJ = NULL
df_model$IMP_DEROG = NULL
df_model$IMP_DELINQ = NULL
df_model$IMP_CLAGE = NULL
df_model$IMP_NINQ = NULL
df_model$IMP_CLNO = NULL
df_model$IMP_DEBTINC = NULL

head( df_model )

# Decision Tree
tr_set = rpart.control( maxdepth = 10 )
t1G = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='gini') )
t1E = rpart( data=df_model, TARGET_BAD_FLAG ~ ., control=tr_set, method="class", parms=list(split='information') )

rpart.plot( t1G )
rpart.plot( t1E )

t1G$variable.importance
t1E$variable.importance

# The models incorporate key variables like M_DEBTINC, M_VALUE, and M_CLAGE, along with principal components (PC1, PC2, PC3, PC4) and t-SNE derived features (TS1M_RF, TS2M_RF). These advanced statistical techniques—PCA and t-SNE—help capture complex, multidimensional patterns in the data, enhancing the models' ability to predict outcomes effectively. Notably, M_DEBTINC and PC2 emerge as particularly significant predictors in both models. 

# Logistic Rregression
theUpper_LR = glm( TARGET_BAD_FLAG ~ ., family = "binomial", data=df_model )
theLower_LR = glm( TARGET_BAD_FLAG ~ 1, family = "binomial", data=df_model )

summary( theUpper_LR )
summary( theLower_LR )

lr_model = stepAIC(theLower_LR, direction="forward", scope=list(lower=theLower_LR, upper=theUpper_LR))
summary( lr_model )

# The logistic regression model incorporates a diverse set of variables including both traditional loan and demographic data (like M_DEBTINC, M_VALUE, M_DEROG, and job type flags) as well as advanced statistical features such as principal components (PC2) and t-SNE derived variables (TS2M_RF and TS1M_RF). The inclusion of PC2 suggests that the second principal component was particularly relevant in capturing variance in the dataset, possibly encapsulating multiple correlated variables into a single predictor. The t-SNE features (TS2M_RF and TS1M_RF), although included, show relatively lower importance in the model, indicating that while they contribute to the model, their impact is less dominant compared to direct financial measures or PCA features. This combination of variables suggests a robust approach to understanding the factors influencing loan default, leveraging both raw and transformed features to enhance predictive accuracy.

# ROC and AUC
pG = predict( t1G, df_model )
pG2 = prediction( pG[,2], df_model$TARGET_BAD_FLAG )
pG3 = performance( pG2, "tpr", "fpr" )

pE = predict( t1E, df_model )
pE2 = prediction( pE[,2], df_model$TARGET_BAD_FLAG )
pE3 = performance( pE2, "tpr", "fpr" )

plr = predict( lr_model, df_model, type="response" )
plr2 = prediction( plr, df_model$TARGET_BAD_FLAG )
plr3 = performance( plr2, "tpr", "fpr" )

plot( pG3, col="red" )
plot( pE3, col="green", add=TRUE )
plot( plr3, col="blue", add=TRUE )
abline(0,1,lty=2)
legend("bottomright",c("GINI","ENTROPY","REGRESSION"),col=c("red","green","blue"), bty="y", lty=1 )

aucG = performance( pG2, "auc" )@y.values
aucE = performance( pE2, "auc" )@y.values
aucR = performance( plr2, "auc" )@y.values

print( aucG )
print( aucE )
print( aucR )

# The PCA and t-SNE values have improved the models' abilities to differentiate between the classes effectively, especially in logistic regression. This suggests that these dimensionality reduction techniques are vital in handling complex datasets where direct relationships between variables and outcomes are not easily discernible. Thus, when compared to using the original dataset alone, incorporating PCA and t-SNE seems to provide a substantial improvement in model performance, making them valuable tools in predictive analytics, particularly for datasets with complex, high-dimensional structures.
