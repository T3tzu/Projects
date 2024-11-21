# Step 1

library(ggplot2)
library(flexclust)

df <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv")  

SEED = 123
set.seed(SEED)

TARGET = "TARGET_BAD_FLAG"

str(df)

# Step 2: PCA Analysis
df_pca = df
df_pca$TARGET_BAD_FLAG = NULL
df_pca$TARGET_LOSS_AMT = NULL
pca = prcomp(df_pca[,c(1,2,4,6,8,10,12,14,16,18)] ,center=TRUE, scale=TRUE)
summary(pca)
plot(pca, type = "l")

df_new <- data.frame(predict(pca, df_pca))
df_new$TARGET_BAD_FLAG <- df$TARGET_BAD_FLAG

# Print the weights of the Principal Components
print(pca$rotation)

# Plot the first two principal components
ggplot(df_new, aes(x = PC1, y = PC2)) +
  geom_point() +
  theme_minimal() +
  ggtitle("Scatter Plot of the First Two Principal Components") +
  xlab("Principal Component 1") +
  ylab("Principal Component 2")

# Step 3: Cluster Analysis - Find the Number of Clusters
df_new = data.frame(predict(pca, df_pca))
df_kmeans = df_new[1:2]
print(head(df_kmeans))
plot(df_kmeans$PC1, df_kmeans$PC2)

MAX_N = 10
WSS = numeric(MAX_N)
for (N in 1:MAX_N) {
  km = kmeans(df_kmeans, centers=N, nstart=20)
  WSS[N] = km$tot.withinss
}
df_wss = as.data.frame(WSS)
df_wss$clusters = 1:MAX_N

scree_plot = ggplot(df_wss, aes(x=clusters, y=WSS, group=1)) +
  geom_point(size=4) +
  geom_line() +
  scale_x_continuous(breaks=c(2,4,6,8,10)) +
  xlab("Number of Clusters")
scree_plot

# 3 clusters would likely be the most effective choice, balancing between too many clusters (which might overfit and capture noise rather than true structure) and too few (which might underfit and miss important patterns). 

# Step 4: Cluster Analysis
BEST_N = 5
km = kmeans(df_kmeans, centers=BEST_N, nstart=20)
print(km$size)
print(km$centers)

kf = as.kcca(object=km, data=df_kmeans, save.data=TRUE)
kfi = kcca2df(kf)
agg = aggregate(kfi$value, list(kfi$variable, kfi$group), FUN=mean)
barplot(kf)

clus = predict(kf, df_kmeans)
plot(df_kmeans$PC1, df_kmeans$PC2)
plot(df_kmeans$PC1, df_kmeans$PC2, col=clus)
legend(x="topright", legend=c(1:BEST_N), fill=c(1:BEST_N))

df$CLUSTER = clus
agg = aggregate(df$TARGET_BAD_FLAG, list(df$CLUSTER), FUN=mean)

# Step 5: Describe the Clusters Using Decision Trees

library(rpart)
library(rpart.plot)
df_tree = df_pca
df_tree$CLUSTER = as.factor(clus)
dt = rpart(CLUSTER ~ ., data=df_tree)
dt = rpart(CLUSTER ~ ., data=df_tree, maxdepth=3)
rpart.plot(dt)

# The decision tree effectively segments the dataset into five distinct clusters based on financial indicators such as property value, outstanding mortgage, derogatory marks, and recent credit inquiries. Cluster 1 represents individuals with past credit issues and low property values, suggesting financial recovery or stabilization. Cluster 2 indicates more credit-stable individuals with similar property values but without derogatory records. Cluster 3 includes those potentially underwater on their mortgages, implying financial distress. Cluster 4 consists of individuals with moderate mortgages but high credit-seeking behavior, possibly indicating financial strain or opportunity seizing. Finally, Cluster 5 represents a wealthier segment with higher property values, suggesting better financial health. These clusters seem logically organized, reflecting varying financial conditions and behaviors that can aid in targeted financial decision-making and risk assessment.

# Step 6: Comment

# In a corporate setting, the clusters derived from your analysis can be utilized to enhance targeted marketing efforts, optimize risk management strategies, and improve customer segmentation. By understanding the distinct characteristics of each cluster, companies can tailor their services and products to meet the specific needs and preferences of different customer groups, thereby increasing efficiency and effectiveness in their operations. This approach not only aids in precise marketing and risk mitigation but also supports strategic decision-making and the development of predictive models that anticipate customer behaviors.