# Step 1: Read in the Data
HMEQ_Loss <- read.csv("C:/Users/zachz/Desktop/HMEQ/HMEQ_Loss.csv")
str(HMEQ_Loss)
summary(HMEQ_Loss)
head(HMEQ_Loss, 6)

# Step 2: Box-Whisker Plots
par(mfrow=c(2, 3))
boxplot(LOAN ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "LOAN")
boxplot(MORTDUE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "MORTDUE")
boxplot(VALUE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "VALUE")
boxplot(YOJ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "YOJ")
boxplot(DEROG ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DEROG")
boxplot(DELINQ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DELINQ")
par(mfrow=c(2, 3))
boxplot(CLAGE ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "CLAGE")
boxplot(NINQ ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "NINQ")
boxplot(CLNO ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "CLNO")
boxplot(DEBTINC ~ TARGET_BAD_FLAG, data = HMEQ_Loss, main = "Hanzhe Zhang", col = c("blue", "green"),
        xlab = "TARGET_BAD_FLAG", ylab = "DEBTINC")
# Reset the plotting layout to default
par(mfrow=c(1, 1))

# Step 3: Histograms
# Histogram with density line
hist(HMEQ_Loss$LOAN, breaks=50, main="Histogram of LOAN", xlab="LOAN Amount", freq=FALSE)
dens <- density(na.omit(HMEQ_Loss$LOAN))
lines(dens, col="red") # Superimpose density line

# Step 4: Impute "Fix" all the numeric variables that have missing values
# Set missing Target variables to zero
HMEQ_Loss$TARGET_LOSS_AMT[is.na(HMEQ_Loss$TARGET_LOSS_AMT)] <- 0
# Impute numeric variables with missing values using median and create flags
# LOAN has no missing values as it's the loan amount requested
# MORTDUE
HMEQ_Loss$IMP_MORTDUE <- ifelse(is.na(HMEQ_Loss$MORTDUE), median(HMEQ_Loss$MORTDUE, na.rm = TRUE), HMEQ_Loss$MORTDUE)
HMEQ_Loss$M_MORTDUE <- ifelse(is.na(HMEQ_Loss$MORTDUE), 1, 0)
HMEQ_Loss$MORTDUE <- NULL # Delete the original MORTDUE variable
# complex imputation for VALUE 
# First, compute the median VALUE for each JOB category
median_values_by_job <- aggregate(VALUE ~ JOB, data = HMEQ_Loss, FUN = median, na.rm = TRUE)
# Then, impute missing VALUE based on the median of the corresponding JOB category
# Create a new IMP_VALUE variable with imputed values
HMEQ_Loss$IMP_VALUE <- HMEQ_Loss$VALUE
# Loop over the JOB categories to impute missing VALUEs
for (job in median_values_by_job$JOB) {
  # Get the median value for the current JOB
  median_value <- median_values_by_job[median_values_by_job$JOB == job, ]$VALUE
  # Apply the imputation for records with the current JOB and a missing VALUE
  missing_indices <- is.na(HMEQ_Loss$VALUE) & HMEQ_Loss$JOB == job
  HMEQ_Loss$IMP_VALUE[missing_indices] <- median_value
}
# Create the M_VALUE flag variable
HMEQ_Loss$M_VALUE <- as.integer(is.na(HMEQ_Loss$VALUE))
# Now, impute remaining missing IMP_VALUE with the overall median of VALUE, if any
overall_median <- median(HMEQ_Loss$VALUE, na.rm = TRUE)
still_missing <- is.na(HMEQ_Loss$IMP_VALUE)
HMEQ_Loss$IMP_VALUE[still_missing] <- overall_median
HMEQ_Loss$VALUE <- NULL # Delete the original VALUE variable
# YOJ
HMEQ_Loss$IMP_YOJ <- ifelse(is.na(HMEQ_Loss$YOJ), median(HMEQ_Loss$YOJ, na.rm = TRUE), HMEQ_Loss$YOJ)
HMEQ_Loss$M_YOJ <- as.integer(is.na(HMEQ_Loss$YOJ))
HMEQ_Loss$YOJ <- NULL  # Delete the original YOJ variable
# DEROG
HMEQ_Loss$IMP_DEROG <- ifelse(is.na(HMEQ_Loss$DEROG), median(HMEQ_Loss$DEROG, na.rm = TRUE), HMEQ_Loss$DEROG)
HMEQ_Loss$M_DEROG <- as.integer(is.na(HMEQ_Loss$DEROG))
HMEQ_Loss$DEROG <- NULL  # Delete the original DEROG variable
# DELINQ
HMEQ_Loss$IMP_DELINQ <- ifelse(is.na(HMEQ_Loss$DELINQ), median(HMEQ_Loss$DELINQ, na.rm = TRUE), HMEQ_Loss$DELINQ)
HMEQ_Loss$M_DELINQ <- as.integer(is.na(HMEQ_Loss$DELINQ))
HMEQ_Loss$DELINQ <- NULL  # Delete the original DELINQ variable
# CLAGE
HMEQ_Loss$IMP_CLAGE <- ifelse(is.na(HMEQ_Loss$CLAGE), median(HMEQ_Loss$CLAGE, na.rm = TRUE), HMEQ_Loss$CLAGE)
HMEQ_Loss$M_CLAGE <- as.integer(is.na(HMEQ_Loss$CLAGE))
HMEQ_Loss$CLAGE <- NULL  # Delete the original CLAGE variable
# NINQ
HMEQ_Loss$IMP_NINQ <- ifelse(is.na(HMEQ_Loss$NINQ), median(HMEQ_Loss$NINQ, na.rm = TRUE), HMEQ_Loss$NINQ)
HMEQ_Loss$M_NINQ <- as.integer(is.na(HMEQ_Loss$NINQ))
HMEQ_Loss$NINQ <- NULL  # Delete the original NINQ variabl
# CLNO
HMEQ_Loss$IMP_CLNO <- ifelse(is.na(HMEQ_Loss$CLNO), median(HMEQ_Loss$CLNO, na.rm = TRUE), HMEQ_Loss$CLNO)
HMEQ_Loss$M_CLNO <- as.integer(is.na(HMEQ_Loss$CLNO))
HMEQ_Loss$CLNO <- NULL  # Delete the original CLNO variable
# DEBTINC
HMEQ_Loss$IMP_DEBTINC <- ifelse(is.na(HMEQ_Loss$DEBTINC), median(HMEQ_Loss$DEBTINC, na.rm = TRUE), HMEQ_Loss$DEBTINC)
HMEQ_Loss$M_DEBTINC <- as.integer(is.na(HMEQ_Loss$DEBTINC))
HMEQ_Loss$DEBTINC <- NULL  # Delete the original DEBTINC variable
# After all imputations, run a summary to check for missing values
summary(HMEQ_Loss)
# Compute a sum for all the M_ variables
sum_M_variables <- colSums(HMEQ_Loss[, grepl("^M_", names(HMEQ_Loss))])
print(sum_M_variables)

# Step 5: One Hot Encoding
categoryVars <- sapply(HMEQ_Loss, is.character)
for (var in names(HMEQ_Loss)[categoryVars]) {
  levels <- unique(HMEQ_Loss[[var]])
  for (level in levels) {
    HMEQ_Loss[[paste0("FLAG_", var, "_", level)]] <- ifelse(HMEQ_Loss[[var]] == level, 1, 0)
  }
  HMEQ_Loss[[var]] <- NULL # Remove original variable
}
summary(HMEQ_Loss) # Show the dataset after encoding

# Save the processed dataset to a CSV file
output_file_path <- "C:/Users/zachz/Desktop/HMEQ/HMEQ_Scrubbed.csv"
write.csv(HMEQ_Loss, output_file_path, row.names = FALSE)
cat("Processed dataset saved to:", output_file_path, "\n")
