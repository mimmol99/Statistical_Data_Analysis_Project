# Get current working directory
getwd()
# Set working directory
setwd('/home/domenico/Desktop/MAGISTRALE/Statistical Data Analysis/Progetto')
# Read data
data <- read.csv("RegressionDataset_DA_group3.csv")
# Display data
print(data)

# Load libraries
library(glmnet)
library(caret)
library(leaps)
library(corrplot) 

# Set random seed for reproducibility
set.seed(123)
# Create index for training data
trainIndex <- sample(1:nrow(data), 0.8*nrow(data))
# Create training and test sets
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

#x is a matrix of all train data columns except the first (the response) while y is the first column(the response)
x_train = as.matrix(trainData[, -1])
y_train = trainData[, 1]
x_test = as.matrix(testData[, -1])
y_test = testData[,1]

# Create data frames for training and testing
data_frame_train <- data.frame(y_train, x_train)
colnames(data_frame_train)[1] <- "y_train"
data_frame_test <- data.frame(y_test, x_test)
colnames(data_frame_test)[1] <- "y_test"

# Print the lengths
cat("Length of x_train is ", nrow(x_train), "\n")
cat("Length of y_train is ", length(y_train), "\n")
cat("Length of x_test is ", nrow(x_test), "\n")
cat("Length of y_test is ", length(y_test), "\n")


#CORRELATION MATRIX
corData <- round(cor(x_train), digits = 2) 
corrplot(corData, cex = 0.22, show.legend = TRUE, main = "Correlation Matrix") 




# MULTIPLE LINEAR REGRESSION
#fit multiple linear regression model
multiple_lr_fit <- lm(y_train ~ ., data = data_frame_train)
# Summary of the fit
summary(multiple_lr_fit)

# Extract coefficients
coefficients_matrix <- multiple_lr_fit$coefficients
#remove intercept
coefficients_matrix <- coefficients_matrix[-1]
# p-values are in the 4th column
# Extract p-values for all coefficients
p_values <- summary(multiple_lr_fit)$coefficients[-1, 4]
# Filter significant coefficients (e.g., p < 0.05)
significant_coeff_names <- which(p_values < 0.05)
significant_coeff_values = coefficients_matrix[significant_coeff_names]
# Display the names of significant coefficients
cat("significant predictors for multiple linear regression:",significant_coeff_names)

ascii_codes <- round(significant_coeff_values / 100)
# Convert to ASCII characters
characters <- intToUtf8(ascii_codes)
cat("clue using simple linear regression:",characters)

# Make predictions on test data
predictions_mlr <- predict(multiple_lr_fit, newdata = data_frame_test)
# Calculate residuals (difference between observed and predicted values)
residuals <- y_test - predictions_mlr
# Calculate the Mean Squared Error (MSE)
mse <- mean(residuals^2)
# Create a data frame for plotting
mse_df <- data.frame(MSE = mse)
# Print the MSE value
cat("Test Mean Squared Error (MSE) of multiple linear regression is:", mse, "\n")
#test MSE is very high




#BEST SUBSET SELECTION
regfit.full<-regsubsets(x_train,y_train,nvmax = 8,really.big = T)
reg.summary<-summary(regfit.full)

oser_indexes <- list()
# Loop through the summaries
for (reg_summary_name in c("bic", "cp", "adjr2")) {
  # Select the appropriate summary based on the name
  reg_summary <- reg.summary[[reg_summary_name]]
  # Determine the index of the minimum or maximum value based on the summary name
  if (reg_summary_name == "adjr2") {
    idx <- which.max(reg_summary)
  } else {
    idx <- which.min(reg_summary)
  }
  # Calculate the standard error
  se <- sd(reg_summary) / sqrt(length(x_train))
  # Find the indexes within one standard error of the minimum or maximum
  indexes <- which(reg_summary >= (reg_summary[idx] - se) & reg_summary <= (reg_summary[idx] + se))
  # Find the minimum index within the range
  min_index <- min(indexes)
  # Insert the min_index into oser_indexes
  oser_indexes[[reg_summary_name]] <- min_index
  # Initialize plot
  plot(reg_summary, xlab = "Number of Variables", ylab = reg_summary_name, type = "l")
  # Add points for max or min value
  points(idx, reg_summary[idx], col = "blue", cex = 2, pch = 20)
  # Add points for min_index
  points(min_index, reg_summary[min_index], col = "red", cex = 2, pch = 20)
  # Add legend
  legend("right", legend = c("Optimal", "One SE Rule"), col = c("blue", "red"), pch = 20, cex = 0.6, inset = c(0.05, 0.05))
}

# Convert the list to a numeric vector
oser_indexes_vector <- unlist(oser_indexes)
#find the mean of oser
mean_oser_index = round(mean(oser_indexes_vector))
# Extract coefficients for the model with mean_oser_index predictors
best_model_coef = coef(regfit.full, id = mean_oser_index)
# Extract the names of the predictors
best_predictors = names(best_model_coef)[-1]  # Exclude intercept
# Display the names of the best predictors
cat("Best predictors name using bestsubset selection: ",best_predictors)

ascii_codes <- round(best_model_coef / 100)
# Convert to ASCII characters
characters <- intToUtf8(ascii_codes)
cat("clue using bestsubset selection:",characters)

#FORWARD SELECTION


#RIDGE AND LASSO 

# Define lambda sequence
lambda_seq <- seq(0,20, by=0.1)

# Fit Ridge model with specified lambda values
fit.ridge <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq)
# Plot Ridge coefficients
plot(fit.ridge, xvar="lambda", label=TRUE, main="Ridge Coefficient Paths")

# Fit Lasso model with specified lambda values
fit.lasso <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_seq)
# Plot Lasso coefficients
plot(fit.lasso, xvar="lambda", label=TRUE, main="Lasso Coefficient Paths")

# Cross-validation for Ridge with specified lambda values
cv.ridge <- cv.glmnet(x_train, y_train, alpha=0, lambda = lambda_seq, nfolds=10)
plot(cv.ridge, main="Ridge CV Error")

# Cross-validation for Lasso with specified lambda values
cv.lasso <- cv.glmnet(x_train, y_train, alpha=1, lambda = lambda_seq, nfolds=10)
plot(cv.lasso, main="Lasso CV Error")

# Prediction and comparison on test set for Ridge and Lasso
pred.ridge <- predict(fit.ridge, s = cv.ridge$lambda.min, newx = x_test)
pred.lasso <- predict(fit.lasso, s = cv.lasso$lambda.min, newx = x_test)
# Calculate RMSE for Ridge and Lasso
rmse.ridge <- sqrt(mean((pred.ridge - y_test)^2))
rmse.lasso <- sqrt(mean((pred.lasso - y_test)^2))

# Plot RMSE comparison
barplot(c(rmse.ridge, rmse.lasso), 
        names.arg=c("Ridge", "Lasso"), 
        main="RMSE Comparison on Test Set")

# Identify the model with the lowest RMSE
lowest_rmse_model <- which.min(c(rmse.ridge, rmse.lasso))
lowest_rmse_model

# Extract best coefficients based on the model with the lowest RMSE
if (lowest_rmse_model == 1) {  # Ridge
  best_coef_matrix <- as.matrix(coef(cv.ridge, s = cv.ridge$lambda.min))
} else if (lowest_rmse_model == 2) {  # Lasso
  best_coef_matrix <- as.matrix(coef(cv.lasso, s = cv.lasso$lambda.min))
}

# Remove the intercept from the matrix
best_coef_matrix <- best_coef_matrix[-1, , drop = FALSE]
# Convert sparse matrix to dense vector
dense_coef <- best_coef_matrix
# Filter out zeros and NAs
filtered_coef <- dense_coef[!is.na(dense_coef) & dense_coef != 0]
# Divide by 100 and round to nearest integer
ascii_codes <- round(filtered_coef / 100)
# Convert to ASCII characters
characters <- intToUtf8(ascii_codes)
cat("clue using Ridge/Lasso:",characters)