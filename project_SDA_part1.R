# Get current working directory
getwd()
# Set working directory
setwd('/home/domenico/Desktop/MAGISTRALE/Statistical Data Analysis/Progetto')
# Read data
data <- read.csv("RegressionDataset_DA_group3.csv")
# Display data
print(data)
#images path
images_path <- paste(getwd(), "Images_part1", sep = "/")
if (!dir.exists(images_path)) {
  dir.create(images_path)
}

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
corrplot(corData, tl.cex = 0.22, cl.cex = 0.22, main = "Correlation Matrix", mar=c(3,3,3,3))
png(filename = paste(images_path, "/correlation_matrix.png", sep = ""))
par(mar = c(5, 4, 6, 2) + 1)  # Increase top margin for title
corrplot(corData, tl.cex = 0.22, cl.cex = 0.22, main = "Correlation Matrix", mar=c(3,3,3,3))
dev.off()



# MULTIPLE LINEAR REGRESSION
#fit multiple linear regression model
multiple_lr_fit <- lm(y_train ~ ., data = data_frame_train)
# Summary of the fit
summary(multiple_lr_fit)

# Save multiple lr fit plot
png(filename = paste(images_path, "/multiple_linear_regression_plot.png", sep = ""))
plot(multiple_lr_fit)
dev.off()

# Extract coefficients
coefficients_matrix <- multiple_lr_fit$coefficients
#remove intercept
coefficients_matrix <- coefficients_matrix[-1]
# Create a bar plot and save it as a PNG
png(filename = paste(images_path, "/multiple_linear_regression_coefficients_barplot.png", sep = ""))
barplot(coefficients_matrix, main = "Coefficient Values", xlab = "Predictor Variables", ylab = "Coefficient Value")
dev.off()
# p-values are in the 4th column
# Extract p-values for all coefficients
p_values <- summary(multiple_lr_fit)$coefficients[-1, 4]
# Create a bar plot and save it as a PNG
png(filename = paste(images_path, "/multiple_linear_regression_p_values_barplot.png", sep = ""))
barplot(p_values, main = "P Values", xlab = "Predictor Variables", ylab = "P Value")
dev.off()
# Filter significant coefficients (e.g., p < 0.05)
significant_coeff_names <- which(p_values < 0.05)
significant_coeff_values = coefficients_matrix[significant_coeff_names]
# Display the names of significant coefficients
cat("significant predictors for multiple linear regression:",significant_coeff_names)

ascii_codes <- round(significant_coeff_values / 100)
# Convert to ASCII characters
characters <- intToUtf8(ascii_codes)
cat("clue using multiple linear regression:",characters)

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




#Analysis functions

coeff_analysis <- function(model,file_name){
  for (metric in c("bic", "Cp", "adjr2")){
    png(filename = paste(images_path, paste("/",file_name,"_coeff_plot_", metric, ".png", sep=""), sep = ""))
    plot(model, scale = metric)
    dev.off()
    
  }
  
}

oser_analysis <- function(reg_summary, file_name) {
  oser_indexes <- list()
  cat("Analysing ",file_name," .. \n")
  for (metric in c("bic", "cp", "adjr2")) {  # Removed "rss"
    png(filename = paste(images_path, paste("/", file_name, "_osre_plot_", metric, ".png", sep=""), sep = ""))
    # Existing code for generating plots
    reg_summary_metric <- reg_summary[[metric]]  # Use a different variable to avoid overwriting
    
    if (metric == "adjr2") {
      idx <- which.max(reg_summary_metric)
    } else {
      idx <- which.min(reg_summary_metric)
    }
    se <- sd(reg_summary_metric) / sqrt(length(x_train))
    indexes <- which(reg_summary_metric >= (reg_summary_metric[idx] - se) & reg_summary_metric <= (reg_summary_metric[idx] + se))
    min_index <- min(indexes)
    oser_indexes[[metric]] <- min_index
    plot(reg_summary_metric, xlab = "Number of Variables", ylab = metric, type = "l")
    points(idx, reg_summary_metric[idx], col = "blue", cex = 2, pch = 20)
    points(min_index, reg_summary_metric[min_index], col = "red", cex = 2, pch = 20)
    # Add horizontal lines
    abline(h = reg_summary_metric[idx] - se, col = "green", lty = 2)
    abline(h = reg_summary_metric[idx] + se, col = "green", lty = 2)
    legend("right", c("Optimal", "One SE Rule", "One Se limits"), col = c("blue", "red", "green"), pch = c(20, 20, NA), lty = c(NA, NA, 2), cex = 0.6, inset = c(0.05, 0.05))
    
    # Close the PNG device
    dev.off()
  }
  
  oser_indexes_vector <- unlist(oser_indexes)
  mean_oser_index = round(mean(oser_indexes_vector))
  cat("Number of variables selected with OSE rule: ", mean_oser_index, "\n")
  best_model_coef = coef(regfit.full, id = mean_oser_index)
  best_model_coef <- best_model_coef[-1]
  
  png(filename = paste(images_path, paste("/", file_name, "_best_model_coef_after_oser_barplot.png", sep=""), sep = ""))
  barplot(best_model_coef, main = "Best Model Coefficients", xlab = "Predictor Variables", ylab = "Coefficient Value")
  dev.off()
  
  best_predictors = names(best_model_coef)
  cat(paste("Best predictors name using ", file_name, ": "), best_predictors, "\n")
  
  ascii_codes <- round(best_model_coef / 100)
  characters <- intToUtf8(ascii_codes)
  cat(paste("Clue using ", file_name, ": "), characters, "\n")
}




#BEST SUBSET SELECTION
regfit.full<-regsubsets(x_train,y_train,nvmax = 8,really.big = T)
coeff_analysis(regfit.full,"best_subset_selection")
reg.summary<-summary(regfit.full)
oser_analysis(reg.summary,"best_subset_selection")


#FORWARD SELECTION
fwd.regfit <- regsubsets(x_train,y_train,,method = "forward",really.big = T) # Forward selection on the training data
coeff_analysis(fwd.regfit,"forward_stepwise")
summary_fwd_regfit <- summary(fwd.regfit) # Summary of the results of forward selection
oser_analysis(summary_fwd_regfit,"forward_stepwise")


#BACKWARD SELECTION
bwd.regfit <- regsubsets(x_train,y_train,method = "backward",really.big = T) # Backward selection on the training data
coeff_analysis(bwd.regfit,"backward_stepwise")
summary_bwd_regfit <- summary(bwd.regfit)
oser_analysis(summary_bwd_regfit,"backward_stepwise")




#RIDGE AND LASSO 

# Define lambda sequence
lambda_seq <- seq(0,20, by=0.1)

# Fit Ridge model with specified lambda values
fit.ridge <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq)
# Save Ridge coefficients plot
png(filename = paste(images_path, "/ridge_coefficient_paths.png", sep = ""))
plot(fit.ridge, xvar="lambda", label=TRUE, main="Ridge Coefficient Paths")
dev.off()

# Fit Lasso model with specified lambda values
fit.lasso <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_seq)
# Save Lasso coefficients plot
png(filename = paste(images_path, "/lasso_coefficient_paths.png", sep = ""))
plot(fit.lasso, xvar="lambda", label=TRUE, main="Lasso Coefficient Paths")
dev.off()

# Cross-validation for Ridge with specified lambda values
cv.ridge <- cv.glmnet(x_train, y_train, alpha=0, lambda = lambda_seq, nfolds=10)
# Save Cross-validation plot for Ridge
png(filename = paste(images_path, "/ridge_cv_error.png", sep = ""))
plot(cv.ridge, main="Ridge CV Error")
dev.off()

# Cross-validation for Lasso with specified lambda values
cv.lasso <- cv.glmnet(x_train, y_train, alpha=1, lambda = lambda_seq, nfolds=10)
png(filename = paste(images_path, "/lasso_cv_error.png", sep = ""))
plot(cv.lasso, main="Lasso CV Error")
dev.off()

# Prediction and comparison on test set for Ridge and Lasso
pred.ridge <- predict(fit.ridge, s = cv.ridge$lambda.min, newx = x_test)
pred.lasso <- predict(fit.lasso, s = cv.lasso$lambda.min, newx = x_test)
# Calculate RMSE for Ridge and Lasso
rmse.ridge <- sqrt(mean((pred.ridge - y_test)^2))
rmse.lasso <- sqrt(mean((pred.lasso - y_test)^2))

png(filename = paste(images_path, "/rmse_comparison.png", sep = ""))
barplot(c(rmse.ridge, rmse.lasso), 
        names.arg=c("Ridge", "Lasso"), 
        main="RMSE Comparison on Test Set")
dev.off()

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