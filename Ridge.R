#######
# Ridge
#######

# Many things here are similar to what you will see in LASSO code. 

#########
# Library
#########

library(data.table)
library(ggplot2)
library(glmnet)
library(reshape2)

######
# Data
######

split_point <- 0.8
split_index <- 540 #floor(nrow(low_dat) * split_point) #splitindex
forecast_horison <- 24
low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/complete_low_dat.csv')
# Step 1: Identify columns containing infinite values
columns_with_inf <- sapply(low_dat, function(x) any(is.infinite(x)))

# Print names of columns with infinite values
print(names(columns_with_inf[columns_with_inf]))


########
# Data 2
########


# Lag dependent variable
low_dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horison, type = "lag")]

# Since differencing introduces NA values in the first row, remove NAs
low_dat <- na.omit(low_dat)
low_dat$date <- NULL

# Now, proceed with your data splitting and model fitting as before
# split_index <- floor(nrow(low_dat) * 0.7)
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]


################
# Training model
################

y_train <- train_dat$CPIAUCSL
X_train <- as.matrix(train_dat[, -which(names(train_dat) == 'CPIAUCSL'), with = FALSE])


cv_model <- cv.glmnet(X_train, y_train, alpha = 0, standardize = TRUE)

lambda_sequence <- cv_model$lambda
coefficient_matrix <- as.matrix(coef(cv_model, s = lambda_sequence)[-1, , drop = FALSE])

# Prepare data for ggplot
coefs_long <- reshape2::melt(coefficient_matrix)
coefs_long$Lambda <- rep(lambda_sequence, each = nrow(coefficient_matrix))

# Plot without distinguishing lines by variable
ggplot(coefs_long, aes(x = Lambda, y = value)) +
  geom_line(alpha = 0.4, color = "blue") +  # Use a single color and adjust transparency
  scale_x_log10() +
  labs(x = "Lambda (Log Scale)", y = "Coefficient Size", title = "Ridge Path Plot") +
  theme_minimal()

# choose lambda based on validation sample.

# optimal lamba
best_lambda <- cv_model$lambda.min #lambda optimal based on function
# best_lambda <- 0.3 # I do not totally agree with the model
best_lambda_model <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda)
print(best_lambda)

print(log(best_lambda))

plot(cv_model)
summary(cv_model)

###############
# testing model 
###############

X_test <- as.matrix(test_dat[, - which(names(test_dat) == 'CPIAUCSL'), with = FALSE])
y_test <- test_dat$CPIAUCSL

prediction <- predict(best_lambda_model, s = best_lambda, newx = X_test)


######
# RMSE
######


rmse <- sqrt((prediction - y_test)^2)
print(mean(rmse))
rmse_df <- data.frame(rmse)

rmse_name <- paste('RMSE_RIDGE', forecast_horison, '.csv', sep = '')
file_path <- paste('C:/Users/tormo/OneDrive/Skole/Masteroppgave/DM testing/Ridge long', rmse_name, sep='/')

# write.csv(rmse_df, file = file_path, row.names = FALSE)

plot(cv_model)

predicted_values <- as.vector(prediction)

print(mean(rmse))

