#######
# LASSO
#######
# feedback on the data or data2 is the best approch would be nice

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

# Adjust split index based on the forecast horizon. As the forecast horizon
# changes the observation equal to the 2010-1-1 changes. Set split index
# such that it equals the 2010-1-1 observation.

split_point <- 0.8
split_index <- 516 #floor(nrow(low_dat) * split_point) #splitindex
forecast_horison <- 12


low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/complete_low_dat.csv')
# Step 1: Identify columns containing infinite values
columns_with_inf <- sapply(low_dat, function(x) any(is.infinite(x)))

# Print names of columns with infinite values
print(names(columns_with_inf[columns_with_inf]))


# Lag dependent variable
low_dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horison, type = "lag")]

# Since differencing introduces NA values in the first row, remove NAs
low_dat <- na.omit(low_dat)

# Now, proceed with your data splitting and model fitting as before
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]

################
# Training model
################

y_train <- train_dat$CPIAUCSL
X_train <- as.matrix(train_dat[, -which(names(train_dat) == 'CPIAUCSL'), with = FALSE])


# cv_model <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE)

lambda_sequence <- cv_model$lambda
coefficient_matrix <- as.matrix(coef(cv_model, s = lambda_sequence)[-1, , drop = FALSE])

# Prepare data for ggplot
coefs_long <- reshape2::melt(coefficient_matrix)
coefs_long$Lambda <- rep(lambda_sequence, each = nrow(coefficient_matrix))

# Plot without distinguishing lines by variable
ggplot(coefs_long, aes(x = Lambda, y = value)) +
  geom_line(alpha = 0.4, color = "blue") +  # Use a single color and adjust transparency
  scale_x_log10() +
  labs(x = "Lambda (Log Scale)", y = "Coefficient Size", title = "LASSO Path Plot") +
  theme_minimal()

# Find the lambda that works well. 
# You need to have it in a way that doesnt overfit

# optimal lamba
best_lambda <- cv_model$lambda.min #lambda optimal based on function
best_lambda <- 0.3 # I do not totally agree with the model
best_lambda_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)
print(best_lambda)
print(log(best_lambda))

plot(cv_model)
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

rmse_name <- paste('RMSE_LASSO', forecast_horison, '.csv', sep = '')
file_path <- paste('C:/Users/tormo/OneDrive/Skole/Masteroppgave/DM testing/LASSO long', rmse_name, sep='/')

write.csv(rmse_df, file = file_path, row.names = FALSE)

##########
# Plotting 
##########

plot(cv_model)

predicted_values <- as.vector(prediction)

plot_data <- data.frame(
  Date = date[(split_index +1):nrow(low_dat)],
  Actual = y_test,
  Predicted = predicted_values
  
)

plot_data$Date <- as.Date(plot_data$Date)

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, colour = 'Actual')) +
  geom_line(aes(y = Predicted, colour = 'Predicted'), linetype = 'dashed') +
  labs(title = 'Actual vs Predicted CPI mnd 12', y = 'CPI', x ='Date') + 
  theme_minimal() + # remember to set (ex 6mnd) the correct name 
  scale_colour_manual('', values = c(Actual = 'blue', Predicted = 'red')) +
  coord_cartesian(ylim = c(-10, 15))


###########
#Save plots
###########
# only have turned on when you want to make plots
# Save the last plot to a specific directory

# Remember to change name and un #

# ggsave("C:/Users/tormo/OneDrive/Skole/Masteroppgave/Plots/LASSOmnd12.png", plot = last_plot(), width = 10, height = 8, units = "in")

print(mean(rmse))

