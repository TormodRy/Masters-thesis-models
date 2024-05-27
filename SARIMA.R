
########################
# SARIMA model long time
########################


#########
# Library
#########

library(forecast)
library(data.table)
library(ggplot2)

######
# Data
######

low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/long_low_dat23.csv')
summary(low_dat)
split_index <- 600
forecast_horizon <- 48
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]
p <- 2  
d <- 1
q <- 1
P <- 1 
D <- 1 
Q <- 1
m <- 12
seasonal_frequency <- 12

y_train <- ts(train_dat$CPIAUCSL, frequency = seasonal_frequency)
y_test <- ts(test_dat$CPIAUCSL, frequency = seasonal_frequency)
################
# Training model
################
# autoarima
# Convert the training data to a time series object
train_ts <- ts(train_dat$CPIAUCSL, frequency=seasonal_frequency)

# Fit SARIMA model on training data
fit <- auto.arima(train_ts)
print(fit)

# ACF

# ACF function
ACF_test <- acf(train_dat$CPIAUCSL, plot = TRUE)
print(ACF_test)
plot(ACF_test) # suggests some non stationary process, but not significant

# PACF function
PACF_test <- pacf(train_dat$CPIAUCSL, plot = TRUE)
print(PACF_test)


# BIC testing 
arma_lag_testBIC <- function(limit2) {
  for (i in 1:limit2) {
    arma_model_test <- arima(train_dat$CPIAUCSL, order = c(i,d,q))
    cat(BIC(arma_model_test), 'BIC', i , '\n')
    
  }
}

# BIC, 
arma_lag_testBIC(13)

#####################
# Model specification
#####################

# Fit an AR(p) model
# Correct specification of the SARIMA model with seasonal components
model_sarima <- Arima(y_train, order=c(p, d, q), seasonal=list(order=c(P, D, Q), period=m))

# model_ar <- Arima(y_train, order=c(p,d,q), seasonal =c(P,D,Q,m))

# Initialize a vector to store the forecast for each time step in the test set
forecasts <- numeric(length = length(y_test) - forecast_horizon + 1)

# Loop through each point in the test dataset
for(i in 1:(length(y_test) - forecast_horizon + 1)) {
  # Update the model with the actual data point
  model_sarima <- Arima(c(y_train, y_test[1:i + forecast_horizon - 1]), order=c(p, d, q), seasonal=list(order=c(P, D, Q), period=m))
  # model_ar_updated <- Arima(c(y_train, y_test[1:i + forecast_horizon - 1]), order=c(p,0,0))
  
  # Forecast the next time point based on the horizon
  next_forecast <- forecast(model_sarima, h=forecast_horizon)$mean
  
  # Store the forecast corresponding to the horizon
  forecasts[i] <- next_forecast[forecast_horizon]
}

# Prepare the data for plotting
plot_data <- data.frame(
  Date = time(y_test)[forecast_horizon:(length(y_test))],
  Actual = as.numeric(y_test)[forecast_horizon:(length(y_test))],
  Forecasted = forecasts
)

# Plot the actual vs forecasted values
ggplot(plot_data, aes(x = Date)) + 
  geom_line(aes(y = Actual, colour = "Actual")) + 
  geom_line(aes(y = Forecasted, colour = "Forecasted"), linetype="dashed") + 
  scale_colour_manual("", values = c("Actual" = "blue", "Forecasted" = "red")) + 
  labs(title = paste("AR(", p, ") Model: Actual vs Forecasted with Horizon ", forecast_horizon, sep=""), x = "Date", y = "CPIAUCSL") + 
  theme_minimal() +
  guides(colour = guide_legend(title = "Data Type"))


# RMSE
# Assuming the rest of the script remains unchanged, and focusing on adding the RMSE calculation

# After the forecasting loop, calculate the RMSE between actual and predicted values
actual_values <- as.numeric(y_test)[forecast_horizon:(length(y_test))]
predicted_values <- forecasts

# Calculate RMSE
rmse <- sqrt(mean((actual_values - predicted_values)^2))

# Print the RMSE to the console
cat("RMSE for forecast horizon", forecast_horizon, "is:", rmse, "\n")




