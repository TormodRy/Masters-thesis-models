
########################
# AR(p) model long time
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

# Adjust split index based on the forecast horizon. As the forecast horizon
# changes the observation equal to the 2010-1-1 changes. Set split index such
# that it equals the 2010-1-1 observation.

low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/long_low_dat23.csv')
split_index <- 600
forecast_horizon <- 6
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]
p <- 3  # Example order, this should be determined based on your data

y_train <- ts(train_dat$CPIAUCSL)
y_test <- ts(test_dat$CPIAUCSL)
################
# Training model
################
# Train the AR(p) model by looking at the Bayesian information criterion

# BIC testing 
arma_lag_testBIC <- function(limit2) {
  for (i in 1:limit2) {
    arma_model_test <- arima(train_dat$CPIAUCSL, order = c(i,0,0))
    cat(BIC(arma_model_test), 'BIC', i , '\n')
    
  }
}

 # BIC, 
arma_lag_testBIC(5)


# Fit an AR(p) model
model_ar <- Arima(y_train, order=c(p,0,0))  # ARIMA(p,d,q) with p=1, d=0 (no differencing), q=0 (no MA part)

# Initialize a vector to store the forecast for each time step in the test set
forecasts <- numeric(length = length(y_test) - forecast_horizon+1)

# Loop through each point in the test dataset
for(i in 1:(length(y_test) - forecast_horizon+1)) {
  # Update the model with the actual data point
  model_ar_updated <- Arima(c(y_train, y_test[1:i + forecast_horizon]), order=c(p,0,0))
  
  # Forecast the next time point based on the horizon
  next_forecast <- forecast(model_ar_updated, h=forecast_horizon)$mean
  
  # Store the forecast corresponding to the horizon
  forecasts[i] <- next_forecast[forecast_horizon]
}
# length(forecasts)

# Prepare the data for plotting
plot_data <- data.frame(
  Date = time(y_test)[forecast_horizon:(length(y_test))],
  Actual = as.numeric(y_test)[(forecast_horizon):(length(y_test))],
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


# After the forecasting loop, calculate the RMSE between actual and predicted values
actual_values <- as.numeric(y_test)[(forecast_horizon):(length(y_test))]
predicted_values <- forecasts

# Calculate RMSE
rmse <- sqrt(mean((actual_values - predicted_values)^2))

