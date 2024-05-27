
######
# LSTM
######

# This is a short version of the LSTM code

###########
# Libraries
###########

library(keras)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(glmnet)
library(reshape2)

######
# Data
######


# Read and prepare the dataset
# Rearrange so 'date' is the first column if necessary
dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/complete_low_dat.csv')

# Lag dependent variable
dat[, CPIAUCSL := shift(CPIAUCSL, 12, type = "lag")]
dat <- na.omit(dat)

# Convert 'date' column to timestamps
dat$date <- as.numeric(as.POSIXct(dat$date, format = "%Y-%m-%d"))

################
# Model defining 
################

# Prepare the target variable
CPI_plt <- dat$CPIAUCSL
y <- dat$CPIAUCSL
dat$CPIAUCSL <- NULL

# Create sequences
create_sequences <- function(data, y, n_steps) {
  X <- list()
  Y <- list()
  for (i in 1:(nrow(data) - n_steps)) {
    X[[i]] <- as.matrix(data[i:(i + n_steps - 1), ])
    Y[[i]] <- y[i + n_steps]
  }
  return(list(X = array(unlist(X), dim = c(length(X), n_steps, ncol(data))), Y = unlist(Y)))
}

# sum(is.na(dat_numeric))
# sum(is.na(dat_scaled))
# dat_numeric <- dat_numeric %>apply(dat_numeric, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
# dat_numeric <- na.omit(dat_numeric)
n_steps <- 10
sequences <- create_sequences(dat, y, n_steps)
X <- sequences$X
Y <- sequences$Y

# Split the data
split_index <- round(length(Y) * 0.7)
X_train <- X[1:split_index, , ]
Y_train <- Y[1:split_index]
X_test <- X[(split_index + 1):length(Y), , ]

Y_test <- Y[(split_index + 1):length(Y)]
sum(ncol(dat))
# sum(ncol(dat_scaled))
# sum(ncol(dat_numeric))

CPI_plt <- CPI_plt[(split_index+1):length(CPI_plt)]

# Define and compile the model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(n_steps, 12), return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = 'mean_absolute_error'
)

# dat_numeric <- dat_numeric
# dat_scaled <- dat_scaled
# Train the model
history <- model %>% fit(
  X_train, Y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)
#########
# Testing
#########

# Evaluate the model
model %>% evaluate(X_test, Y_test)

prediction <- model %>% predict(X_test)

# Adjust predictions
prediction <- prediction * sd(y) + mean(y)
actual_values <- Y_test * sd(y) + mean(y)


##########
# Plotting
##########

###########
#Save plots
###########
# only have turned on when you want to make plots
# Save the last plot to a specific directory
# Start PNG device driver to create a file
png(filename = "C:/Users/tormo/OneDrive/Skole/Masteroppgave/Plots/LSTM_mnd12.png", width = 800, height = 600)

# Your plotting commands
plot(actual_values, type = 'l', col = 'blue', ylim = c(min(c(actual_values, prediction)), max(c(actual_values, prediction))), xlab = 'Date', ylab = 'CPIAUSCL')
lines(prediction, col = 'red')
legend("topright", legend = c("Actual CPI", "Predicted CPI"), col = c("blue", "red"), lty = 1, cex = 0.7)
title(main = "Overlay of Actual and Predicted CPI mnd12", sub = "Red: Predicted, Blue: Actual")

# Turn off the device driver to actually create the file
dev.off()
