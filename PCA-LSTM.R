##########
# PCA-LSTM
##########

# This LSTM version is the one used to replicate the findings in the littereture
# The most important part is the feature selection shown. 
# It shows the features chosen for different
# levels of strictness of number of features.
# A lot of similarities to the other LSTM models in this repository

#########
# Library
#########

library(data.table)
library(ggplot2)
library(glmnet)
library(reshape2)
library(keras)
# library(tidyverse)
library(dplyr)
library(stats)


######
# Data
######
# Since the data need to be prepered in a certain way for a neural net, we need 
# to load the data in once more. We load and split between training and test 
# in the same way as before. The before processing the data we remove all data 
# that was not part of the feature selection, done with LASSO.
# Read and prepare the dataset
# Rearrange so 'date' is the first column if necessary
dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/complete_low_dat.csv')
forecast_horizon <- 3
split_index <- 600
# Lag dependent variable
dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horizon, type = "lag")]
dat <- na.omit(dat)
# dat$date <- NULL
y <- dat$CPIAUCSL

# PCA
# Exclude non-numeric columns
# numeric_dat <- select(dat, -date)  # Assuming 'date' column is non-numeric, adjust as needed

# Perform PCA
# pca_result <- prcomp(numeric_dat, scale. = TRUE)


# Extract the loadings (contribution of each original variable to each principal component)
# loadings <- pca_result$rotation

# top_contributors_pc1 <- sort(abs(loadings[, 1]), decreasing = TRUE)[1:80]
# names(top_contributors_pc1)

# 20 variables
# dat_list <- c("DMANEMP", "MANEMP", "CLF16OV", "TB6MS", "CP3Mx", "TB3MS", "FEDFUNDS",
#   "GS1", "NDMANEMP", "USWTRADE", "PCEPI", "USTRADE", "USFIRE", "GS5",
#   "CUMFNS", "BUSINVx", "USTPU", "DSERRG3M086SBEA", "SRVPRD", "CUSR0000SA0L5")
# 50 variables
# dat_list <- c("DMANEMP", "MANEMP", "CLF16OV", "TB6MS", "CP3Mx", "TB3MS", "FEDFUNDS",
#               "GS1", "NDMANEMP", "USWTRADE", "PCEPI", "USTRADE", "USFIRE", "BUSINVx",
#               "GS5", "USTPU", "SRVPRD", "CUMFNS", "CUSR0000SA0L5", "DSERRG3M086SBEA",
#               "CUSR0000SA0L2", "CES3000000008", "CES0600000008", "AAAFFM", "UEMP15OV",
#               "CPIULFSL", "BAAFFM", "CUSR0000SAC", "GS10", "CUSR0000SAS", "DDURRG3M086SBEA",
#               "UEMP15T26", "UEMPMEAN", "REALLN", "PAYEMS", "CPIAPPSL", "UEMP27OV",
#               "CPIAUCSL", "DNDGRG3M086SBEA", "CE16OV", "EXJPUSx", "S.P.div.yield",
#               "RETAILx", "S.P.PE.ratio", "IPFINAL", "IPFPNSS", "IPBUSEQ", "AAA",
#               "CES2000000008", "CUSR0000SAD")
# 80 variables
# dat_list <- c("DMANEMP", "MANEMP", "CLF16OV", "TB6MS", "CP3Mx", "TB3MS", "FEDFUNDS",
#               "GS1", "NDMANEMP", "USWTRADE", "PCEPI", "USTRADE", "USFIRE", "BUSINVx",
#               "GS5", "USTPU", "SRVPRD", "CUMFNS", "CUSR0000SA0L5", "DSERRG3M086SBEA",
#               "CUSR0000SA0L2", "CES3000000008", "CES0600000008", "AAAFFM", "UEMP15OV",
#               "CPIULFSL", "BAAFFM", "CUSR0000SAC", "GS10", "CUSR0000SAS", "DDURRG3M086SBEA",
#               "UEMP15T26", "UEMPMEAN", "REALLN", "PAYEMS", "CPIAPPSL", "UEMP27OV",
#               "CPIAUCSL", "DNDGRG3M086SBEA", "CE16OV", "EXJPUSx", "S.P.div.yield",
#               "RETAILx", "S.P.PE.ratio", "IPFINAL", "IPFPNSS", "IPBUSEQ", "AAA",
#               "CES2000000008", "CUSR0000SAD", "WPSFD49207", "HOUSTNE", "HOUSTMW",
#               "T10YFFM", "TB3SMFFM", "AMDMUOx", "EXSZUSx", "IPNCONGD", "WPSFD49502",
#               "UEMP5TO14", "CPITRNSL", "EXUSUKx", "USGOVT", "HOUST", "BAA", "WPSID61",
#               "ISRATIOx", "CPIMEDSL", "USGOOD", "PERMITNE", "INDPRO", "IPMANSICS",
#               "TB6SMFFM", "AMDMNOx", "DTCTHFNM", "T5YFFM", "IPNMAT", "PERMITMW",
#               "BOGMBASE", "W875RX1")

# dat$date <- as.numeric(as.POSIXct(dat$date, format = "%Y-%m-%d"))
# date <- dat$date
# dat$date <- NULL

# test_date <- dat[(split_index+1): nrow(dat),]
# dat$cpi
# important_variables <- c(
#   'HOUST',#: Housing Starts: Total New Privately Owned
#   'HOUSTNE',#: Housing Starts Northeast
#   'HOUSTMW',#: Housing Starts Midwest
#   'HOUSTS',#: Housing Starts South
#   'HOUSTW',#: Housing Starts West
#   'PERMIT',#: New Private Housing Permits (SAAR)
#   'PERMITNE',#: New Private Housing Permits Northeast (SAAR)
#   'PERMITMW',#: New Private Housing Permits Midwest (SAAR)
#   'PERMITS',#: New Private Housing Permits South (SAAR)
#   'PERMITW',#: New Private Housing Permits West (SAAR)
#   'AAAFFM',
#   'T10YFFM',#: 10-Year Treasury Constant Maturity Minus Federal Funds Rate
#   'T5YFFM',#: 5-Year Treasury Constant Maturity Minus Federal Funds Rate
#   'TB3MS',#: 3-Month Treasury Bill: Secondary Market Rate
#   'AWHMAN',#: Average Weekly Hours of Production and Nonsupervisory Employees, Manufacturing
#   'PAYEMS',#: All Employees, Total Nonfarm
#   'CPIULFSL',
#   'USFIRE'#: All Employees: Financial Activities
# )
# # dat$fi
################
# Model defining 
################

# Prepare the target variable
# CPI_plt <- dat$CPIAUCSL


# remove data not chosen in feature selection by LASSO
# dat <- dat[, c("date", dat_list), with = FALSE]

# Print the dimensions of the updated dataframe
print(dim(dat))
dat$date <- NULL
dat$CPIAUCSL <- NULL



# Create sequences
create_sequences <- function(data, y, forecast_horizon) {
  X <- list()
  Y <- list()
  for (i in 1:(nrow(data) - forecast_horizon)) {
    X[[i]] <- as.matrix(data[i:(i + forecast_horizon - 1), ])
    Y[[i]] <- y[i + forecast_horizon]
  }
  return(list(X = array(unlist(X), dim = c(length(X), forecast_horizon, ncol(data))), Y = unlist(Y)))
}


n_features <- length(dat)
print(n_features)

sequences <- create_sequences(dat, y, forecast_horizon)
X <- sequences$X
Y <- sequences$Y

# Split the data
X_train <- X[1:split_index, , ]
Y_train <- Y[1:split_index]
X_test <- X[(split_index + 1):length(Y), , ]
Y_test <- Y[(split_index + 1):length(Y)]
sum(ncol(dat))

####################
# training the model
####################

# Define model
model <- keras_model_sequential() %>%
  layer_lstm(units = 120, input_shape = c(forecast_horizon, n_features), return_sequences = TRUE) %>%
  # layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 70, return_sequences = TRUE) %>%
  layer_lstm(units = 70, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.3) %>%
  # layer_lstm(units = 40, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  # layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = 'mean_absolute_error'
)

# Specify callbacks
callbacks_list <- list(
  callback_model_checkpoint("path_to_save_model.h5", save_best_only = TRUE, monitor = "val_loss", verbose = 1),
  callback_early_stopping(patience = 100, restore_best_weights = TRUE, monitor = "val_loss", verbose = 1)
)

# Fit the model with callbacks
history <- model %>% fit(
  X_train, Y_train,
  epochs = 400,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callbacks_list
)


#########
# Testing
#########

# Evaluate the model
model %>% evaluate(X_test, Y_test)

prediction <- model %>% predict(X_test)

######
# RMSE
######


rmse <- sqrt((prediction - Y_test)^2)
print(mean(rmse))
rmse_df <- data.frame(rmse)

rmse_name <- paste('RMSE_LSTM', forecast_horizon, '.csv', sep = '')
# file_path <- paste('C:/Users/tormo/OneDrive/Skole/Masteroppgave/DM testing/LASSO LSTM short', rmse_name, sep='/')

# write.csv(rmse_df, file = file_path, row.names = FALSE)


##########
# Plotting
##########

###########
#Save plots
###########
# only have turned on when you want to make plots
# Save the last plot to a specific directory
# Start PNG device driver to create a file

# Plotting the actual vs. predicted CPI directly in R
plot(Y_test, type = 'l', col = 'blue', ylim = c(min(c(Y_test, prediction)), max(c(Y_test, prediction))), xlab = 'Date', ylab = 'CPIAUSCL')
lines(prediction, col = 'red')
legend("topright", legend = c("Actual CPI", "Predicted CPI"), col = c("blue", "red"), lty = 1, cex = 0.8) # You might need to adjust 'cex' based on your preference
title(main = paste("Overlay of Actual and Predicted CPI", forecast_horizon, "Months"), sub = "Red: Predicted, Blue: Actual")

# Plotting commands
plot(Y_test, type = 'l', col = 'blue', ylim = c(min(c(Y_test, prediction)), max(c(Y_test, prediction))), xlab = 'Date', ylab = 'CPIAUSCL')
lines(prediction, col = 'red')
legend("topright", legend = c("Actual CPI", "Predicted CPI"), col = c("blue", "red"), lty = 1, cex = 0.8) # Adjusted cex to a fixed value for simplicity
title(main = paste("Overlay of Actual and Predicted CPI", forecast_horizon, "Months"), sub = "Red: Predicted, Blue: Actual")

# Turn off the device driver to actually create the file
# dev.off()
print(mean(rmse))



