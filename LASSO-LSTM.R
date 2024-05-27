############
# LASSO-LSTM
############

#########
# Library
#########

library(data.table)
library(ggplot2)
library(glmnet)
library(reshape2)
library(keras)
library(tidyverse)
library(dplyr)

######
# Data
######

low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/long_low_dat23.csv')


# We put in some features that we got. You can adjust lambda to find your own
# specification of features given dataset. Below youll find a range of
# features based on strictness of lambda and forecast horizons.


# small_important_variables <- c("W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "INDPRO", "IPCONGD", "IPNCONGD",
#                                "IPBUSEQ", "IPDMAT", "IPB51222S", "HWI", "CLF16OV", "USCONS", "SRVPRD",
#                                "USFIRE", "USGOVT", "AMDMNOx", "ANDENOx", "AMDMUOx", "M1SL", "M2SL" 'date')
# important_variables <- small_important_variables
# 
# 
# low_dat <- low_dat[, c(important_variables), with = FALSE]
# write.csv(low_dat, file = 'C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/EIKON/LASSO.csv')


# Adjust split index based on the forecast horizon. As the forecast horizon
# changes the observation equal to the 2010-1-1 changes. Set split index
# such that it equals the 2010-1-1 observation.

# constant variables
split_point <- 0.8
split_index <- 553 #floor(nrow(low_dat) * split_point) #splitindex
forecast_horizon <- 48

# Lag dependent variable
low_dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horizon, type = "lag")]
# low_dat[, test_CPI := shift(test_CPI, (forecast_horizon +1), type = "lag")]

low_dat <- na.omit(low_dat)
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]

# Since differencing introduces NA values in the first row, remove NAs
low_dat <- na.omit(low_dat)
low_dat$date <- NULL
low_dat$da


################
# Training model
################

y_train <- train_dat$CPIAUCSL

X_train <- train_dat %>%
  select(-date, -CPIAUCSL) %>%
  as.matrix()


print(colnames(X_train))
cv_model <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE)


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


best_lambda <- cv_model$lambda.min #lambda optimal based on function
best_lambda <- 0.005 # If I do not totally agree with the model
best_lambda_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)
print(best_lambda)
print(log(0.005))
print(sqrt(3.1))


# # Extracting non-zero coefficient names and values
selected_variables <- rownames(coef(best_lambda_model))[-1] # Exclude intercept term
coefficients <- coef(best_lambda_model)[-1] # Exclude intercept term

# Calculate absolute coefficients
absolute_coefficients <- abs(coefficients)

# Filter variables with absolute coefficients larger than 0.001
important_variables <- selected_variables[absolute_coefficients > 0.000001]

# Print the list of important variables
print(important_variables)

plot(cv_model)
######
# LSTM 
######

######
# Data
######
# Since the data need to be prepered in a certain way for a neural net, we need 
# to load the data in once more. We load and split between training and test 
# in the same way as before. The before processing the data we remove all data 
# that was not part of the feature selection, done with LASSO.


###########
# 1,3,6 MND
###########

# small_important_variables <- c("CMRMTSPLx", "IPDCONGD", "IPNMAT", "HWI", "HWIURATIO", "NONREVSL", "WPSFD49207", "WPSID61", "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L5", "DSERRG3M086SBEA", "CES3000000008", "VIXCLSx", "S.P.PE.ratio", "PERMITNE")
# important_variables <- small_important_variables

# medium_important_variables <- c("RPI", "CMRMTSPLx", "RETAILx", "IPDCONGD", "IPNMAT", "IPB51222S", "IPFUELS", "USTPU", "USTRADE", "USGOVT", "AMDMNOx", "BUSINVx", "M1SL", "M2REAL", "BOGMBASE", "NONBORRES", "BUSLOANS", "NONREVSL", "S.P.500", "WPSFD49207", "WPSID61", "WPSID62", "PPICMM", "CPIAPPSL", "CUSR0000SAD", "CUSR0000SAS", "CUSR0000SA0L5", "DSERRG3M086SBEA", "CES2000000008", "CES3000000008", "UMCSENTx", "DTCTHFNM", "VIXCLSx", "EXUSUKx", "BAAFFM", "BAA", "S.P.PE.ratio", "S.P.div.yield", "PERMITW", "HOUSTW", "NDMANEMP", "UEMP15T26", "UEMPMEAN")
# important_variables <- medium_important_variables

# large_important_variables <- c("RPI", "DPCERA3M086SBEA", "CMRMTSPLx", "RETAILx", "IPDCONGD", "IPBUSEQ", "IPB51222S", "IPFUELS", "CLF16OV", "CE16OV", "USGOOD", "USCONS", "USTPU", "USTRADE", "USGOVT", "AMDMNOx", "AMDMUOx", "BUSINVx", "M2REAL", "BOGMBASE", "NONBORRES", "BUSLOANS", "REALLN", "NONREVSL", "S.P.500", "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "PPICMM", "CPIAPPSL", "CPIMEDSL", "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L5", "DSERRG3M086SBEA", "CES0600000008", "CES2000000008", "CES3000000008", "UMCSENTx", "DTCTHFNM", "INVEST", "VIXCLSx", "EXCAUSx", "EXUSUKx", "BAAFFM", "BAA", "S.P.PE.ratio", "S.P.div.yield", "PERMITMW", "PERMITNE", "CES0600000007", "NDMANEMP", "DMANEMP", "MANEMP", "UEMP15T26", "UEMPMEAN", "UNRATE")
# important_variables <- large_important_variables
##############
# 12,24,48 MND
##############

# small_important_variables <- c("W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "INDPRO", "IPCONGD", "IPNCONGD",
# "IPBUSEQ", "IPDMAT", "IPB51222S", "HWI", "CLF16OV", "USCONS", "SRVPRD",
# "USFIRE", "USGOVT", "AMDMNOx", "ANDENOx", "AMDMUOx", "M1SL", "M2SL")#, 'Date')
# important_variables <- small_important_variables

# medium_important_variables <- c("W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "INDPRO", "IPCONGD", "IPNCONGD",
# "IPBUSEQ", "IPDMAT", "IPB51222S", "HWI", "CLF16OV", "USCONS", "SRVPRD",
# "USFIRE", "USGOVT", "AMDMNOx", "ANDENOx", "AMDMUOx", "M1SL", "M2SL",
# "BOGMBASE", "NONBORRES", "REALLN", "S.P.500", "WPSFD49502", "WPSID61",
# "WPSID62", "OILPRICEx", "PPICMM", "CPIAPPSL", "CPIMEDSL", "CUSR0000SAC",
# "CUSR0000SAS", "DDURRG3M086SBEA", "DSERRG3M086SBEA", "CES2000000008",
# "CES3000000008", "UMCSENTx", "DTCOLNVHFNM", "DTCTHFNM")
# important_variables <- medium_important_variables

# large_important_variables <- c("W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", "INDPRO", "IPCONGD", "IPNCONGD",
#              "IPBUSEQ", "IPDMAT", "IPB51222S", "HWI", "CLF16OV", "USCONS", "SRVPRD",
#              "USFIRE", "USGOVT", "AMDMNOx", "ANDENOx", "AMDMUOx", "M1SL", "M2SL",
#              "BOGMBASE", "NONBORRES", "REALLN", "S.P.500", "WPSFD49502", "WPSID61",
#              "WPSID62", "OILPRICEx", "PPICMM", "CPIAPPSL", "CPIMEDSL", "CUSR0000SAC",
#              "CUSR0000SAS", "DDURRG3M086SBEA", "DSERRG3M086SBEA", "CES2000000008",
#              "CES3000000008", "UMCSENTx", "DTCOLNVHFNM", "DTCTHFNM", "VIXCLSx",
#              "EXCAUSx", "EXUSUKx", "EXSZUSx", "TWEXAFEGSMTHx", "TB6SMFFM", "TB3SMFFM",
#              "COMPAPFFx", "BAA", "TB6MS", "S.P.PE.ratio", "S.P.div.yield", "ISRATIOx",
#              "PERMITW", "HOUSTMW", "HOUSTS", "AWOTMAN", "CES0600000007", "DMANEMP",
#              "CLAIMSx", "UEMP27OV", "UEMPLT5", "UEMPMEAN", "UNRATE")
# important_variables <- large_important_variables



# long_fin_low_dat.csv variables
important_variables <- c("V1", "BakerHughes", "Coffee", "HARPEX", "Oil", "Silver", "Agriculture", 
                         "PaulVolcker", "OilCrisis", "GFC", "Covid", "W875RX1", "DPCERA3M086SBEA", "CMRMTSPLx", 
                         "INDPRO", "IPCONGD", "IPNCONGD", "IPBUSEQ", "IPDMAT", "IPB51222S", "HWI", "CLF16OV", 
                         "USCONS", "SRVPRD", "USFIRE", "USGOVT", "AMDMNOx", "ANDENOx", "AMDMUOx", "M1SL", "M2SL", 
                         "CPI")


# Read and prepare the dataset
# Rearrange so 'date' is the first column if necessary
dat <- low_dat[, c(important_variables), with = FALSE]
# write.csv(dat, file = 'C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/EIKON/LASSO.csv')
  # fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/long_low_dat23.csv')

# Lag dependent variable
# dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horizon, type = "lag")]
# dat[, test_CPI := shift(test_CPI, (forecast_horizon +1), type = "lag")]


dat$date <- NULL
dat <- na.omit(dat)
# Convert 'date' column to timestamps
dat$date <- as.numeric(as.POSIXct(dat$date, format = "%Y-%m-%d"))

# test_date <- dat$date[split_index+1: nrow(dat)]

################
# Model defining 
################

# Prepare the target variable
# CPI_plt <- dat$CPIAUCSL
y <- low_dat$CPIAUCSL
# dat$CPIAUCSL <- NULL
# dat$date <- NULL
# remove data not chosen in feature selection by LASSO
# dat <- dat[, c("date", important_variables), with = FALSE]

# Print the dimensions of the updated dataframe
print(dim(dat))


dat$date <- NULL


create_sequences <- function(data, y, sequence_length, horizon) {
  X <- list()
  Y <- list()
  for (i in 1:(nrow(data) - sequence_length - horizon + 1)) {
    X[[i]] <- as.matrix(data[i:(i + sequence_length - 1), ])
    Y[[i]] <- y[i + sequence_length + horizon - 1]  # Notice the indexing for Y
  }
  return(list(X = array(unlist(X), dim = c(length(X), sequence_length, ncol(data))), Y = unlist(Y)))
}


n_features <- length(important_variables)
print(n_features)

sequences <- create_sequences(dat, y, 24, forecast_horizon)

X <- sequences$X
Y <- sequences$Y


X_train <- X[1:split_index, , ]
Y_train <- Y[1:split_index]
test_start_index <- split_index + 1
X_test <- X[test_start_index:length(Y), , ]
Y_test <- Y[test_start_index:length(Y)]


sum(ncol(dat))
print(colnames(dat))
dat$date <- NULL
print(colnames(dat))

length(Y_test)

dim(X_test)

###################
# Running the model
###################

# Adjust hyperparameters based on results from validation RMSE. 


dropout_rate <- 0.1
model_units <- length(important_variables)
# Define and compile the model
model <- keras_model_sequential() %>%
  layer_lstm(units = model_units, input_shape = c(24, n_features), return_sequences = TRUE) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_lstm(units = model_units, return_sequences = TRUE) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_lstm(units = model_units, return_sequences = FALSE) %>%
  # layer_dropout(rate = dropout_rate) %>%
  layer_dropout(rate = dropout_rate) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(lr = 0.001),#'adam',
  loss = 'mean_squared_error',
  metrics = 'mean_absolute_error'
)


# Train the model
callback_early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 500, restore_best_weights = TRUE)
history <- model %>% fit(
  X_train, Y_train,
  epochs = 2000,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping)
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
file_path <- paste('C:/Users/tormo/OneDrive/Skole/Masteroppgave/DM testing/LASSO LSTM short', rmse_name, sep='/')

write.csv(rmse_df, file = file_path, row.names = FALSE)


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




# saving the plot
plot_name <- paste('CPI_Prediction_', forecast_horizon, 'months.png', sep = '')
plot_file_path <- paste('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Plots/', plot_name, sep='')

# Start PNG device driver to create the file with the dynamic name
png(filename = plot_file_path, width = 800, height = 600)

# Plotting commands

print(mean(rmse))
