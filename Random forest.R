################
# Random forest
################
# random forest seem to be working well for the first observations but as time goes on there is a drop in performance
# the need for an moving average or bayesian updating would help the rf model improve significantly.
# unsure how the LSTM paper is able to get such great results with the rf model. 
# going to have to look into how to improve the model. rf should be prone to parametric prolifaration 
# and dimationality.

##########
# library 
##########

library(data.table)
library(randomForest)
library(ranger)
library(ggplot2)

######
# Data
######

split_point <- 0.8
split_index <- 577  #floor(nrow(low_dat) * split_point) #splitindex
forecast_horison <- 3


low_dat <- fread('C:/Users/tormo/OneDrive/Skole/Masteroppgave/Data/long_low_dat23.csv')



# Lag dependent variable
low_dat[, CPIAUCSL := shift(CPIAUCSL, forecast_horison, type = "lag")]

low_dat <- na.omit(low_dat)
sum(is.na(low_dat))

date_vec <- low_dat$date  # Assuming 'data' is your data frame or data table and 'date' is the column name
date_vec <- date_vec[3:length(date_vec)]  # Now you can correctly lag the dates
summary(low_dat$CPIAUCSL)

########
# Data 2
########

# Now, proceed with your data splitting and model fitting as before
# split_index <- floor(nrow(low_dat) * 0.7)
train_dat <- low_dat[1:split_index,]
test_dat <- low_dat[(split_index + 1):nrow(low_dat),]



###########################
# Model specification tests
###########################

sum(is.na(low_dat))
sum(is.na(train_dat))
rf_model <- randomForest(CPIAUCSL ~ ., data = train_dat, ntree = 500, mtry = sqrt(ncol(train_dat)-1))

# print(importance)
varImpPlot(rf_model)


selected_var <- c(
  'CPIULFSL',
  'GS5',
  'CES1021000001',
  'BAA',
  'CUSR0000SAC',
  'S.P.div.yield',
  'WPSFD49207',
  'CES0600000008',
  'CES3000000008',
  'CUSR0000SAD',
  'CPIMEDSL',
  'S.P.PE.ratio',
  'CUSR0000SA0L2',
  'CUSR0000SA0L5',
  'DDURRG3M086SBEA',
  'PCEPI',
  'CUSR0000SAS',
  'DSERRG3M086SBEA')

# selected_var <- c('DTCOLNVHFNM', 'PCEPI', 'WPSFD49207', 'INVEST', 'WPSFD49502', 'CUSR0000SAC','CES2000000008', 'CPIAUCSL','CPIMEDSL', 
#                   'DNDGRG3M086SBEA','CPITRNSL','BUSINVx','DTCTHFNM','CPIULFSL','M2SL','DSERRG3M086SBEA','
#                   CUSR0000SAS', 'M1SL')

# selected_var <- c('CLAIMSx','CE16OV','PAYEMS','USGOOD','CLF16OV','DPCERA3M086SBEA','HWIURATIO','HWI','USTPU','W875RX1','MANEMP','SRVPRD',
#                   'IPMAT', 'CPIULFSL')#,'RPI','CUMFNS','IPNMAT','USCONS', 'IPFPNSS', 'IPDCONGD','IPMANSICS')



##########
# RF model
##########

rf_formula <- as.formula(paste('CPIAUCSL ~', paste(selected_var, collapse = '+')))

rf_model <- randomForest(rf_formula, data = train_dat, ntree = 500, mtry = sqrt(length(selected_var)))

importance <- randomForest::importance(rf_model)

randomForest::varImpPlot(rf_model)

####################
# out of sample test 
####################
summary(test_dat)
str(test_dat)
predicted_values <- predict(rf_model, newdata = test_dat)
actual_values <- test_dat$CPIAUCSL
mse <- mean((actual_values - predicted_values)^2)
print(paste(mse))
print(predicted_values)


######
# RMSE
######


rmse <- sqrt((predicted_values - actual_values)^2)
print(mean(rmse))

# ##########
# # Plotting 
# ##########
# 
plot_data <- data.frame(
  Time = test_dat$d,
  Actual = test_dat$CPIAUCSL,
  Predicted = predicted_values
)

ggplot(plot_data, aes(x = Time)) +
  geom_line(aes(y = Actual, colour = "Actual")) +
  geom_line(aes(y = Predicted, colour = "Predicted")) +
  labs(title = "Actual vs. Predicted Values", x = "Time", y = "Value") +
  scale_colour_manual("", values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

###########
#Save plots
###########
# only have turned on when you want to make plots
# Save the last plot to a specific directory

ggsave("C:/Users/tormo/OneDrive/Skole/Masteroppgave/Plots/RFmnd 1 .png", plot = last_plot(), width = 10, height = 8, units = "in")

print(mean(rmse))
