# Load necessary libraries
library(quantmod)
library(forecast)
library(tseries)
library(caret)
library(ggplot2)
library(data.table)
library(TTR)
library(lubridate)
library(keras)
library(tensorflow)
library(randomForest)
library(rpart)

# Load stock data
stock_data <- getSymbols("AAPL", src = "yahoo", from = "2015-01-01", to = "2023-12-31", auto.assign = FALSE)

# Use the Adjusted Close price
adj_close <- stock_data[, 6]

# Check for missing values
missing_values <- sum(is.na(adj_close))
print(paste("Missing values:", missing_values))


# Plot the data
plot(adj_close, main = "Adjusted Close Price of AAPL", ylab = "Price", xlab = "Date")

# Decompose the time series
adj_close_ts <- ts(adj_close, frequency = 252)
decomposed <- decompose(adj_close_ts, type = "multiplicative")

# Plot the decomposed components
plot(decomposed)

# Holt-Winters Forecasting
hw_model <- HoltWinters(adj_close_ts, seasonal = "multiplicative")
hw_forecast <- forecast(hw_model, h = 60)

# Plot the Holt-Winters forecast
plot(hw_forecast, main = "Holt-Winters Forecast")

# Auto ARIMA model
arima_model <- auto.arima(adj_close_ts, seasonal = TRUE)
arima_forecast <- forecast(arima_model, h = 60)

# Plot the ARIMA forecast
plot(arima_forecast, main = "ARIMA Forecast")

# Evaluate the model
train_end <- floor(0.8 * length(adj_close_ts))
train_data <- adj_close_ts[1:train_end]
test_data <- adj_close_ts[(train_end + 1):length(adj_close_ts)]

# Refit the ARIMA model on the training data
arima_model <- auto.arima(train_data, seasonal = TRUE)
arima_forecast <- forecast(arima_model, h = length(test_data))

# Plot the forecast
plot(arima_forecast)
lines(test_data, col = "red")

# Calculate evaluation metrics
arima_rmse <- sqrt(mean((test_data - arima_forecast$mean)^2))
arima_mae <- mean(abs(test_data - arima_forecast$mean))
arima_mape <- mean(abs((test_data - arima_forecast$mean) / test_data)) * 100
arima_r2 <- 1 - sum((test_data - arima_forecast$mean)^2) / sum((test_data - mean(test_data))^2)

print(paste("ARIMA RMSE:", arima_rmse))
print(paste("ARIMA MAE:", arima_mae))
print(paste("ARIMA MAPE:", arima_mape))
print(paste("ARIMA R-squared:", arima_r2))

# Preparing data for LSTM, Random Forest, and Decision Tree
adj_close_df <- data.frame(Date = index(adj_close), Adj_Close = as.numeric(adj_close))
adj_close_df$Lag_1 <- lag(adj_close_df$Adj_Close, 1)
adj_close_df$Lag_2 <- lag(adj_close_df$Adj_Close, 2)
adj_close_df$Lag_3 <- lag(adj_close_df$Adj_Close, 3)
adj_close_df$Lag_4 <- lag(adj_close_df$Adj_Close, 4)
adj_close_df$Lag_5 <- lag(adj_close_df$Adj_Close, 5)
# Remove NA values
adj_close_df <- na.omit(adj_close_df)

# Split the data into training and test sets
train_index <- 1:floor(0.8 * nrow(adj_close_df))
train_data <- adj_close_df[train_index, ]
test_data <- adj_close_df[-train_index, ]



# Random Forest model
library(randomForest)
rf_model <- randomForest(Adj_Close ~ Lag_1 + Lag_2 + Lag_3 + Lag_4 + Lag_5, data = train_data)
rf_predictions <- predict(rf_model, test_data)

# Evaluate the Random Forest model
rf_rmse <- sqrt(mean((test_data$Adj_Close - rf_predictions)^2))
rf_mae <- mean(abs(test_data$Adj_Close - rf_predictions))
rf_mape <- mean(abs((test_data$Adj_Close - rf_predictions) / test_data$Adj_Close)) * 100
rf_r2 <- 1 - sum((test_data$Adj_Close - rf_predictions)^2) / sum((test_data$Adj_Close - mean(test_data$Adj_Close))^2)

print(paste("Random Forest RMSE:", rf_rmse))
print(paste("Random Forest MAE:", rf_mae))
print(paste("Random Forest MAPE:", rf_mape))
print(paste("Random Forest R-squared:", rf_r2))

# Decision Tree model
library(rpart)
dt_model <- rpart(Adj_Close ~ Lag_1 + Lag_2 + Lag_3 + Lag_4 + Lag_5, data = train_data)
dt_predictions <- predict(dt_model, test_data)

# Evaluate the Decision Tree model
dt_rmse <- sqrt(mean((test_data$Adj_Close - dt_predictions)^2))
dt_mae <- mean(abs(test_data$Adj_Close - dt_predictions))
dt_mape <- mean(abs((test_data$Adj_Close - dt_predictions) / test_data$Adj_Close)) * 100
dt_r2 <- 1 - sum((test_data$Adj_Close - dt_predictions)^2) / sum((test_data$Adj_Close - mean(test_data$Adj_Close))^2)

print(paste("Decision Tree RMSE:", dt_rmse))
print(paste("Decision Tree MAE:", dt_mae))
print(paste("Decision Tree MAPE:", dt_mape))
print(paste("Decision Tree R-squared:", dt_r2))





