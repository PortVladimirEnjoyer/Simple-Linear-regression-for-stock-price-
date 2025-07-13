#trying to predict stock price of a given ticker with a simple linear regression
#im going to use yfinance bcz it's easy and it doesnt need a key( also its just a simple
# model so i dont need to recreate the wheel or anything )

import yfinance as yf #yfinance for data
from sklearn.metrics import mean_squared_error , r2_score
#evaluation metrics , stick to simple MSE , R2 and MRE ( these are the ones i learnt about in lectures so it's the easiest for me )
from sklearn.linear_model import LinearRegression
import pandas as pd #panda is necessary to retrieve dat fromm yfinance easily
import numpy as np #to create a list out of the adj close price
import matplotlib.pyplot as plt #plotting our regression
from sklearn.model_selection import train_test_split #to split our data between training and testing



def get_price(  ticker_symbol , start_date=None , end_date=None , period=None , interval="1d"):
    data = yf.download(
        ticker_symbol,
        start = start_date,
        end = end_date,
        period=period,
        interval=interval,
        progress=False
    )
    if data.empty: #top notch error handeling right there
        print("no data retrive for{ticker_symbol}")



    if 'Adj Close' in data.columns:
        adj_close = data['Adj Close'].values
        #now i need to make the X variable in the linear regression : Y = a.X + b where X is a numerical index (cant use random here obviously)
        x = np.arange(len(adj_close)).reshape(-1 , 1) #reshaping (-1 , 1) bcs sklearn except a 2D data array
    elif 'Close' in data.columns: #sometimes adjusted close is not available so in that case i just use the Close price
      adj_close = data['Close'].values
      x = np.arange(len(adj_close)).reshape(-1 , 1)
    #and our y variable is simple the stock price
    y = adj_close

    #actually calculating the linear regression.
    linear_regression = LinearRegression() #inialising the model

    #here im going to split the data between a training and testing set to avoid overfitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #now time to make the model fit
    linear_regression.fit(x_train, y_train)

    #making predictions
    y_pred = linear_regression.predict(x_test) #prediction based on training data
    y_pred_real = linear_regression.predict(x) #prediction based on real data -> this is the one we will plot


    #deciding the slope a and the origin b -> y = ax + b
    slope  = linear_regression.coef_[0]
    origin = linear_regression.intercept_

    #lets now evaluate the model using mse ,  and R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #we can finally plot it and plot the evalutation metrix too
    print(f"MSE values :{mse}")
    print(f" value r2: {r2}")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Prices')
    plt.plot(x, y_pred_real, color='red', linewidth=2, label='Regression Line')
    plt.title(f'{ticker_symbol} Stock Price Linear Regression ({start_date} to {end_date})')
    plt.xlabel('Time Index (Days)')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    #we are not done yet , we still need a prediction - lets say that we want to predict the next day

    if ( len(x) > 0):   #we are saying that if there is data (len of x > 0)we take the last index and we predict it using our regression
      last_index = x[-1][0]
      next_index = last_index + 1
      predicted_price = linear_regression.predict(np.array([[next_index]])) #the actual prediction
      print(f"Predicted price for the next index ({next_index}): {predicted_price}")




get_price("AAPL" , '2020-01-01' , '2020-12-31' , period=None ,interval="1d")
