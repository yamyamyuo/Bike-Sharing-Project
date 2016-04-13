import pandas as pd
import numpy as np
import math
from ggplot import *
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

def dataprocess(df):
    # for centeralizing the temp, atemp, humidity, windspeed
    min_temp = df['temp'].min()
    max_temp = df['temp'].max()
    min_atemp = df['atemp'].min()
    max_atemp = df['atemp'].max()
    min_humidity = df['humidity'].min()
    max_humidity = df['humidity'].max()
    min_windspeed = df['windspeed'].min()
    max_windspeed = df['windspeed'].max()
    # seperate the date info
    df['year'] = pd.DatetimeIndex(df.datetime).year
    df['dayofweek'] = pd.DatetimeIndex(df.datetime).dayofweek
    df['hour'] = pd.DatetimeIndex(df.datetime).hour
    df['month'] = pd.DatetimeIndex(df.datetime).month
    df['weekofyear'] = pd.DatetimeIndex(df.datetime).weekofyear
    df['day'] = pd.DatetimeIndex(df.datetime).day
    # centeralize the data
    df['temp_centered'] = df['temp'].map(lambda x: (x - min_temp)*1.0/(max_temp - min_temp))
    df['atemp_centered'] = df['atemp'].map(lambda x: (x - min_atemp)*1.0/(max_atemp - min_atemp))
    df['humidity_centered'] = df['humidity'].map(lambda x: (x - min_humidity)*1.0/(max_humidity - min_humidity))
    df['windspeed_centered'] = df['windspeed'].map(lambda x: (x - min_windspeed)*1.0/(max_windspeed - min_windspeed))
    # log transformation
    # plus 1 can avoid if casual = 0, log(casual) = -inf
    df['temp_log'] = df['temp_centered'].map(lambda x: math.log(x+1,10))
    df['atemp_log'] = df['atemp_centered'].map(lambda x: math.log(x+1,10))
    df['humidity_log'] = df['humidity_centered'].map(lambda x: math.log(x+1,10))
    df['windspeed_log'] = df['windspeed_centered'].map(lambda x: math.log(x+1,10))

    df['ideal'] = df[['temp', 'windspeed']].apply(lambda x: (0, 1)[x['temp'] > 27 and x['windspeed'] < 30], axis = 1)
    df['sticky'] = df[['humidity', 'workingday']].apply(lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis = 1)

    return df



# plot
train_df = dataprocess(train_data)

# RandomForestRegressor to evaluate the feature_importances_
cols = ['day', 'hour', 'year', 'season', 'weather',
    'holiday', 'workingday', 'temp', 'atemp', 'humidity',
    'windspeed', 'month', 'temp_centered', 'temp_log'
]
rf = RandomForestRegressor(n_estimators=1000, min_samples_split=5, oob_score=True)
casual = rf.fit(train_df[cols], train_df.casual)
feature_importance =  rf.feature_importances_

sorted_idx = np.argsort(feature_importance)
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)

plt.yticks(pos, cols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

p0 = ggplot(aes(x='hour', y='casual', color='dayofweek'), data=train_df) +\
    stat_smooth(span=.2, se=False, size=2) +\
    geom_point() +\
    xlab("Hour of the day") +\
    ylab("Number of Bike Rentals") +\
    ggtitle("Casual Rental Trend by Day of Week and Time")
print p0

p1 = ggplot(aes(x='hour', y='registered', color='dayofweek'), data=train_df) +\
    stat_smooth(span=.2, se=False, size=2) +\
    geom_point() +\
    xlab("Hour of the day") +\
    ylab("Number of Bike Rentals") +\
    ggtitle("Registered Rental Trend by Day of Week and Time")
print p1

p2 = ggplot(aes(x='hour', y='count', color='dayofweek'), data=train_df) +\
    stat_smooth(span=.2, se=False, size=4) +\
    xlab("Hour of the day") +\
    ylab("Number of Bike Rentals") +\
    ggtitle("Rental Trend by Day of Week and Time")
print p2
