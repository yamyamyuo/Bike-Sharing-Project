import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from sklearn import decomposition
from ggplot import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
from sklearn import linear_model

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
    # one hot encoding the season features
    df['spring'] = df['season'].map(lambda x: 1 if x == 1 else 0)
    df['summer'] = df['season'].map(lambda x: 1 if x == 2 else 0)
    df['fall'] = df['season'].map(lambda x: 1 if x == 3 else 0)
    df['winter'] = df['season'].map(lambda x: 1 if x == 4 else 0)
    # peak
    df['peak'] = df[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 12)) or (x['workingday'] == 0 and  10 <= x['hour'] <= 19)], axis = 1)

    return df

def get_rmsle(count, true_count):
    count = count.astype(np.int)
    true_count = count.astype(np.int)
    n = len(true_count)
    summation_arg = (np.log(count+1.) - np.log(true_count+1.))**2.
    rmsle = np.sqrt(np.sum(summation_arg)/n)
    print 'RMSLE', rmsle, '\n'

def custom_train_test_split(data, cutoff_day=15):
    train = data[data['day'] <= cutoff_day]
    test = data[data['day'] > cutoff_day]

    return train, test

train_df = dataprocess(train_data)
for col in ['casual', 'registered','count']:
    train_df['%s_log' % col] = np.around(np.log(train_df[col] + 1))

# delete datetime to fit PCA
train_df = train_df.drop(['datetime'], axis = 1)
# PCA
pca = decomposition.PCA()
pca.fit(train_df)

# for training validation
train_vali, test_vali = custom_train_test_split(train_df)
print "Validation"
########### for Logistic Regression validation ####################
#cols = ['dayofweek', 'hour', 'year', 'season', 'weather', 'holiday', 'workingday', 'temp_centered', 'atemp_centered','humidity_centered','windspeed_centered','temp_log','atemp_log','humidity_log','windspeed_log','ideal', 'sticky']
#RMSLE 1.83271257725
cols = ['weather', 'temp', 'atemp', 'windspeed','workingday', 'season',
'holiday', 'sticky','hour', 'dayofweek', 'weekofyear', 'year', 'peak', 'ideal']
#RMSLE 1.59026287511
# after casual_log and registered_log RMSLE 1.01149361642
logistic_regression = linear_model.LogisticRegression(fit_intercept=True)
model_vali_casual = logistic_regression.fit(train_vali[cols], train_vali.casual_log)
predict_vali_casual = logistic_regression.predict(test_vali[cols])

model_vali_registered = logistic_regression.fit(train_vali[cols], train_vali.registered_log)
predict_vali_registered = logistic_regression.predict(test_vali[cols])

predict_vali_count = np.around(np.exp(predict_vali_casual) + np.exp(predict_vali_registered) - 2)
predict_vali_count[np.where(predict_vali_count < 0.)] = 0.

count = predict_vali_count.astype(np.int)
true_count = test_vali['count'].astype(np.int)
n = len(true_count)
summation_arg = (np.log(count+1.) - np.log(true_count+1.))**2.
rmsle = np.sqrt(np.sum(summation_arg)/n)
print 'RMSLE', rmsle, '\n'

################## for Logistic Regression submisstion ##################
print Logistic Regression
cols = ['weather', 'temp', 'atemp', 'windspeed','workingday', 'season',
'holiday', 'sticky','hour', 'dayofweek', 'weekofyear', 'year', 'peak', 'ideal',
'temp_log', 'humidity_log','windspeed_log', 'humidity', 'atemp_log']
#RMSLE 0.99624
#Rank 2700
logistic_regression = linear_model.LogisticRegression(fit_intercept=True)
test_df = dataprocess(test_data)
model_casual = logistic_regression.fit(train_df[cols], train_df.casual_log)
predict_casual = logistic_regression.predict(test_df[cols])

model_registered = logistic_regression.fit(train_df[cols], train_df.registered_log)
predict_registered = logistic_regression.predict(test_df[cols])

predict_count = [int(round(math.exp(i)+math.exp(j)-2)) for i,j in zip(predict_casual, predict_registered)]
#out put submisstion
test_df['count'] = predict_count
final_submit = test_df[['datetime', 'count']].copy()
final_submit.to_csv('./mannalogistic.csv', index=False)

########### for validation SVR ####################
#SVR
cols = ['dayofweek', 'hour', 'year', 'season', 'weather', 'holiday', 'workingday',
 'temp_centered', 'atemp_centered','humidity_centered','windspeed_centered',
 'temp_log','atemp_log','humidity_log','windspeed_log','ideal', 'sticky','peak']
# RMSLE 0.407266584354

# cols = ['weather', 'temp', 'atemp', 'windspeed','workingday', 'season', 'holiday', 'sticky','hour', 'dayofweek', 'weekofyear', 'year']
# RMSLE = 0.987

#change of C = [1, 1e1, 1e2, 1e3]
# RMSLE 0.598082214828
# RMSLE 0.43669504309
# RMSLE 0.419205828232
# RMSLE 0.468913066599

# change of gamma = [10, 8, 6, 4, 2, 1, 0.1, 0.001, 0.0001] C= 1e2
# RMSLE 1.15714668935
# RMSLE 1.12928540956
# RMSLE 1.09320798656
# RMSLE 1.0275230323
# RMSLE 0.836737756567
# RMSLE 0.612638041047
# RMSLE 0.419205828232
# RMSLE 0.967700899828
# RMSLE 1.1638615511
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)
model_vali_casual = svr_rbf.fit(train_vali[cols], train_vali.casual)
predict_vali_casual = svr_rbf.predict(test_vali[cols])

model_vali_registered = svr_rbf.fit(train_vali[cols], train_vali.registered)
predict_vali_registered = svr_rbf.predict(test_vali[cols])

predict_vali_count = np.around(predict_vali_casual + predict_vali_registered)
predict_vali_count[np.where(predict_vali_count < 0.)] = 0.
# predict_vali_count = [int(round(i+j)) for i,j in zip(predict_vali_casual, predict_vali_registered)]

count = predict_vali_count.astype(np.int)
# count = predict_vali_count
true_count = test_vali['count'].astype(np.int)
n = len(true_count)
summation_arg = (np.log(count+1.) - np.log(true_count+1.))**2.
rmsle = np.sqrt(np.sum(summation_arg)/n)
print 'RMSLE', rmsle, '\n'


print "for submission"
################## for SVR submission#############################
cols = ['dayofweek', 'hour', 'year', 'season', 'weather', 'holiday', 'workingday',
 'temp_centered', 'atemp_centered','humidity_centered','windspeed_centered',
 'temp_log','atemp_log','humidity_log','windspeed_log','ideal', 'sticky','peak']
#rank 819
#score 0.44784
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)

test_df = dataprocess(test_data)
model_casual = svr_rbf.fit(train_df[cols], train_df.casual_log)
predict_casual = svr_rbf.predict(test_df[cols])
predict_casual[np.where(predict_casual < 0.)] = 0.

model_registered = svr_rbf.fit(train_df[cols], train_df.registered_log)
predict_registered = svr_rbf.predict(test_df[cols])
predict_registered[np.where(predict_registered < 0.)] = 0.

predict_count_svr = [int(round(math.exp(i)+math.exp(j)-2)) for i,j in zip(predict_casual, predict_registered)]

#out put submisstion
test_df['count'] = predict_count_svr

final_submit = test_df[['datetime', 'count']].copy()
final_submit.to_csv('./mannasvr.csv', index=False)


################## for Bayesian Ridge Regression #############################
cols = ['dayofweek', 'hour', 'year', 'season', 'weather', 'holiday', 'workingday',
 'temp_centered', 'atemp_centered','humidity_centered','windspeed_centered',
 'temp_log','atemp_log','humidity_log','windspeed_log','ideal', 'sticky','peak']
# RMSLE 0.91148
# tune from RMSLE 1.24938739607 to 0.9114

bayes_Ridge = linear_model.BayesianRidge(alpha_1=1.e-6, lambda_1=1.e-6, compute_score=True, fit_intercept=True, copy_X=True)
model_vali_casual = bayes_Ridge.fit(train_vali[cols], train_vali.casual_log)
predict_vali_casual = bayes_Ridge.predict(test_vali[cols])

model_vali_registered = bayes_Ridge.fit(train_vali[cols], train_vali.registered_log)
predict_vali_registered = bayes_Ridge.predict(test_vali[cols])

predict_vali_count = np.around(np.exp(predict_vali_casual) + np.exp(predict_vali_registered)-2)
predict_vali_count[np.where(predict_vali_count < 0.)] = 0.

count = predict_vali_count.astype(np.int)
true_count = test_vali['count'].astype(np.int)
n = len(true_count)
summation_arg = (np.log(count+1.) - np.log(true_count+1.))**2.
rmsle = np.sqrt(np.sum(summation_arg)/n)
print 'RMSLE', rmsle, '\n'


################## for Adaboost ##################
#rank 2700
#score: 0.92493
cols = [
    'weather', 'temp', 'atemp', 'windspeed', 'workingday', 'season', 'holiday', 'sticky',
    'hour', 'dayofweek', 'weekofyear', 'year'
]
# cols = ['dayofweek', 'hour', 'year', 'season', 'weather', 'holiday', 'workingday',
#  'temp_centered', 'atemp_centered','humidity_centered','windspeed_centered',
#  'temp_log','atemp_log','humidity_log','windspeed_log','ideal', 'sticky','peak']

ada = AdaBoostClassifier(learning_rate=1,n_estimators=400,algorithm="SAMME.R")
test_df = dataprocess(test_data)
model_casual = ada.fit(train_df[cols], train_df.casual_log)
predict_casual = ada.predict(test_df[cols])

model_registered = ada.fit(train_df[cols], train_df.registered_log)
predict_registered = ada.predict(test_df[cols])

predict_count = [int(round(math.exp(i)+math.exp(j)-2)) for i,j in zip(predict_casual, predict_registered)]
#out put submisstion
test_df['count'] = predict_count
final_submit = test_df[['datetime', 'count']].copy()
final_submit.to_csv('./mannaada.csv', index=False)


################## for RandomForestRegressor #############################

#cols = ['weather', 'temp', 'atemp', 'windspeed','workingday', 'season',
# 'holiday', 'sticky','hour', 'dayofweek', 'weekofyear', 'year']
# this features without log
# rank 840
# score 0.44967

cols = [
    'dayofweek', 'hour', 'year', 'season',
    'holiday', 'workingday', 'temp_centered',
    'atemp_centered', 'humidity_centered',
    'windspeed_centered', 'temp_log', 'atemp_log',
    'humidity_log', 'windspeed_log','peak'
]
# without log
# rank 475
# 0.42163

# with log
# rank 470
# score 0.42144
rf_estimator = RandomForestRegressor(n_estimators=1000, min_samples_split=5, max_depth=15,oob_score=True, n_jobs=5,random_state=0)
test_df = dataprocess(test_data)
model_casual = rf_estimator.fit(train_df[cols], train_df.casual_log)
predict_casual = rf_estimator.predict(test_df[cols])

model_registered = rf_estimator.fit(train_df[cols], train_df.registered_log)
predict_registered = rf_estimator.predict(test_df[cols])

predict_count_rf = [int(round(math.exp(i)+math.exp(j)-2)) for i,j in zip(predict_casual, predict_registered)]

#out put submisstion
test_df['count'] = predict_count_rf
final_submit = test_df[['datetime', 'count']].copy()
final_submit.to_csv('./mannarandomforest.csv', index=False)


################## for GradientBoostingRegressor #############################
# cols = [
#     'weather', 'temp', 'atemp', 'humidity', 'windspeed',
#     'holiday', 'workingday', 'season',
#     'hour', 'dayofweek', 'year', 'ideal',
# ]
# and without casual_log
# rank 2100
# score 0.57051
# cols = [
#     'dayofweek', 'hour', 'year', 'season',
#     'holiday', 'workingday', 'temp_centered',
#     'atemp_centered', 'humidity_centered',
#     'windspeed_centered', 'temp_log', 'atemp_log',
#     'humidity_log', 'windspeed_log','peak'
# ]
# rank 700
# score 0.43508
cols = [
    'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'season',
    'hour', 'dayofweek', 'year', 'ideal','peak'
]
#rank 90
#score 0.37127

gbm_estimator = GradientBoostingRegressor(n_estimators=150,max_depth=5,random_state=0,min_samples_leaf=10,learning_rate=0.1,subsample=0.7,loss='ls')
test_df = dataprocess(test_data)
model_casual = gbm_estimator.fit(train_df[cols], train_df.casual_log)
predict_casual = gbm_estimator.predict(test_df[cols])
predict_casual[np.where(predict_casual < 0.)] = 0.

model_registered = gbm_estimator.fit(train_df[cols], train_df.registered_log)
predict_registered = gbm_estimator.predict(test_df[cols])
predict_registered[np.where(predict_registered < 0.)] = 0.

predict_count_gbm = [int(round(math.exp(i)+math.exp(j)-2)) for i,j in zip(predict_casual, predict_registered)]

#out put submisstion
test_df['count'] = predict_count_gbm
final_submit = test_df[['datetime', 'count']].copy()
final_submit.to_csv('./mannagbm.csv', index=False)


################## for blending two models #############################
# blending random forest and GradientBoostingRegressor
# blending_predict = [int(round(.2*i+.8*j)) for i,j in zip(predict_count_rf, predict_count_gbm)]
# test_df['count'] = blending_predict
# final_submit = test_df[['datetime', 'count']].copy()
# final_submit.to_csv('./mannablending.csv', index=False)

for w in range(0,11):
    xi = .1*w
    xj = 1-0.1*w
    blending_predict = [int(round(xi*i+xj*j)) for i,j in zip(predict_count_rf, predict_count_gbm)]
    test_df['count'] = blending_predict
    final_submit = test_df[['datetime', 'count']].copy()
    final_name = './'+str(w)+'.csv'
    final_submit.to_csv(final_name, index=False)

################## for blending two models #############################
# blending svr and GradientBoostingRegressor

for w in range(0,11):
    xi = .1*w
    xj = 1-0.1*w
    blending_predict = [int(round(xi*i+xj*j)) for i,j in zip(predict_count_svr, predict_count_gbm)]
    test_df['count'] = blending_predict
    final_submit = test_df[['datetime', 'count']].copy()
    final_name = './'+str(w)+'.csv'
    final_submit.to_csv(final_name, index=False)


################## for blending two models #############################
# blending svr and random forest

for w in range(0,11):
    xi = .1*w
    xj = 1-0.1*w
    blending_predict = [int(round(xi*i+xj*j)) for i,j in zip(predict_count_svr, predict_count_rf)]
    test_df['count'] = blending_predict
    final_submit = test_df[['datetime', 'count']].copy()
    final_name = './'+str(w)+'.csv'
    final_submit.to_csv(final_name, index=False)

