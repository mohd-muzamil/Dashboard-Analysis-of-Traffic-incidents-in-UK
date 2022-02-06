from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math


def predict_next(df, steps, time_range):
    df = df.reset_index()
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)
    return_df = df_train
    if time_range != 'Y':
        history = [x for x in df_train.iloc[:, -1]]
        model_predictions = []
        model = ARIMA(history, order=(1, 1, 0), seasonal_order=(0, 1, 0, 24))
        model_fit = model.fit()
        steps = df_test.index.size + steps
        predict_values = model_fit.get_forecast(steps=steps)
        # print(predict_values.summary_frame())
        [model_predictions.append(x) for x in predict_values.summary_frame()['mean']]

        date = df_train['Date'].iloc[-1]
        for predict in model_predictions:
            if time_range == 'W':
                date += relativedelta(days=+7)

            elif time_range == 'M':
                date += relativedelta(months=+1)

            # elif time_range == 'Y':
            #     date += relativedelta(years=+1)

            if predict < 0 or math.isnan(predict):
                predict = 0
            return_df = return_df.append({'Date': date, 'count': int(predict)}, ignore_index=True)

    model_error = metrics.mean_squared_error(df_test['count'].tolist(),
                                             return_df.loc[df_train.index.size+1:df.index.size, 'count'].tolist())
    return_df.set_index('Date', inplace=True)
    return return_df, df_train.index.size, model_error

