#imports for fileupload
from imp import init_frozen
from flask import Flask, jsonify, request, Response, redirect
from werkzeug.utils import secure_filename
import os

#imports for ml purpose
import pandas as pd
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools

#rcparams for trend, seasonality, noise graph
from pylab import rcParams
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

#created a flask application
app = Flask(__name__)

#file upload - getting post request from angular-flask
app.config['UPLOAD_FOLDER'] = 'D:/SHIVA/angular/sales-prediction'
@app.route('/upload',methods = ['GET', 'POST'])
def upload_File():

    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        global fname
        fname = secure_filename(file.filename)    
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
        
        return redirect(request.url)

    return 'File Uploaded'

@app.route('/forecast', methods = ['GET', 'POST'])
def forecast():
    information = request.data
    global stepCount
    stepCount = 12
    if information == '6months':
        stepCount = 6
    elif information == '12months':
        stepCount = 12
    elif information == '2years':
        stepCount = 24
    elif information == '5years':
        stepCount = 60
    else:
        stepCount = 12    

    return "1"

#show the final graph required
@app.route('/plot.png',methods = ['GET', 'POST'])
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

length = 14
breadth = 7
#create a parser to modify the stepCount format
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

#function for the creation of arima graph
def create_figure():
    global length
    global breadth

    fig = Figure()

    #read the csv file
    df = pd.read_csv(fname,parse_dates =['Order Date'])

    technology = df.loc[df['Category'] == 'Technology']
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    technology.drop(cols, axis=1, inplace=True)
    technology = technology.sort_values('Order Date')
    technology = technology.groupby('Order Date')['Sales'].sum().reset_index()
    technology = technology.set_index('Order Date')

    #visualizing the timeseries data
    y = technology['Sales'].resample('MS').mean()
    y.plot(figsize=(length,breadth))
    plt.show()

    #visualizing trend, seasonality, noise
    rcParams['figure.figsize'] = length,breadth
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show() 
    length = 14
    breadth = 7

    #arima
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = mod.fit()
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime('2014-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = y['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Technology Sales')
    dict = {'lower Sales': 'Predicted_Lower_Sales',
        'upper Sales': 'Predicted_Upper_Sales'}

    pred_ci.rename(columns=dict,
            inplace=True)
    salesVal = pred_ci

    df1 = pd.DataFrame(pred_ci) 
 
    df1['Predicted_Value'] = (df1.Predicted_Lower_Sales+df1.Predicted_Upper_Sales)/2
    df1.drop(['Predicted_Lower_Sales'], axis=1)
    df1.drop(['Predicted_Lower_Sales'], axis=1)
    df1['Actual_Value'] = y['2017':]
    df1.to_csv('Prediction.csv')

    plt.legend()
    plt.show()  

    y_forecasted = pred.predicted_mean
    y_truth = y['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
    salesVal = y.tolist()


    pred_uc = results.get_forecast(steps=stepCount)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='r', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Technlogy Sales')

    print(pred_ci)
    plt.legend()
    plt.show()
    return fig

# A simple function to calculate the square of a number
@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):
    return jsonify({'data': num**2})


# driver function
if __name__ == '__main__':
    app.run(debug = True)