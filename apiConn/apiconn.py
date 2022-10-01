# Using flask to make an api
# import necessary libraries and functions
#from contextlib import _RedirectStream

from asyncio.windows_events import NULL
from datetime import datetime
import os
from pyexpat import model
from flask import Flask, jsonify, request, Response, redirect
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib
from pylab import rcParams
from math import sqrt
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# creating a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:/SHIVA/angular/apiConn'
@app.route('/upload',methods = ['GET', 'POST'])
def upload_File():

    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            
            return redirect(request.url)
        global file
        file = request.files['file']
        global fname
        fname = secure_filename(file.filename)    
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
        return redirect(request.url)
    return 'File Uploaded'

path = 'D:/Desktop/upload'
#full_path = os.path.join(path,fname)
# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/plot.png',methods = ['GET', 'POST'])
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

length = 14
breadth = 14
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

def create_figure():
    global length
    global breadth
    fig = Figure()
    #data = "hello world"
    df = pd.read_csv(fname,parse_dates =['Order Date'])

    furniture = df.loc[df['Category'] == 'Furniture']
    cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
    furniture.drop(cols, axis=1, inplace=True)
    furniture = furniture.sort_values('Order Date')
  

    furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

    furniture = furniture.set_index('Order Date')

    y = furniture['Sales'].resample('MS').mean()
    y.plot(figsize=(length,breadth))
    plt.show()
    
    #arima_model = ARIMA(df.value, order=(1,1,2))
    #model  =arima_model.fit()
    #model.plot_predict(dynamic = False)
    #plt.show()

    rcParams['figure.figsize'] = length,breadth
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    plt.show() 
    length = 10
    breadth = 10
    return fig

@app.route('/', methods = ['GET', 'POST'])
def home():
 if(request.method == 'GET'):
  data = "hello world"
  
  return data
  #return data


# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000/home/10
# this returns 100 (square of 10)
@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):

 return jsonify({'data': num**2})


# driver function
if __name__ == '__main__':

 app.run(debug = True)