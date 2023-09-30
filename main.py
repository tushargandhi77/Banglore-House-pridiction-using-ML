from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('pipe.pkl','rb'))
df = pd.read_csv('Cleaned_dat.csv')

@app.route('/')
def index():
    area_type  = sorted(df['area_type'].unique())
    location = sorted(df['location'].unique())
    size = sorted(df['size'].unique())
    bath = sorted(df['bath'].unique())
    balcony = sorted(df['balcony'].unique())
    return render_template('index.html', area_type=area_type, location=location, size=size, bath=bath,balcony=balcony)

@app.route('/predict',methods=['POST'])
def predict():
    area_type = request.form.get('area_type')
    location = request.form.get('location')
    size = request.form.get('size')
    total_sqft = float(request.form.get('total_sqft'))
    bath = float(request.form.get('bath'))
    balcony = float(request.form.get('balcony'))
    prediction = model.predict(pd.DataFrame([[area_type,location,size,total_sqft,bath,balcony]],columns=['area_type','location','size','total_sqft','bath','balcony']))
    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)