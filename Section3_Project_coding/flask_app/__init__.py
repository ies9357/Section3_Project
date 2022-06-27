from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pickle

def create_app():
    app = Flask(__name__)
    model = pickle.load(open('project_best_estimator.pkl', 'rb'))

    @app.route('/')
    def hello_world():
            return render_template('index.html')

    @app.route('/predict', methods = ['POST'])
    def predict():
        highbp = int(request.form['HighBP'])
        highchol = int(request.form['HighChol'])
        genhlth = int(request.form['GenHlth'])
        prediction = model.predict_proba([[highbp, highchol, genhlth]])[:, 1] > 0.36

        if prediction[0] == False:
            output = '정상 상태입니다.'
        else:
            output = '당뇨병 전단계 또는 당뇨병입니다.'

        return render_template('index.html', prediction_text = output)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run()
