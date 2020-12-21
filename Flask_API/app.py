from flask import Flask, render_template, request, url_for
import numpy as np
import joblib

model = joblib.load("D:\Projects\irisflowers\irismodel.model")

app = Flask(__name__)

"""@app.route('/prediction', methods = ['POST'])
def predict(sl, sw, pl, pw):
    x = np.array([sl, sw, pl, pw]).reshape(1,-1)
    p = model.predict(x)
    print(p)"""

@app.route('/')
@app.route('/predict', methods = ["POST"])
def home():
    
    if request.method == "POST":
        sl = request.form.get('sepal_length', type = float)
        sw = request.form.get('sepal_width', type = float)
        pl = request.form.get('petal_length', type = float)
        pw = request.form.get('petal_width', type = float)

        print(sl,sw,pl,pw)

    #predict(sl, sw, pl, pw)
    else:
        return render_template('home.html')
"""@app.route('/prediction', methods = ['POST'])
def predict(sl, sw, pl, pw):
    x = np.array([sl, sw, pl, pw]).reshape(1,-1)
    p = model.predict(x)
    print(p)"""

if __name__ == "__main__":
    app.run(port = 5050, debug=True)