from flask import Flask, render_template, request, url_for
import numpy as np
import joblib

model = joblib.load("/home/amanshihab/Iris-Flower-Classification/irismodel.model")

app = Flask(__name__, template_folder = 'templates')

def list2string(s):
    str1 = ""

    for ele in s:
        str1 += ele
    
    return str1
@app.route('/prediction', methods = ['POST'])
def predict():

    sl = request.form['sepal_length']
    sw = request.form['sepal_width']
    pl = request.form['petal_length']
    pw = request.form['petal_width']

    x = np.array([sl, sw, pl, pw]).reshape(1,-1)
    p = str(model.predict(x))
    p = list2string(p)
    #p = p.replace("\"", "")
    p = p.strip('"')

    return render_template('home.html', p = p)

@app.route('/', methods = ["POST","GET"])
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(port = 5050, debug=True)