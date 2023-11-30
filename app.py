from flask import Flask, render_template, request
from guageHeight import predict_flood
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.form['date']
    result = predict_flood(date)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True,port='5000')
