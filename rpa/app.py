from flask import Flask, render_template, request
import LR
import PR  # Import your machine learning scripts

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get the height input from the HTML form
    height = float(request.form['height'])

    # Use the appropriate model from your script to make predictions
    linear_regression_result = LR.reg.predict([[height]])
    polynomial_regression_result = PR.polynomial_regression.predict([[height]])

    # Returning results to the HTML page
    return render_template(
        'index.html',
        height=height,
        linear_result=linear_regression_result[0],
        polynomial_result=polynomial_regression_result[0]
    )

if __name__ == '__main__':
    app.run(debug=True)
