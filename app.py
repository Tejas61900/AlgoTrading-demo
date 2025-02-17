from flask import Flask, render_template, request, jsonify
import os
from backtesting import UpstoxBacktest  # Assuming your backtesting.py is named 'backtesting.py'
import pandas as pd

app = Flask(__name__)

# Load access token from environment variable
access_token = os.getenv("ACCESS_TOKEN")

# Ensure you have access token stored in your environment variables
if not access_token:
    raise ValueError("Access token is missing. Set it in environment variables.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    # Get inputs from the form
    expiry_date = request.form.get('expiry_date', '2025-02-20')
    from_date = request.form.get('from_date', '2025-02-13')
    to_date = request.form.get('to_date', '2025-02-16')
    angle_threshold = float(request.form.get('angle_threshold', '15.0'))
    
    # Initialize backtest
    backtest = UpstoxBacktest(access_token)
    
    # Run backtest
    results = backtest.run_backtest(from_date, to_date, expiry_date, angle_threshold)
    
    # Return results as a DataFrame in HTML table
    if not results.empty:
        results_html = results.to_html(classes='table table-striped')
    else:
        results_html = "No trades executed."
    
    return render_template('index.html', results=results_html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)