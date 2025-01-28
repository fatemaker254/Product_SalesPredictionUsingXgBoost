from flask import Flask, render_template, request
from modelUpdated import process_and_predict_sales  # Import the function from model.py
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Landing page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from form submission
        selected_option = request.form.get('option')  # Retrieve selected option
        start_date = request.form.get('start_date')  # Retrieve start date
        end_date = request.form.get('end_date')      # Retrieve end date
        print(selected_option)

        # Determine the input CSV based on user-selected option
        if selected_option == 'Patel-Brothers':
            input_csv = 'PateBrothersData.csv'
            days=10
        elif selected_option == 'Sabzi-Mandi':
            input_csv = 'SabziMandiData.csv'
            days=21
        else:
            input_csv = 'PateBrothersData.csv' 
           # days=7 # Default dataset or any other option
            print("Invalid option selected. Using default dataset.")

        output_csv = 'predictions.csv'  # Define output CSV file

        # Call the prediction function from model.py
        process_and_predict_sales(input_csv, start_date, end_date, output_csv,days)

        
        # if prediction_data is not None:
        #     # If predictions are generated, render them in the template
        #     table_data = prediction_data.to_dict(orient='records')
        #     return render_template('predict.html', table_data=table_data, selected_option=selected_option)

        try:
            prediction_data = pd.read_csv(output_csv)  # Load predictions from the CSV file
    
            # Convert data to a dictionary format for rendering in the template
            table_data = prediction_data.to_dict(orient='records')
    
            # Render the data in the template
            return render_template('predict.html', table_data=table_data, selected_option=selected_option, start_date=start_date, end_date=end_date)
        except FileNotFoundError:
            error_message = "Predictions could not be generated or the output file is missing."
            return render_template('predict.html', error_message=error_message, selected_option=selected_option,start_date=start_date, end_date=end_date)
    return render_template('predict.html', error_message="Please submit the form to view predictions.")

if __name__ == '__main__':
    app.run(debug=True)
