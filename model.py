import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def max_sold_products(data):
    # Ensure 'Date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Handle rows where 'Date' could not be converted
    if data['Date'].isnull().any():
        print("Some rows have invalid dates and will be excluded.")
        data = data.dropna(subset=['Date'])

    # Calculate the date 6 months ago
    six_months_ago = data['Date'].max() - pd.DateOffset(months=6)
    
    # Filter data for the last 6 months
    last_six_months_data = data[data['Date'] >= six_months_ago]
    
    # Group by product (PNo) and calculate total quantities sold
    top_products = (
        last_six_months_data.groupby('PNo')['Qty']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    
    # Merge with original data to get product names (Memo)
    top_products = top_products.merge(data[['PNo', 'Memo']].drop_duplicates(), on='PNo', how='left')
    
    # Print the top 10 products
    print("\nTop 10 products sold in the last 6 months:")
    for _, row in top_products.iterrows():
        print(f"Product: {row['Memo']}, PNo: {row['PNo']}, Quantity Sold: {row['Qty']}")
    return top_products

def process_and_predict_sales(input_csv, start_date, end_date, output_csv, days):
    # Load the data
    data = pd.read_csv(input_csv, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Ensure 'Date' is datetime
    
    last_date_in_data = data['Date'].max()
    print(f"Last date in dataset: {last_date_in_data}")
    
    # Filter data for recent activity
    recent_sales_cutoff = last_date_in_data - pd.Timedelta(weeks=4)
    active_products = data[data['Date'] >= recent_sales_cutoff]['PNo'].unique()
    filtered_data = data[data['PNo'].isin(active_products)]
    print(f"Filtered data contains {len(filtered_data)} rows after filtering for active products.")
    
    if filtered_data.empty:
        print("No active product data for predictions. Exiting...")
        return

    # Feature engineering
    filtered_data['Days_Since_Last_Sale'] = (
        last_date_in_data - filtered_data.groupby('PNo')['Date'].transform('max')
    ).dt.days

    filtered_data['Sales_Last_4W'] = filtered_data.groupby('PNo')['Qty'].transform(
        lambda x: x.rolling(window=4, min_periods=1).sum()
    )

    filtered_data['Avg_Weekly_Sales'] = filtered_data.groupby('PNo')['Qty'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )

    filtered_data['Week'] = filtered_data['Date'].dt.isocalendar().week
    filtered_data['Month'] = filtered_data['Date'].dt.month
    
    # Encode 'PNo'
    le = LabelEncoder()
    filtered_data['PNo_encoded'] = le.fit_transform(filtered_data['PNo'])

    # Add binary classification target
    filtered_data['WillSellNextWeek'] = (
        filtered_data.groupby('PNo')['Date'].shift(-1) <= last_date_in_data + pd.Timedelta(days=7)
    ).astype(int)

    # Classification model to predict if a product will sell
    X_class = filtered_data[['PNo_encoded', 'Week', 'Month', 'Days_Since_Last_Sale', 'Avg_Weekly_Sales']]
    y_class = filtered_data['WillSellNextWeek']

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    classifier = XGBClassifier()
    classifier.fit(X_train_class, y_train_class)

# Filter and rank significant products
    filtered_data['Predicted_Sell_Prob'] = classifier.predict_proba(X_class)[:, 1]
    filtered_data['Predicted_Sell'] = (filtered_data['Predicted_Sell_Prob'] > 0.7).astype(int)

# Add recency score (update as needed for your business logic)
    filtered_data['Recency_Score'] = filtered_data['Days_Since_Last_Sale'] / 7  # Weeks since last sale

# Apply filtering conditions
    products_to_predict = filtered_data[
        (filtered_data['Predicted_Sell'] == 1) &# High probability of selling
        (filtered_data['Avg_Weekly_Sales'] > 1) # &  # Significant weekly sales
        #(filtered_data['Recency_Score'] > 3)  # Recency score threshold
    ].sort_values(by='Predicted_Sell_Prob', ascending=False)  # Select top 10 products

    # if products_to_predict.empty:
    #     print("No products predicted to sell based on the conditions. Exiting...")
    #     return

    # Limit to top N products (optional)
    # top_n = 10
    # products_to_predict = products_to_predict.nlargest(top_n, 'Predicted_Sell_Prob')
    
    # Regression model to predict quantities for likely-to-sell products
    X_reg = products_to_predict[['PNo_encoded', 'Week', 'Month', 'Days_Since_Last_Sale', 'Avg_Weekly_Sales']]
    y_reg = products_to_predict['Qty']

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    regressor = XGBRegressor()
    regressor.fit(X_train_reg, y_train_reg)

    # Prepare predictions
    predictions = []
    for date in pd.date_range(start_date, end_date, freq='W-MON'):  # Assuming weekly sales prediction
        for pno in products_to_predict['PNo'].unique():
            pno_encoded = le.transform([pno])[0]
            week_num = date.isocalendar().week
            weight_features = products_to_predict[products_to_predict['PNo'] == pno][[
                'Days_Since_Last_Sale', 'Avg_Weekly_Sales'
            ]].iloc[0]

            features = [pno_encoded, week_num, date.month] + list(weight_features)
            predicted_qty = regressor.predict([features])[0]

            # Cap predictions at 150% of historical max
            historical_max = filtered_data[filtered_data['PNo'] == pno]['Qty'].max()
            capped_prediction = min(predicted_qty, historical_max * 1.5)
            rounded_prediction = np.ceil(capped_prediction)

            product_name = filtered_data[filtered_data['PNo'] == pno]['Memo'].iloc[0]
            predictions.append([date, pno, product_name, rounded_prediction])

    # Save predictions
    output_df = pd.DataFrame(predictions, columns=['Date', 'PNo', 'Product Name', 'Predicted Quantity'])
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Save as JSON
    output_json = output_df.to_json(orient='records', date_format='iso')
    json_output_path = output_csv.replace('.csv', '.json')
    with open(json_output_path, 'w') as json_file:
        json_file.write(output_json)
    print(f"Predictions saved to {json_output_path}")


process_and_predict_sales("output\Patel_Brothers_MD_Gaithersburg.csv", "2024-10-28", "2024-11-03", "predictionstrial.csv", 7)
max_sold_products(pd.read_csv("output\Patel_Brothers_MD_Gaithersburg.csv"))