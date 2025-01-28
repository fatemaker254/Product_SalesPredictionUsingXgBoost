# import csv
# import re
# import pandas as pd

# def clean_and_add_pno(input_file, output_file):
#     # List of exception PNos that should be considered valid
#     exception_pnos = ["TIGIN", "TICAR", "TICHA"]
    
#     # Read the CSV file using pandas for easier data manipulation
#     df = pd.read_csv(input_file)

#     # Remove rows where any value in 'Qty', 'Sales Price', or 'Amount' is NaN or contains 'credit'
#     df = df.dropna(subset=['Qty', 'Sales Price', 'Amount'])  # Drop rows with NaN in these specific columns
#     df = df[~df['Memo'].str.contains('credit', case=False, na=False)]  # Remove rows containing 'credit' in 'Memo'

#     # Define a function to extract the PNo
#     def get_pno(item_name):
#         # Ensure that item_name is a string before proceeding
#         if isinstance(item_name, str):
#             # First, check for any known exception PNo
#             pno = "N/A"
#             for exception in exception_pnos:
#                 if exception in item_name:
#                     pno = exception
#                     break

#             # If no exception is found, use the regex to search for PNo (3-5 alphabets followed by digits)
#             if pno == "N/A":
#                 pno_match = re.search(r'[A-Za-z]{3,5}\d+[A-Za-z]*', item_name)

#                 # Extract the matched PNo or assign 'N/A' if no match found
#                 pno = pno_match.group(0).strip() if pno_match else "N/A"
#         else:
#             pno = "N/A"

#         return pno

#     # Add the 'PNo' column by applying the get_pno function to the 'Item' column
#     df['PNo'] = df['Item'].apply(get_pno)

#     # Write the cleaned and updated data to a new CSV file
#     df.to_csv(output_file, index=False)

# # Specify the input and output file names
# input_file = "sm.csv"  # Change this to your actual input file path
# output_file = "SabziMandiData.csv"  # Change this to your desired output file path

# # Call the function to clean the data and add the PNo column
# clean_and_add_pno(input_file, output_file)

import re
import pandas as pd

def clean_and_add_pno(input_file, output_file):
    # List of exception PNos that should be considered valid
    exception_pnos = ["TIGIN", "TICAR", "TICHA"]
    
    # Read the CSV file using pandas for easier data manipulation
    df = pd.read_csv(input_file)
    
    # Remove rows where any value in 'Qty', 'Sales Price', or 'Amount' is NaN or contains 'credit'
    df = df.dropna(subset=['Qty', 'Sales Price', 'Amount'])  # Drop rows with NaN in these specific columns
    df = df[~df['Memo'].str.contains('credit', case=False, na=False)]  # Remove rows containing 'credit' in 'Memo'
    
    # Ensure the 'Date' column is in datetime format for sorting
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert 'Date' column to datetime, invalid parsing will be NaT
        df = df.dropna(subset=['Date'])  # Drop rows where 'Date' conversion failed

    # Define a function to extract the PNo
    def get_pno(item_name):
        # Ensure that item_name is a string before proceeding
        if isinstance(item_name, str):
            # First, check for any known exception PNo
            pno = "N/A"
            for exception in exception_pnos:
                if exception in item_name:
                    pno = exception
                    break

            # If no exception is found, use the regex to search for PNo (3-5 alphabets followed by digits)
            if pno == "N/A":
                pno_match = re.search(r'[A-Za-z]{3,5}\d+[A-Za-z]*', item_name)

                # Extract the matched PNo or assign 'N/A' if no match found
                pno = pno_match.group(0).strip() if pno_match else "N/A"
        else:
            pno = "N/A"

        return pno

    # Add the 'PNo' column by applying the get_pno function to the 'Item' column
    df['PNo'] = df['Item'].apply(get_pno)

    # Sort the data by 'Date' from oldest to newest
    df = df.sort_values(by='Date')

    # Write the cleaned and updated data to a new CSV file
    df.to_csv(output_file, index=False)

# Specify the input and output file names
input_file = "PateBrothersData.csv"  # Change this to your actual input file path
output_file = "SabziMandiData.csv"  # Change this to your desired output file path

# Call the function to clean the data, add the PNo column, and sort by date
clean_and_add_pno(input_file, output_file)
