"""Utility file for graphs
"""
import pandas as pd

def filter_airlines(input_file, output_file):
    """
    Filters flights to only include specific airline codes and saves the result to a new CSV file.
    
    Parameters:
        input_file (str): The path to the input flights.csv file.
        output_file (str): The path to save the filtered data as a CSV file.
    """
    # Load the data
    flights = pd.read_csv(input_file)

    # Define the list of desired airline codes
    airline_codes = ['AA', 'B6', 'DL', 'EV', 'UA', 'US', 'WN']

    # Filter the data
    filtered_flights = flights[flights['carrier'].isin(airline_codes)]

    # Save the filtered data to a new CSV file
    filtered_flights.to_csv(output_file, index=False)

    print(f"Filtered data saved to {output_file}. Number of rows: {len(filtered_flights)}")
