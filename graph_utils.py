import pandas as pd

def filter_airlines(input_file, output_file):
    """
    Filters data to include only the major 8 airlines and destinations with more than 2,500 flights,
    and saves the result to a new CSV file.
    
    Parameters:
        input_file (str): The path to the input CSV file containing flight data.
        output_file (str): The path to save the filtered data as a CSV file.
    """
    # Load the data
    flights = pd.read_csv(input_file)

    # Define the list of major airline codes
    airline_codes = ['AA', 'B6', 'DL', 'EV', 'UA', 'US', 'WN']

    # Filter the data by the major airlines
    major_airlines_flights = flights[flights['carrier'].isin(airline_codes)]

    # Count the number of flights to each destination
    destination_counts = major_airlines_flights['dest'].value_counts()

    # Filter destinations with more than 2,500 flights
    valid_destinations = destination_counts[destination_counts > 2500].index
    
    # Filter the original dataset for these destinations
    filtered_flights = major_airlines_flights[major_airlines_flights['dest'].isin(valid_destinations)]

    # Save the filtered data to a new CSV file
    filtered_flights.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}. Number of rows: {len(filtered_flights)}")
