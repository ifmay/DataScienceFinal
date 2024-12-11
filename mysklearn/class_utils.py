
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier
import mysklearn.myevaluation as myevaluation

def random_subsample(X, y, k=10, test_size=0.33, random_state=None):
    """
    Perform random subsampling to calculate predictive accuracy and error rate for each classifier.

    The function splits the dataset k times into training and test sets, fits a k-Nearest Neighbors 
    classifier and a Dummy classifier on each split, and calculates their accuracy and error rates.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    k (int, optional): Number of subsamples to generate. Defaults to 10.
    test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.33.
    random_state (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    tuple: Accuracy and error rates for both classifiers.
    """
    knn_accuracy_sum = 0
    dummy_accuracy_sum = 0

    for _ in range(k):
        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=random_state)

        # K nearest neighbors classifier
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracy_sum += myevaluation.accuracy_score(y_test, knn_pred)

        # Dummy Classifier
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_accuracy_sum += myevaluation.accuracy_score(y_test, dummy_pred)

    # Average error rate over k splits
    knn_error_rate = knn_accuracy_sum / k
    dummy_error_rate = dummy_accuracy_sum / k

    # Calculate accuracy rates
    knn_accuracy = 1 - knn_error_rate
    dummy_accuracy = 1 - dummy_error_rate

    # Print results
    print("===========================================")
    print("Random Subsample (k={k}, 2:1 Train/Test)")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_accuracy:.2f}, error rate = {knn_error_rate:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_accuracy:.2f}, error rate = {dummy_error_rate:.2f}")
    return knn_accuracy, knn_error_rate, dummy_accuracy, dummy_error_rate

def cross_val_predict(X, y, k=10, stratified=False, random_state=None):
    """
    Compute predictive accuracy and error rate using k-fold cross-validation.

    This function uses k-fold cross-validation to evaluate the k-Nearest Neighbors 
    and Dummy classifiers, calculating their average accuracy and error rates over all folds.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    k (int, optional): Number of folds. Defaults to 10.
    stratified (bool, optional): Whether to use stratified sampling. Defaults to False.
    random_state (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    tuple: Accuracy and error rates for both classifiers.
    """
    knn_accuracy_sum = 0
    dummy_accuracy_sum = 0

    # Select k-fold split method
    if stratified:
        folds = myevaluation.stratified_kfold_split(X, y, n_splits=k, random_state=random_state, shuffle=True)
    else:
        folds = myevaluation.kfold_split(X, n_splits=k, random_state=random_state, shuffle=True)

    # Loop through each fold
    for train_indices, test_indices in folds:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # K Nearest neighbors classifier
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracy_sum += myevaluation.accuracy_score(y_test, knn_pred)

        # Dummy classifier
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_accuracy_sum += myevaluation.accuracy_score(y_test, dummy_pred)

     # Average error rate over k splits
    knn_error_rate = knn_accuracy_sum / k
    dummy_error_rate = dummy_accuracy_sum / k

    # Calculate accuracy rates
    knn_accuracy = 1 - knn_error_rate
    dummy_accuracy = 1 - dummy_error_rate

    # Print results
    print("\n===========================================")
    print("Stratified 10-Fold Cross Validation")
    print("===========================================")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_accuracy:.2f}, error rate = {knn_error_rate:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_accuracy:.2f}, error rate = {dummy_error_rate:.2f}")
    return knn_accuracy, knn_error_rate, dummy_accuracy, dummy_error_rate



def bootstrap_method(X, y, k=10, random_state=None):
    """
    Compute predictive accuracy and error rate for each classifier using the bootstrap method.

    The function performs k bootstrap samples, where each sample is used to train the classifiers 
    and the out-of-bag samples are used to evaluate accuracy and error rates.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    k (int, optional): Number of bootstrap samples. Defaults to 10.
    random_state (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    tuple: Accuracy and error rates for both classifiers.
    """
    knn_accuracy_sum = 0
    dummy_accuracy_sum = 0

    for _ in range(k):
        # Generate bootstrap sample
        X_sample, X_test, y_sample, y_test = myevaluation.bootstrap_sample(X, y, random_state=random_state)

        # K nearest neighbors classifier
        knn = MyKNeighborsClassifier()
        knn.fit(X_sample, y_sample)
        knn_pred = knn.predict(X_test)
        knn_accuracy_sum += myevaluation.accuracy_score(y_test, knn_pred)

        # Dummy classifier
        dummy = MyDummyClassifier()
        dummy.fit(X_sample, y_sample)
        dummy_pred = dummy.predict(X_test)
        dummy_accuracy_sum += myevaluation.accuracy_score(y_test, dummy_pred)


    # Average error rate over k bootstrap samples
    knn_error_rate = knn_accuracy_sum / k if k > 0 else 0
    dummy_error_rate = dummy_accuracy_sum / k if k > 0 else 0

    # Calculate accuracy rates
    knn_accuracy = 1 - knn_error_rate
    dummy_accuracy = 1 - dummy_error_rate

    # Print results in the specified format
    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")
    print(f"k={k} Bootstrap Method")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_accuracy:.2f}, error rate = {knn_error_rate:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_accuracy:.2f}, error rate = {dummy_error_rate:.2f}")

    return knn_accuracy, knn_error_rate, dummy_accuracy, dummy_error_rate

def random_subsample_with_predictions(X, y, k=10, test_size=0.33, random_state=None):
    """
    Perform random subsampling to calculate predictive accuracy, error rate, and predictions for each classifier.

    This function splits the dataset k times, fits the classifiers on each split, and stores 
    predictions for future use along with accuracy and error rates.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target labels.
    k (int, optional): Number of subsamples to generate. Defaults to 10.
    test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.33.
    random_state (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    tuple: knn_predictions, knn_accuracy, knn_error_rate, dummy_accuracy, dummy_error_rate
    """

    knn_accuracy_sum = 0
    dummy_accuracy_sum = 0
    knn_predictions = []

    for _ in range(k):
        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=random_state)

        # k Nearest Neighbors Classifier
        knn = MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_predictions.extend(knn_pred)
        knn_accuracy_sum += myevaluation.accuracy_score(y_test, knn_pred)

        # Dummy Classifier
        dummy = MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_accuracy_sum += myevaluation.accuracy_score(y_test, dummy_pred)

    # Average accuracy and error rate over k splits
    knn_error_rate = knn_accuracy_sum / k
    dummy_error_rate = dummy_accuracy_sum / k

    # Calculate accuracy rates
    knn_accuracy = 1 - knn_error_rate
    dummy_accuracy = 1 - dummy_error_rate

    return knn_predictions, knn_accuracy, knn_error_rate, dummy_accuracy, dummy_error_rate


def calculate_confusion_matrix_totals(matrix):
    """Calculate totals and recognition percentages for each row in the confusion matrix.

    Args:
        matrix (list of list of int): The confusion matrix as a list of lists.

    Returns:
        list of list of int/float: The confusion matrix with added total and recognition % columns.
    """
    new_matrix = []
    for i, row in enumerate(matrix):
        total = sum(row)
        recognition = (row[i] / total * 100) if total > 0 else 0  # row[i] is the diagonal element (True Positives)
        new_matrix.append(row + [total, recognition])
    return new_matrix

def display_confusion_matrix(matrix, labels):
    """Display the formatted confusion matrix with MPG rating, 1-10, Total, and recognition %.

    Args:
        matrix (list of list of int): The confusion matrix with totals and recognition %.
        labels (list of str): The list of MPG ratings as column headers.

    Returns:
        None
    """
    headers = ["Delayed"] + labels + ["Total", "Recognition %"]
    table = [[label] + row for label, row in zip(labels, matrix)]
    print(tabulate(table, headers, floatfmt=".2f"))
    

def select_top_elements(trees, accuracies, m_classifiers):
    """Selects the indices of the top `m_classifiers` trees based on their accuracies.

    Args:
        trees (list): A list of decision trees.
        accuracies (list): A list of accuracies corresponding to each tree.
        m_classifiers (int): The number of top classifiers (trees) to select.

    Returns:
        list: A list of indices of the top `m_classifiers` trees based on their accuracies.
    """
    # Ensure the accuracies and trees have the same length
    assert len(trees) == len(accuracies), "The number of trees must match the number of accuracies."
    
    # Sort the indices based on accuracies in descending order, then select the top `m_classifiers`
    top_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)[:m_classifiers]
    
    return top_indices

from collections import Counter

def get_most_frequent(values):
    """Returns the most frequent value in a list.

    Args:
        values (list): A list of values (e.g., class labels or any other items).

    Returns:
        The most frequent value in the list. If there is a tie, the smallest value is returned.
    """
    # Count the occurrences of each value in the list
    counts = Counter(values)
    
    # Find the maximum count
    max_count = max(counts.values())
    
    # Get the list of values that have the maximum count (in case of ties)
    most_frequent = [key for key, count in counts.items() if count == max_count]
    
    # Return the smallest value in case of a tie
    return min(most_frequent)

def analyze_flight_data(file_path):
    """
    This function loads flight data, processes it, and computes the correlation matrix 
    for various flight-related features.
    
    Parameters:
        file_path (str): Path to the CSV file containing the flight data.
        
    Returns:
        pd.DataFrame: The correlation matrix of the features.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Compute flight delay (in minutes, assuming sched_dep_time and dep_time are timestamps or integers)
    data['flight_delay'] = data['sched_dep_time'] - data['dep_time']

    # Define delay categories for classification
    def categorize_delay(delay):
        if delay <= 0:
            return 0
        elif 0 < delay <= 30:
            return 1
        elif 30 < delay <= 60:
            return 2
        elif 60 < delay <= 120:
            return 3
        elif 120 < delay <= 180:
            return 4
        else:
            return 5
    
    # Map delays to categories
    data['delay_category'] = data['arr_delay'].apply(categorize_delay)

    # Convert categorical features to integers (label encoding)
    categorical_cols = ['carrier', 'flight', 'tailnum', 'origin', 'dest']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le  # Store the encoder for later use if needed

    # Prepare features (X) and labels (y)
    X = data[['dep_time', 'sched_dep_time', 'dep_delay', 'arr_time', 'sched_arr_time', 'arr_delay',
              'air_time', 'distance', 'month', 'hour']].fillna(0)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate the correlation matrix
    correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()

    return correlation_matrix

def visualize_flight_delays(file_path):
    """
    Visualizes flight delays by categorizing them into specified time intervals.
    
    Parameters:
        file_path (str): The path to the flights.csv file.
    """
    # Load the CSV file into a Pandas DataFrame
    flights = pd.read_csv(file_path)
    
    # Ensure the 'dep_delay' column is numeric
    flights['dep_delay'] = pd.to_numeric(flights['dep_delay'], errors='coerce')
    
    # Remove rows with missing 'dep_delay' values
    flights = flights.dropna(subset=['dep_delay'])
    
    # Categorize delays
    delay_intervals = {
        "On Time": flights['dep_delay'] <= 0,
        "0-30 mins": (flights['dep_delay'] > 0) & (flights['dep_delay'] <= 30),
        "30 mins - 1 hour": (flights['dep_delay'] > 30) & (flights['dep_delay'] <= 60),
        "1-2 hours": (flights['dep_delay'] > 60) & (flights['dep_delay'] <= 120),
        "2-3 hours": (flights['dep_delay'] > 120) & (flights['dep_delay'] <= 180),
        "3-4 hours": (flights['dep_delay'] > 180) & (flights['dep_delay'] <= 240),
        "Over 4 hours": flights['dep_delay'] > 240
    }
    
    # Count occurrences in each category
    counts = {category: flights[condition].shape[0] for category, condition in delay_intervals.items()}
    
    # Create a bar chart
    categories = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title("Flight Delay Categories", fontsize=16)
    plt.xlabel("Delay Category", fontsize=14)
    plt.ylabel("Number of Flights", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_delays_by_airline(file_path):
    """
    Visualizes average flight delays by airline, considering only delayed flights.
    
    Parameters:
        file_path (str): The path to the flights.csv file.
    """
    # Load the CSV file into a Pandas DataFrame
    flights = pd.read_csv(file_path)
    
    # Ensure 'dep_delay' is numeric and 'carrier' exists
    flights['dep_delay'] = pd.to_numeric(flights['dep_delay'], errors='coerce')
    flights = flights.dropna(subset=['dep_delay', 'carrier'])
    
    # Filter only delayed flights
    delayed_flights = flights[flights['dep_delay'] > 0]
    
    # Group by 'carrier' and calculate the average delay
    delay_by_airline = delayed_flights.groupby('carrier')['dep_delay'].mean().sort_values()
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    delay_by_airline.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Flight Delay by Airline (Delayed Flights Only)", fontsize=16)
    plt.xlabel("Airline", fontsize=14)
    plt.ylabel("Average Delay (minutes)", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def on_time_vs_delayed_by_airline(file_path):
    """
    Visualizes the number of on-time vs. delayed flights for each airline.
    
    Parameters:
        file_path (str): The path to the flights.csv file.
    """
    # Load the CSV file into a Pandas DataFrame
    flights = pd.read_csv(file_path)
    
    # Ensure 'dep_delay' is numeric and 'carrier' exists
    flights['dep_delay'] = pd.to_numeric(flights['dep_delay'], errors='coerce')
    flights = flights.dropna(subset=['dep_delay', 'carrier'])
    
    # Categorize flights as "on-time" or "delayed"
    flights['status'] = flights['dep_delay'].apply(lambda x: 'Delayed' if x > 0 else 'On-time')
    
    # Group by airline and status
    status_by_airline = flights.groupby(['carrier', 'status']).size().unstack(fill_value=0)
    
    # Plot the data
    status_by_airline.plot(kind='bar', figsize=(12, 6), edgecolor='black', width=0.8)
    plt.title("On-Time vs. Delayed Flights by Airline", fontsize=16)
    plt.xlabel("Airline", fontsize=14)
    plt.ylabel("Number of Flights", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="Status", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_flight_delays_balanced(file_path):
    """
    Visualizes flight delays by categorizing them into specified time intervals.
    
    Parameters:
        file_path (str): The path to the flights.csv file.
    """
    # Load the CSV file into a Pandas DataFrame
    flights = pd.read_csv(file_path)
    
    # Ensure the 'dep_delay' column is numeric
    flights['dep_delay'] = pd.to_numeric(flights['dep_delay'], errors='coerce')
    
    # Remove rows with missing 'dep_delay' values
    flights = flights.dropna(subset=['dep_delay'])
    
    # Categorize delays
    delay_intervals = {
        "On Time": flights['dep_delay'] <= 0,
        "0-30 mins": (flights['dep_delay'] > 0) & (flights['dep_delay'] <= 30),
        "30 mins - 1 hour": (flights['dep_delay'] > 30) & (flights['dep_delay'] <= 60),
        "1-2 hours": (flights['dep_delay'] > 60) & (flights['dep_delay'] <= 120),
        "2-3 hours": (flights['dep_delay'] > 120) & (flights['dep_delay'] <= 180),
        "Over 3 hours": flights['dep_delay'] > 180
    }
    
    # Count occurrences in each category
    counts = {category: flights[condition].shape[0] for category, condition in delay_intervals.items()}
    
    # Create a bar chart
    categories = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title("Flight Delay Categories", fontsize=16)
    plt.xlabel("Delay Category", fontsize=14)
    plt.ylabel("Number of Flights", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_delays_by_time_of_day(file_path):
    """
    Visualizes flight delays by time of day, considering only delayed flights.
    
    Parameters:
        file_path (str): The path to the flights.csv file.
    """
    # Load the CSV file into a Pandas DataFrame
    flights = pd.read_csv(file_path)
    
    # Ensure 'dep_delay' is numeric and 'sched_dep_time' exists
    flights['dep_delay'] = pd.to_numeric(flights['dep_delay'], errors='coerce')
    flights = flights.dropna(subset=['dep_delay', 'sched_dep_time'])
    
    # Filter only delayed flights
    delayed_flights = flights[flights['dep_delay'] > 0]
    
    # Extract the hour from 'sched_dep_time'
    delayed_flights['sched_hour'] = (delayed_flights['sched_dep_time'] // 100).astype(int)
    
    # Group by 'sched_hour' and calculate the average delay
    delay_by_hour = delayed_flights.groupby('sched_hour')['dep_delay'].mean()
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(delay_by_hour.index, delay_by_hour.values, marker='o', linestyle='-', color='orange', label='Average Delay')
    plt.title("Average Flight Delay by Time of Day (Delayed Flights Only)", fontsize=16)
    plt.xlabel("Scheduled Departure Hour (24-hour format)", fontsize=14)
    plt.ylabel("Average Delay (minutes)", fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()