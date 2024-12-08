"""
Programmer: Drew Fitzpatrick
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024

Description: Utility file for utility functions.
"""

import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier
import mysklearn.myevaluation as myevaluation

def discretize_delays(delay_time):
    """
    Categorizes flight delays into specified time intervals and prints statistics.
    
    Parameters:
        delay_time (float or int): The delay time.
    """
    if delay_time <= 0:
        return "On Time"
    if 0 < delay_time <= 60:
        return "0-1 hours"
    if 60 < delay_time <= 120:
        return "1-2 hours"
    if 120 < delay_time <= 180:
        return "2-3 hours"
    if delay_time > 180:
        return "Over 3 hours"


def randomize_in_place(alist, parallel_list=None):
    '''Shuffles a list in place
    '''
    for i, _ in enumerate(alist):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def compute_euclidean_distance(v1, v2):
    '''Computes the euclidean distance between two points, v1 and v2
    '''
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def compute_categorical_distance(v1, v2):
    '''Computes the distance between two lists of categorical values'''
    dist = 0
    for i, _ in enumerate(v1):
        if v1[i] == v2[i]:
            dist += 0
        else:
            dist += 1

    return dist

def get_frequency(y_train):
    '''Finds the most common label in y_train
    Args:
        y_train (list of obj): list of class labels, parallel to X_train
        
    Returns:
        (str): the most common label
    '''
    label_counts = {}
    for label in y_train:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return max(label_counts, key=label_counts.get)

def calc_entropy(instances):
    """Calculates the entropy of the given instances
    """
    class_counts = {}
    for instance in instances:
        class_label = instance[-1]
        if class_label not in class_counts:
            class_counts[class_label] = 1
        else:
            class_counts[class_label] += 1

    entropy = 0.0
    for count in class_counts.values():
        probability = count / len(instances)
        entropy -= probability * np.log2(probability)
    return entropy

def calc_weighted_entropy(partitions):
    """Calculates the weighted entropy for the partitions
    """
    total_instances = sum(len(partition) for partition in partitions.values())
    weighted_entropy = 0.0
    for partition in partitions.values():
        entropy = calc_entropy(partition)
        weighted_entropy += (len(partition) / total_instances) * entropy
    return weighted_entropy

def check_all_same_class(instances): # need class_index?
    """True if all instances have the same label
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True

def get_class_label(instances):
    """ Given a list of instances (each instance is a list of attribute values including the class label), 
    returns the class label. Assumes all instances have the same class label. """
    # Assuming the class label is the last element in each instance
    class_label = instances[0][-1]
    return class_label

def get_majority_class(instances):
    """Gets the majority class vote
    """
    class_counts = {}
    for instance in instances:
        class_label = instance[-1]

        if class_label not in class_counts:
            class_counts[class_label] = 1
        else:
            class_counts[class_label] += 1
    max_count = max(class_counts.values())
    max_classes = [class_label for class_label, count in class_counts.items() if count == max_count]
    return max_classes[0]

def mpg_discretizer(mpg_value):
    """
    Discretize the MPG (Miles Per Gallon) value into a class based on the 
    Department of Energy (DOE) rating scale.

    The discretization is done by categorizing the MPG value into specific 
    ranges, assigning a class label from 1 to 10.

    Parameters:
    mpg_value (float): The continuous MPG value to be discretized.

    Returns:
    int: The corresponding class label based on the DOE rating scale.
    """
    if mpg_value >= 45:
        return 10
    elif mpg_value >= 37:
        return 9
    elif mpg_value >= 31:
        return 8
    elif mpg_value >= 27:
        return 7
    elif mpg_value >= 24:
        return 6
    elif mpg_value >= 20:
        return 5
    elif mpg_value >= 17:
        return 4
    elif mpg_value >= 15:
        return 3
    elif mpg_value == 14:
        return 2
    else:
        return 1

def calculate_accuracy(predictions, actuals):
    """
    Calculate the accuracy of the classifier by comparing predicted classes
    with actual classes.

    The function counts the number of correct predictions where the predicted 
    class matches the actual class and computes the accuracy as the ratio of 
    correct predictions to the total number of predictions.

    Parameters:
    predictions (list of int): List of predicted class labels.
    actuals (list of int): List of actual class labels.

    Returns:
    float: The accuracy of the predictions, a value between 0 and 1.
    """
    correct_predictions = sum(np.round(pred) == actual for pred, actual in zip(predictions, actuals))
    return correct_predictions / len(actuals)

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