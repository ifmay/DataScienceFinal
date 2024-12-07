"""
Programmer: Drew Fitzpatrick
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024

Description: Utility file for utility functions.
"""

import numpy as np
from tabulate import tabulate

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
        dummy = MyDummyClassifier(strategy='most_frequent')
        dummy.fit(X_train)
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
        dummy = MyDummyClassifier(strategy='most_frequent')
        dummy.fit(X_train)
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
        dummy = MyDummyClassifier(strategy='most_frequent')
        dummy.fit(X_sample)
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
        dummy = MyDummyClassifier(strategy='most_frequent')
        dummy.fit(X_train)
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
    headers = ["Survived"] + labels + ["Total", "Recognition %"]
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