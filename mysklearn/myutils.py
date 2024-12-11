"""
Programmer: Drew Fitzpatrick, Izzy May
Class: CPSC 322, Fall 2024
Final Project
11/20/2024

Description: Utility file for utility functions.
"""

import numpy as np


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
