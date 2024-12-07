"""
Programmer: Drew Fitzpatrick
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024

Description: This program implements different methods of
            evaluating a dataset as described in the PA6 requirements.
"""
import copy
import random
from tabulate import tabulate
import numpy as np # use numpy's random number generation
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # randomize the table
    np.random.seed(random_state)
    if shuffle:
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        myutils.randomize_in_place(X, y)

    if test_size.is_integer():
        split_index = len(y) - test_size
    else:
        split_index = int((1 - test_size) * len(y)) # 2/3 of randomized table is train, 1/3 is test

    X_train = X[0:split_index]
    X_test = X[split_index:]
    y_train = y[0:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        myutils.randomize_in_place(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop].tolist()
        train_indices = np.concatenate([indices[:start], indices[stop:]]).tolist()
        folds.append((train_indices, test_indices))
        current = stop

    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Initialize the random seed if specified
    if random_state is not None:
        random.seed(random_state)

    # Shuffle the data if specified
    indices = list(range(len(X)))
    if shuffle:
        combined = list(zip(indices, X, y))
        random.shuffle(combined)
        indices, X, y = zip(*combined)

    # Group indices by class to ensure stratification
    label_indices = {}
    for idx, label in enumerate(y):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(indices[idx])

    # Split each label's indices into n_splits folds
    folds = [[] for _ in range(n_splits)]
    for label, idxs in label_indices.items():
        fold_sizes = [len(idxs) // n_splits + (1 if i < len(idxs) % n_splits else 0) for i in range(n_splits)]
        current = 0
        for i, fold_size in enumerate(fold_sizes):
            fold_indices = idxs[current:current + fold_size]
            folds[i].extend(fold_indices)
            current += fold_size

    # Construct the training and testing sets for each fold
    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for fold in folds[:i] + folds[i+1:] for idx in fold]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    np.random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    indices = np.random.randint(0, len(X), size=n_samples)

    X_sample = [X[i] for i in indices]
    X_out_of_bag = [X[i] for i in range(len(X)) if i not in indices]

    if y is not None:
        y_sample = [y[i] for i in indices]
        y_out_of_bag = [y[i] for i in range(len(y)) if i not in indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    matrix = [[0 for _ in labels] for _ in labels]

    # map labels to indices
    label_to_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(y_true, y_pred):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true) | set(y_pred)))
    correct_count = sum(cm[i][i] for i in range(len(cm)))

    if normalize:
        score = correct_count / len(y_true)
    else:
        score = correct_count
    return score

def random_subsample(clf, X, y, k_sub_samples=10, test_size=0.33, discretizer=None):
    '''Repeats train_test_split() k times
    '''
    accuracies = []
    for _ in range(k_sub_samples):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if discretizer is not None:
            for i, _ in enumerate(y_test):
                y_test[i] = discretizer(y_test[i])
            for i, _ in enumerate(y_pred):
                y_pred[i] = discretizer(y_pred[i])

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        avg_accuracy = np.mean(accuracies)
        avg_error_rate = 1 - avg_accuracy

    return avg_accuracy, avg_error_rate

def cross_val_predict(clf, X, y, k_folds=10, random_state=None, discretizer=None, categorical=False):
    """Helper function for predicting accuracy of a classifier using k-fold cross validation
    """
    accuracies = []
    y_preds = []
    y_trues = []
    # Generate the k folds using kfold_split
    folds = kfold_split(X, n_splits=k_folds, random_state=random_state, shuffle=True)
    for train_indices, test_indices in folds:
        X_train, X_test = [X[i] for i in train_indices], [X[i] for i in test_indices]
        y_train, y_test = [y[i] for i in train_indices], [y[i] for i in test_indices]

        clf.fit(X_train, y_train)
        if categorical:
            y_pred = clf.predict(X_test, categorical)
        else:
            y_pred = clf.predict(X_test)

        y_preds.append(y_pred)
        y_trues.append(y_test)

        if discretizer is not None:
            for i, _ in enumerate(y_test):
                y_test[i] = discretizer(y_test[i])
            for i, _ in enumerate(y_pred):
                y_pred[i] = discretizer(y_pred[i])

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    avg_accuracy = np.mean(accuracies)
    avg_error_rate = 1 - avg_accuracy

    return avg_accuracy, avg_error_rate, y_preds, y_trues

def bootstrap_method(clf, X, y, k_sub_samples=10, random_state=None, discretizer=None):
    """Compute predictive accuracy and error rate using the bootstrap method. 
    Args: 
        clf: The classifier to be used for predictions. 
        X (array-like): Feature matrix. 
        y (array-like): Target values. 
        k_sub_samples (int): Number of sub-samples. 
        random_state (int, optional): Random state for reproducibility. 
    Returns: 
        float: Average accuracy over k sub-samples. 
        float: Average error rate over k sub-samples. 
    """
    accuracies = []

    for _ in range(k_sub_samples):
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = bootstrap_sample(X, y, random_state=random_state)
        clf.fit(X_sample, y_sample)
        y_pred = clf.predict(X_out_of_bag)

        if discretizer is not None:
            for i, _ in enumerate(y_out_of_bag):
                y_out_of_bag[i] = discretizer(y_out_of_bag[i])
            for i, _ in enumerate(y_pred):
                y_pred[i] = discretizer(y_pred[i])

        accuracy = accuracy_score(y_out_of_bag, y_pred)
        accuracies.append(accuracy)
        avg_accuracy = np.mean(accuracies)
        avg_error_rate = 1 - avg_accuracy

    return avg_accuracy, avg_error_rate

def display_confusion_matrix(cm, labels):
    """Helper function to display the calculated confusion matrix
    Args:
        clf (str): name of the classifier
        cm (list of list of ints): confusion matrix
        labels (list of strings): labels for the confusion matrix
    """
    #header = ["MPG Ranking"] + labels
    #print(tabulate(cm, headers=header, showindex=labels, tablefmt="simple"))
    #row_totals = [sum(row) for row in cm]
    #for i, _ in enumerate(cm):
    #    cm[i].append(row_totals[i])

    #labels.append("Totals")

    headers = ["Home/Away"] + labels
    row_labels = [[label] + row for label, row in zip(labels[:-1], cm)] # Exclude the "totals" label from row labels
    print(tabulate(row_labels, headers=headers, tablefmt="simple"))

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    matrix = confusion_matrix(y_true, y_pred, labels)
    indexed_labels = {label: index for index, label in enumerate(labels)}
    pos_index = indexed_labels[pos_label]

    tp = matrix[pos_index][pos_index]
    fp = sum(matrix[i][pos_index] for i in range(len(labels)) if i != pos_index)

    if tp + fp == 0:
        return 0.0
    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    matrix = confusion_matrix(y_true, y_pred, labels)
    indexed_labels = {label: index for index, label in enumerate(labels)}
    pos_index = indexed_labels[pos_label]

    tp = matrix[pos_index][pos_index]
    fn = sum(matrix[pos_index][i] for i in range(len(labels)) if i != pos_index)

    if tp + fn == 0:
        return 0.0
    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def gen_precision_recall_f1(y_trues, y_preds, labels=None, pos_label=None):
    """Helper function to generate the precision, recall, and f1 score
    Args:
        y_true(list of list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(foat): precision
        recall(float): recall
        f1(float): F1 score
    """
    precisions = []
    recalls = []

    if labels is None:
        labels = list(set(y_trues[0]))
    if pos_label is None:
        pos_label = labels[0]

    for i, pred in enumerate(y_preds):
        precision = binary_precision_score(y_trues[i], pred, labels, pos_label)
        recall = binary_recall_score(y_trues[i], pred, labels, pos_label)

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    return avg_precision, avg_recall, f1


# Custom implementations for metrics
def binary_precision_score_rf(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    # Pass `labels` as a keyword argument
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    indexed_labels = {label: index for index, label in enumerate(labels)}
    pos_index = indexed_labels[pos_label]

    tp = matrix[pos_index][pos_index]
    fp = sum(matrix[i][pos_index] for i in range(len(labels)) if i != pos_index)

    if tp + fp == 0:
        return 0.0
    precision = tp / (tp + fp)
    return precision

def binary_recall_score_rf(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    # Pass `labels` as a keyword argument
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    indexed_labels = {label: index for index, label in enumerate(labels)}
    pos_index = indexed_labels[pos_label]

    tp = matrix[pos_index][pos_index]
    fn = sum(matrix[pos_index][i] for i in range(len(labels)) if i != pos_index)

    if tp + fn == 0:
        return 0.0
    recall = tp / (tp + fn)
    return recall

def binary_f1_scorerf(y_true, y_pred, labels=None, pos_label=None):
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    precision = binary_precision_score_rf(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score_rf(y_true, y_pred, labels=labels, pos_label=pos_label)

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
