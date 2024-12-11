import pickle
import pandas as pd
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MyRandomForestClassifier, MyNaiveBayesClassifier


import numpy as np

# Load the dataset (assuming `data` is a pandas DataFrame containing the provided sample data)
data = pd.read_csv('balanced_flights.csv') 

# Define delay categories
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
data['delay_category'] = data['dep_delay'].apply(categorize_delay)

# Prepare features (X) and labels (y)
X = data[['dep_time', 'sched_dep_time', 'sched_arr_time', 'air_time', 'hour']].fillna(0)

y = data['delay_category'].values

X = np.array(X)
y = np.array(y)

kf = myevaluation.stratified_kfold_split(X, y, n_splits=10, random_state=None, shuffle=False)

# Define the correct labels for the confusion matrix and metrics
labels = [0, 1, 2, 3, 4, 5]
nb_model = MyNaiveBayesClassifier()

# Loop over each fold in Stratified K-Fold
for train_index, test_index in kf:
    nb_X_train, nb_X_test = X[train_index], X[test_index]
    nb_y_train, nb_y_test = y[train_index], y[test_index]

    # fit MyNaiveBayesClassifier
    nb_model.fit(nb_X_train.tolist(), nb_y_train.tolist())

with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)


"""
# Perform manual stratified K-fold
n_splits = 10
indices = np.arange(len(y))
unique_classes, y_counts = np.unique(y, return_counts=True)
folds = {i: [] for i in range(n_splits)}

for cls in unique_classes:
    cls_indices = indices[y == cls]
    np.random.shuffle(cls_indices)
    for i, index in enumerate(cls_indices):
        folds[i % n_splits].append(index)

fold_indices = [np.array(folds[i]) for i in range(n_splits)]

rf_model = MyRandomForestClassifier(n_estimators=10, max_depth=5)


# Iterate through folds
for i in range(n_splits):
    test_indices = fold_indices[i]
    train_indices = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])

    rf_X_train, rf_X_test = X[train_indices], X[test_indices]
    rf_y_train, rf_y_test = y[train_indices], y[test_indices]

    # Train Random Forest
    rf_model.fit(rf_X_train, rf_y_train)

print()
rf_model.print_trees()

with open("trained_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)
"""