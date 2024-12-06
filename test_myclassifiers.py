from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyNaiveBayesClassifier

# pylint: skip-file
import numpy as np
from mysklearn import myutils
import operator
from scipy import stats

from mysklearn.myclassifiers import MyKNeighborsClassifier,\
    MyDummyClassifier

# note: order is actual/received student value, expected/solution

def test_kneighbors_classifier_kneighbors():
    # A
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ['bad', 'bad', 'good', 'good']
    knn1 = MyKNeighborsClassifier()
    knn1.fit(X_train, y_train)
    X_test = [[0.33, 1]]
    actual_distances, actual_kn_indices = knn1.kneighbors(X_test)
    expected_distances = [[0.6699999999, 1.0, 1.05304320899]]
    expected_kn_indices = [[0, 2, 3]] 

    assert np.allclose(actual_distances, expected_distances)
    assert np.allclose(actual_kn_indices, expected_kn_indices)

    # B
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    X_test = [[2, 3]]
    knn2 = MyKNeighborsClassifier()
    knn2.fit(X_train, y_train)
    actual_distances, actual_kn_indices = knn2.kneighbors(X_test)

    expected_distances = [[1.4142135623730951, 1.4142135623730951, 2.0]]
    expected_kn_indices = [[0, 4, 6]]

    assert np.allclose(actual_distances, expected_distances)
    assert np.allclose(actual_kn_indices, expected_kn_indices)

    #C
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    knn3 = MyKNeighborsClassifier(n_neighbors=5)
    knn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[9.1, 11.0]]
    actual_distances, actual_kn_indices = knn3.kneighbors(X_test)
    
    #expected_distances = [[2.802, 1.237, 0.608, 2.202, 2.915]]
    #expected_kn_indices = [[4, 5, 6, 7, 8]]

    expected_distances = [[0.608276, 1.236931, 2.2022715, 2.80177851, 2.91547594]]
    expected_kn_indices = [[6, 5, 7, 4, 8]]

    assert np.allclose(actual_distances, expected_distances)
    assert np.allclose(actual_kn_indices, expected_kn_indices)

def test_kneighbors_classifier_predict():
    # A
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ['bad', 'bad', 'good', 'good']
    knn1 = MyKNeighborsClassifier()
    knn1.fit(X_train, y_train)
    X_test = [[0.33, 1]]

    y_actual = knn1.predict(X_test)
    y_expected = ['good']

    assert y_actual == y_expected

    # B
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    X_test = [[2, 3]]
    knn = MyKNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_actual = knn.predict(X_test)
    y_expected = ["yes"] 

    assert y_actual == y_expected

    # C
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    knn3 = MyKNeighborsClassifier()
    knn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[9.1, 11.0]]
    
    y_actual = knn3.predict(X_test)
    y_expected = ["+"]

    assert y_actual == y_expected

def test_dummy_classifier_fit():
    # A
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_classifier1 = MyDummyClassifier()
    dummy_classifier1.fit([[]], y_train)

    assert dummy_classifier1.most_common_label == "yes"

    #B 
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_classifier2 = MyDummyClassifier()
    dummy_classifier2.fit([[]], y_train)

    assert dummy_classifier2.most_common_label == "no"

    #C
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.2, 0.6]))
    dummy_classifier3 = MyDummyClassifier()
    dummy_classifier3.fit([[]], y_train)

    assert dummy_classifier3.most_common_label == "maybe"

def test_dummy_classifier_predict():
    # A
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_classifier1 = MyDummyClassifier()
    dummy_classifier1.fit([[]], y_train)
    y_actual = dummy_classifier1.predict([[1]])

    assert y_actual == ["yes"]

    #B 
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_classifier2 = MyDummyClassifier()
    dummy_classifier2.fit([[]], y_train)
    y_actual = dummy_classifier2.predict([[1]])

    assert y_actual == ["no"]

    #C
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.2, 0.6]))
    dummy_classifier3 = MyDummyClassifier()
    dummy_classifier3.fit([[]], y_train)
    y_actual = dummy_classifier3.predict([[1]])

    assert y_actual == ["maybe"]

def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for key, value in dict1.items():
        if key not in dict2:
            return False
        if isinstance(value, float) and isinstance(dict2[key], float):
            if not compare_dicts(value, dict2[key]):
                return False
        elif value != dict2[key]:
            return False

    return True

def sort_and_convert_dict(d):
    """Sort and convert dictionary values to floats for consistent comparison."""
    sorted_dict = {}
    for key, value in d.items():
        sorted_value = {k: float(v) for k, v in sorted(value.items())}
        sorted_dict[key] = sorted_value
    return sorted_dict
def test_naive_bayes_classifier_fit():
    # A
    X_train_A = [[1, 5],
                 [2, 6],
                 [1, 5],
                 [1, 5],
                 [1, 6],
                 [2, 6],
                 [1, 5],
                 [1, 6]]
    y_train_A =['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes']

    nb_class_A = MyNaiveBayesClassifier()
    nb_class_A.fit(X_train_A, y_train_A)
    actual_priors_A = nb_class_A.priors
    actual_posteriors_A = nb_class_A.posteriors

    expected_priors_A = {'no': 3/8, 'yes': 5/8}
    expected_posteriors_A = {
        'yes': {
            'att_0=1': 4/5, 
            'att_0=2': 1/5,
            'att_1=5': 2/5,
            'att_1=6': 3/5
        },
        'no': {
            'att_0=1': 2/3, 
            'att_0=2': 1/3,
            'att_1=5': 2/3,
            'att_1=6': 1/3
        }
    }

    assert actual_priors_A == expected_priors_A
    assert actual_posteriors_A == expected_posteriors_A

    #B
    X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    nb_class_iphone = MyNaiveBayesClassifier()
    nb_class_iphone.fit(X_train_iphone, y_train_iphone)
    actual_priors_iphone = nb_class_iphone.priors
    actual_posteriors_iphone = nb_class_iphone.posteriors

    expected_priors_iphone = {'no': 5/15, 'yes': 10/15}
    expected_posteriors_iphone = {
        'yes': {
            'att_0=1': 2/10, 
            'att_0=2': 8/10,
            'att_1=1': 3/10,
            'att_1=2': 4/10,
            'att_1=3': 3/10,
            'att_2=fair': 7/10,
            'att_2=excellent': 3/10
        },
        'no': {
            'att_0=1': 3/5, 
            'att_0=2': 2/5,
            'att_1=1': 1/5,
            'att_1=2': 2/5,
            'att_1=3': 2/5,
            'att_2=fair': 2/5,
            'att_2=excellent': 3/5
        }
    }
    assert actual_priors_iphone == expected_priors_iphone
    assert actual_posteriors_iphone == expected_posteriors_iphone

    #C
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    
    nb_class_C = MyNaiveBayesClassifier()
    nb_class_C.fit(X_train_train, y_train_train)
    actual_priors_C = nb_class_C.priors
    actual_posteriors_C = nb_class_C.posteriors

    expected_priors_C = {'on time': 14/20, 'late': 2/20, 'very late': 3/20, 'cancelled': 1/20}
    expected_posteriors_C = {
        'on time': {
            'att_0=weekday': 9/14,
            'att_0=saturday': 2/14,
            'att_0=sunday': 1/14,
            'att_0=holiday': 2/14,
            'att_1=spring': 4/14,
            'att_1=summer': 6/14,
            'att_1=autumn': 2/14,
            'att_1=winter': 2/14,
            'att_2=none': 5/14,
            'att_2=high': 4/14,
            'att_2=normal': 5/14,
            'att_3=none': 5/14,
            'att_3=slight': 8/14,
            'att_3=heavy': 1/14
        },
        'late': {
            'att_0=weekday': 1/2,
            'att_0=saturday': 1/2,
            'att_1=winter': 1.0,
            'att_2=high': 1/2,
            'att_2=normal': 1/2,
            'att_3=none': 1/2,
            'att_3=heavy': 1/2
        },
        'very late': {
            'att_0=weekday': 1.0,
            'att_1=autumn': 1/3,
            'att_1=winter': 2/3,
            'att_2=high': 1/3,
            'att_2=normal': 2/3,
            'att_3=none': 1/3,
            'att_3=heavy': 2/3
        },
        'cancelled': {
            'att_0=saturday': 1.0,
            'att_1=spring': 1.0,
            'att_2=high': 1.0,
            'att_3=heavy': 1.0
        }
    }

    assert actual_priors_C == expected_priors_C
    assert actual_posteriors_C == expected_posteriors_C
    assert sort_and_convert_dict(actual_posteriors_C) == sort_and_convert_dict(expected_posteriors_C)

def test_naive_bayes_classifier_predict():
    # A
    X_train_A = [[1, 5],
                 [2, 6],
                 [1, 5],
                 [1, 5],
                 [1, 6],
                 [2, 6],
                 [1, 5],
                 [1, 6]]
    y_train_A =['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes']
    X_test_A = [[1,5]]

    nb_class_A = MyNaiveBayesClassifier()
    nb_class_A.fit(X_train_A, y_train_A)
    y_pred_actual = nb_class_A.predict(X_test_A)
    y_pred_expected = ['yes']

    assert y_pred_actual == y_pred_expected

    #B
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    X_test_iphone = [[2, 2, "fair"]]

    nb_class_iphone = MyNaiveBayesClassifier()
    nb_class_iphone.fit(X_train_iphone, y_train_iphone)
    y_pred_expected_iphone = nb_class_iphone.predict(X_test_iphone)
    y_pred_actual_iphone = ["yes"]

    assert y_pred_actual_iphone == y_pred_expected_iphone

    #C
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    
    X_test_C = [["weekday", "winter", "high", "heavy"],
                ["weekday", "summer", "high", "heavy"],
                ["sunday", "summer", "normal", "slight"]]

    nb_class_C = MyNaiveBayesClassifier()
    nb_class_C.fit(X_train_train, y_train_train)
    y_pred_actual_C = nb_class_C.predict(X_test_C)
    y_pred_expected_C = ["very late", "on time", "on time"]

    assert y_pred_actual_C == y_pred_expected_C

def test_decision_tree_classifier_fit():
    # A
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)

    assert decision_tree.tree == tree_interview

    #B
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone = \
        ["Attribute", "att0",
            ["Value", 1,
                ["Attribute", "att1",
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Leaf", "no", 4, 10]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]
    decision_tree_iphone = MyDecisionTreeClassifier()
    decision_tree_iphone.fit(X_train_iphone, y_train_iphone)

    assert decision_tree_iphone.tree == tree_iphone

def test_decision_tree_classifier_predict():
    # A
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_expected = ["True", "False"]

    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview)
    y_actual = decision_tree.predict(X_test)

    assert y_actual == y_expected

    #B
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    X_test_iphone = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_expected_iphone = ["yes", "yes"]

    decision_tree_iphone = MyDecisionTreeClassifier()
    decision_tree_iphone.fit(X_train_iphone, y_train_iphone)
    y_actual_iphone = decision_tree_iphone.predict(X_test_iphone)

    assert y_actual_iphone == y_expected_iphone
