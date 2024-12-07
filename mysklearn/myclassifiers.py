"""
Programmers: Drew Fitzpatrick, Izzy May
Class: CptS 322-01, Fall 2024
Final Project
12/10/2024
"""
import operator
import numpy as np
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test, categorical=False):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            categorical(boolean): True if X_test is categorical

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            row_indexes_dists = []
            for i, row in enumerate(self.X_train):
                if categorical:
                    dist = myutils.compute_categorical_distance(row, test_instance)
                else:
                    dist = myutils.compute_euclidean_distance(row, test_instance)
                row_indexes_dists.append((i, dist))
            # need to sort row_indexes_dists by dist
            row_indexes_dists.sort(key=operator.itemgetter(-1)) # get the item
            # in the tuple at index -1 and use that for sorting
            top_k = row_indexes_dists[:self.n_neighbors]
            distances.append([dist for _ , dist in top_k])
            neighbor_indices.append([i for i, _ in top_k])

        return distances, neighbor_indices

    def predict(self, X_test, categorical=False):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            categorical(boolean): True if the data is categorical

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test, categorical)
        y_predicted = []
        for indices in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in indices]
            most_common_labels = max(set(neighbor_labels), key=neighbor_labels.count)
            y_predicted.append(most_common_labels)

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = myutils.get_frequency(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label] * len(X_test)

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dictionary): The prior probabilities computed for each
            label in the training set.
        posteriors(dictionary of dictionaries): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        class_counts = {}
        for class_label in y_train:
            if class_label not in class_counts:
                class_counts[class_label] = 0
            class_counts[class_label] += 1

        self.priors = {label: count / len(y_train) for label, count in class_counts.items()}

        self.posteriors = {}
        for label in self.priors:
            self.posteriors[label] = {}
            att_counts = {}
            for sample, true_class in zip(X_train, y_train):
                if true_class == label:
                    for i, att_value in enumerate(sample):
                        att_name = f"att_{i}"
                        if att_name not in att_counts:
                            att_counts[att_name] = {}
                        if att_value not in att_counts[att_name]:
                            att_counts[att_name][att_value] = 0
                        att_counts[att_name][att_value] += 1
            for att_name, value_count in att_counts.items():
                for att_value, count in value_count.items():
                    self.posteriors[label][f"{att_name}={att_value}"] = (count) / (class_counts[label])

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            label_probs = {}
            for label, priors in self.priors.items():
                label_probs[label] = priors

                for i, value in enumerate(instance):
                    att_name = f"att_{i}={value}"
                    if att_name in self.posteriors[label]:
                        label_probs[label] *= self.posteriors[label][att_name]
                    else:
                        label_probs[label] *= 1e-6 # apply smoothing for unseen instances

            prediction = max(label_probs, key=label_probs.get)
            y_predicted.append(prediction)

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        header(list of str): The list of attribute names.
        att_domains(dictionary): The attribute names (generalized) as a key
            to the list of possible attribute values.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.att_domains = None

    def partition_instances(self, instances, attribute):
        """Partitions instances in this format: {att val1:part1, att val2:part2}
        """
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.att_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def select_attribute(self, instances, attributes): # possibly need to at class_index to params
        """Returns attribute index to partition on
        """
        # for each available attribute
            # for each value in the attribute's domain
                # calculate the entropy for the value's partition
            # calculate the weighted avg for the partition entropies
        # select the attribute with the smallest Enew entropy
        best_attribute = None
        min_entropy = float('inf')

        for att_index in attributes:
            partitions = self.partition_instances(instances, att_index)
            weighted_entropy = myutils.calc_weighted_entropy(partitions)

            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = att_index

        return best_attribute

    def tdidt(self, current_instances, available_attributes, prev_instances_len=None):
        """Implements TDIDT approach for building a decision tree
        """
        #print("available attributes: ", available_attributes)
        
        if prev_instances_len is None:
            prev_instances_len = len(current_instances)

        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        #print("splitting on: ", split_attribute)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        #print("partitions: ", partitions)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and myutils.check_all_same_class(att_partition):
                #print("CASE 1") # make a leaf node
                class_label = myutils.get_class_label(att_partition)
                leaf_node = ["Leaf", class_label, len(att_partition), len(current_instances)]
                value_subtree.append(leaf_node)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                #print("CASE 2")
                majority_class = myutils.get_majority_class(att_partition)
                leaf_node = ["Leaf", majority_class, len(att_partition), len(current_instances)]
                value_subtree.append(leaf_node)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            # doesn't mean recurse, means change tree = ['Attribute', attribute_val] to majority vote leaf node
            elif len(att_partition) == 0:
                #print("CASE 3")
                majority_class = myutils.get_majority_class(current_instances)
                leaf_node =["Leaf", majority_class, len(current_instances), prev_instances_len]
                tree = leaf_node
                break
            else:
                # none of the base cases were true, recurse!
                subtree = self.tdidt(att_partition, available_attributes.copy(), len(current_instances))
                # append subtree to value_subtree and append value_subtree to tree appropriately
                value_subtree.append(subtree)

            tree.append(value_subtree)

        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # extract header and att_domains
        self.header = [f"att{i}" for i in range(len(X_train[0]))]

        self.att_domains = {}
        for i in range(len(X_train[0])):
            att_vals = {instance[i] for instance in X_train}
            self.att_domains[self.header[i]] = sorted(att_vals)

        # lets stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        #print(train)

        # make a copy of header, b/c python is pass by object reference
        # and tdidt will be removing attributes from available_attributes
        available_attributes = self.header.copy()
        tree = self.tdidt(train, available_attributes)
        print("tree: ", tree)
        self.tree = tree

    def tdidt_predict(self, tree, instance):
        """Helper function for predict(), uses recursion"""
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1] # class label

        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            pred = self.tdidt_predict(self.tree, instance)
            y_predicted.append(pred)

        return y_predicted

    def help_print_rules(self, node, rule, att_names, class_name):
        """Helper function for print_decision_rules(), uses recursion"""
        if node[0] == "Leaf":
            print(f"{rule} THEN {class_name} = {node[1]}")
        else:
            if att_names is None:
                att_name = f"{node[1]}"
            else:
                att_name = att_names[node[1]]

            for value_subtree in node[2:]:
                new_rule = f"{rule} AND {att_name} == {value_subtree[1]}" if rule else f"IF {att_name} == {value_subtree[1]}"
                self.help_print_rules(value_subtree[2], new_rule, att_names, class_name)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # use recursion
        self.help_print_rules(self.tree, "", attribute_names, class_name)


class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        header(list of str): The list of attribute names.
        att_domains(dictionary): The attribute names (generalized) as a key
            to the list of possible attribute values.
        n(int): The number of decision trees to generate.
        m(int): The number of most accurate decision trees.
        f(int): The number of attributes to randomly select

    """
    def __init__(self, n=10, m=5, f=None):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.trees = []
        self.header = None
        self.att_domains = None
        self.n = n
        self.m = m
        self.f = f

    def partition_instances(self, instances, attribute):
        """Partitions instances in this format: {att val1:part1, att val2:part2}
        """
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.att_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def select_attribute(self, instances, attributes): # possibly need to at class_index to params
        """Returns attribute index to partition on
        """
        # for each available attribute
            # for each value in the attribute's domain
                # calculate the entropy for the value's partition
            # calculate the weighted avg for the partition entropies
        # select the attribute with the smallest Enew entropy
        best_attribute = None
        min_entropy = float('inf')

        for att_index in attributes:
            partitions = self.partition_instances(instances, att_index)
            weighted_entropy = myutils.calc_weighted_entropy(partitions)

            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = att_index

        return best_attribute
    
    def tdidt(self, current_instances, available_attributes, prev_instances_len=None):
        """Implements TDIDT approach for building a decision tree
        """
        #print("available attributes: ", available_attributes)
        
        if prev_instances_len is None:
            prev_instances_len = len(current_instances)

        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        #print("splitting on: ", split_attribute)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        #print("partitions: ", partitions)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and myutils.check_all_same_class(att_partition):
                #print("CASE 1") # make a leaf node
                class_label = myutils.get_class_label(att_partition)
                leaf_node = ["Leaf", class_label, len(att_partition), len(current_instances)]
                value_subtree.append(leaf_node)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                #print("CASE 2")
                majority_class = myutils.get_majority_class(att_partition)
                leaf_node = ["Leaf", majority_class, len(att_partition), len(current_instances)]
                value_subtree.append(leaf_node)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            # doesn't mean recurse, means change tree = ['Attribute', attribute_val] to majority vote leaf node
            elif len(att_partition) == 0:
                #print("CASE 3")
                majority_class = myutils.get_majority_class(current_instances)
                leaf_node =["Leaf", majority_class, len(current_instances), prev_instances_len]
                tree = leaf_node
                break
            else:
                # none of the base cases were true, recurse!
                subtree = self.tdidt(att_partition, available_attributes.copy(), len(current_instances))
                # append subtree to value_subtree and append value_subtree to tree appropriately
                value_subtree.append(subtree)

            tree.append(value_subtree)

        return tree
    
    def stratified_sample(self, X_train, y_train):
        """Stratified sampling to split dataset into test and remainder sets"""
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        test_indices = []
        remainder_indices = []
        for cls, count in zip(unique_classes, class_counts):
            cls_indices = [i for i, y in enumerate(y_train) if y == cls]
            np.random.shuffle(cls_indices)
            split_point = count // 3
            test_indices.extend(cls_indices[:split_point])
            remainder_indices.extend(cls_indices[split_point:])
        
        X_remainder = [X_train[i] for i in remainder_indices]
        y_remainder = [y_train[i] for i in remainder_indices]
        X_test = [X_train[i] for i in test_indices]
        y_test = [y_train[i] for i in test_indices]

        return X_remainder, y_remainder, X_test, y_test
    
    def generate_decision_trees(self, X_remainder, y_remainder):
        """Generates N decision trees using bootstrapping"""
        tree_validation_pairs = []
        for _ in range(self.n):
            bootstrap_indices = np.random.choice(len(X_remainder), len(X_remainder), replace=True)
            X_bootstrap = [X_remainder[i] for i in bootstrap_indices]
            y_bootstrap = [y_remainder[i] for i in bootstrap_indices]

            validation_indices = [i for i in range(len(X_remainder)) if i not in bootstrap_indices]
            X_validation = [X_remainder[i] for i in validation_indices]
            y_validation = [y_remainder[i] for i in validation_indices]

            if not validation_indices:
                continue # skip if validation set is empty

            available_atts = self.header.copy()
            if self.f:
                available_atts = self.compute_random_subset(available_atts, self.f)
            
            tree = self.tdidt(X_bootstrap + [y_bootstrap], available_atts)
            tree_validation_pairs.append((tree, X_validation, y_validation))

        return tree_validation_pairs


    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # extract header and att_domains
        self.header = [f"att{i}" for i in range(len(X_train[0]))]

        self.att_domains = {}
        for i in range(len(X_train[0])):
            att_vals = {instance[i] for instance in X_train}
            self.att_domains[self.header[i]] = sorted(att_vals)

        X_remainder, y_remainder, X_test, y_test = self.stratified_sample(X_train, y_train)

        tree_validation_pairs = self.generate_decision_trees(X_remainder, y_remainder)

        tree_accuracy_pairs = []
        for tree, X_val, y_val in tree_validation_pairs:
            y_pred = self.predict_tree(tree, X_val)
            accuracy = np.mean([pred == true for pred, true in zip(y_pred, y_val)])
            tree_accuracy_pairs.append((tree, accuracy))

        tree_accuracy_pairs.sort(key=lambda x: x[1], reverse=True)
        self.trees = [pair[0] for pair in tree_accuracy_pairs[:self.m]]

        print(f"num trees: {len(self.trees)}")
        print(self.trees)

    def predict_tree(self, tree, X):
        """Makes predictions for test instances in X using a single decision tree. """
        y_predicted = []
        for instance in X:
            pred = self.tdidt_predict(tree, instance)
            y_predicted.append(pred)

        return y_predicted

    def compute_random_subset(self, values, num_values):
        """Selects F (num_values) random attributes from an attribute list
        
        Args:
            values(list of obj): The list of attribute values
            num_values(int): """
        values_copy = values.copy()
        np.random.shuffle(values_copy) # inplace
        return values_copy[:num_values]
    
    def tdidt_predict(self, tree, instance):
        """Helper function for predict(), uses recursion"""
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1] # class label

        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance)

    def get_majority_vote(self, votes):
        vote_count = {}
        for vote in votes:
            if vote in vote_count:
                vote_count[vote] += 1
            else:
                vote_count[vote] = 1
        
        majority_vote = max(vote_count, key=vote_count.get)
        return majority_vote

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            votes = [self.tdidt_predict(tree, instance) for tree in self.trees]
            majority_vote = self.get_majority_vote(votes)
            y_predicted.append(majority_vote)

        return y_predicted
