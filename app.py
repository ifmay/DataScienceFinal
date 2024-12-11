from collections import Counter
import pickle
from flask import Flask, request, jsonify
import numpy as np
app = Flask(__name__)

def load_model():
    with open("naive_bayes_model.pkl", "rb") as f:
        model = pickle.load(f)
    #header = []
    print(model)
    return model

def tdidt_predict(trees, X):
    """
    Predicts the class labels for the given input data using majority voting across all trees.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

    Returns:
        list: Predicted class labels for each sample in X.
    """
    # Collect predictions from all trees
    tree_preds = np.array([tree.predict(X) for tree in trees])
    
    # Perform majority voting for each sample
    return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

def nb_predict(model, X_test):
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
        for label, priors in model.priors.items():
            label_probs[label] = priors

            for i, value in enumerate(instance):
                att_name = f"att_{i}={value}"
                if att_name in model.posteriors[label]:
                    label_probs[label] *= model.posteriors[label][att_name]
                else:
                    label_probs[label] *= 1e-6 # apply smoothing for unseen instances

        prediction = max(label_probs, key=label_probs.get)
        y_predicted.append(prediction)

    return y_predicted

'''
@app.route("/")
def index():
    return """
    <h1>Welcome to the delay predictor app!</h1>
    <h2><a href="http://127.0.0.1:5001/predict?dep_time=1826&sched_dep_time=1830&sched_arr_time=2105&arr_delay=-12.0&air_time=175.0&hour=18">
    Click here to predict delay</a></h2>
    """
'''

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dep_time = request.form.get("dep_time")
        sched_dep_time = request.form.get("sched_dep_time")
        sched_arr_time = request.form.get("sched_arr_time")
        air_time = request.form.get("air_time")
        hour = request.form.get("hour")
       
        # Generate the prediction link with user input
        predict_link = f"http://127.0.0.1:5001/predict?dep_time={dep_time}&sched_dep_time={sched_dep_time}&sched_arr_time={sched_arr_time}&air_time={air_time}&hour={hour}"

        return f"""
        <h1>Welcome to the delay predictor app!</h1>
        <h2>Prediction Link:</h2>
        <a href="{predict_link}">Click here to predict delay</a>
        """
   
    return """
    <h1>Welcome to the delay predictor app!</h1>
    <form method="POST">
        <label for="dep_time">Departure Time:</label>
        <input type="text" id="dep_time" name="dep_time" required><br><br>

        <label for="sched_dep_time">Scheduled Departure Time:</label>
        <input type="text" id="sched_dep_time" name="sched_dep_time" required><br><br>

        <label for="sched_arr_time">Scheduled Arrival Time:</label>
        <input type="text" id="sched_arr_time" name="sched_arr_time" required><br><br>

        <label for="air_time">Air Time:</label>
        <input type="text" id="air_time" name="air_time" required><br><br>

        <label for="hour">Hour:</label>
        <input type="text" id="hour" name="hour" required><br><br>

        <input type="submit" value="Generate Prediction Link">
    </form>
    """

@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    dep_time = request.args.get("dep_time") # defaults to None
    sched_dep_time = request.args.get("sched_dep_time")
    sched_arr_time = request.args.get("sched_arr_time")
    air_time = request.args.get("air_time")
    hour = request.args.get("hour")
    instance = [dep_time, sched_dep_time, sched_arr_time,
                air_time, hour]
    nb_model = load_model()
    # lets make a prediction!
    pred = nb_predict(nb_model, [instance])
    #pred = nb_model.predict(instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

# no delay
# http://127.0.0.1:5001/predict?dep_time=1826&sched_dep_time=1830&sched_arr_time=2105&arr_delay=-12.0&air_time=175.0&hour=18
