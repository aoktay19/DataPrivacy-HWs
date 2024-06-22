import sys
import random

import numpy as np
import pandas as pd
import copy


from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack
    average_acc = 0
    N = 100
    if model_type == "DT":
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == "LR":
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    elif model_type == "SVC":
        model = SVC(C=0.5, kernel='poly', random_state=0, probability=True)

    for iteration in range(N):
        flipped_y_train = label_flipping_attack(y_train, p)
        model.fit(X_train, flipped_y_train)
        predicted_model = model.predict(X_test)
        average_acc += accuracy_score(y_test, predicted_model)

    average_acc /= N

    return average_acc

def label_flipping_attack(y_train, p):
    """
    Performs label flipping attack on training labels.

    Parameters:
    y_train: Training labels
    p: Proportion of labels to flip

    Returns:
    Flipped training labels
    """
    num = int(len(y_train) * p)
    range_of_list = range(len(y_train))
    flipped_index = random.sample(range_of_list, num)
    flipped_y_train = copy.deepcopy(y_train)
    flipped_y_train[flipped_index] = 1 - flipped_y_train[flipped_index]

    return flipped_y_train


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome

    flipped_y_train = label_flipping_attack(y_train, p)
    isolation_forest = IsolationForest(n_estimators=50, contamination = 0.5)
    noise = isolation_forest.fit_predict(X_train) == -1
    flipped_index = np.where(flipped_y_train != y_train)[0]
    noise_of_flipped = sum(noise[flipped_index])

    print(f"Out of {sum(flipped_y_train != y_train)} flipped data points, {noise_of_flipped} were correctly identified.")



###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    # while pred_class == actual_class:
    # do something to modify the instance
    #    print("do something")

    increment_value = 0.5
    predicted_class = actual_class

    while increment_value < 100:
        for index, feature in enumerate(modified_example):
            modified_example = copy.deepcopy(actual_example)
            modified_example[index] += increment_value
            predicted_class = trained_model.predict([modified_example])[0]

            if predicted_class != actual_class:
                return modified_example

        for index, feature in enumerate(modified_example):
            modified_example = copy.deepcopy(actual_example)
            modified_example[index] -= increment_value
            predicted_class = trained_model.predict([modified_example])[0]

            if predicted_class != actual_class:
                return modified_example

        increment_value += 0.5

    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation
    DT_to_LR = DT_to_SVC = LR_to_DT = LR_to_SVC = SVC_to_DT = SVC_to_LR = 0

    for instance in actual_examples:
        model_example = evade_model(SVCmodel, instance)
        if SVCmodel.predict([model_example]) == DTmodel.predict([model_example]):
            SVC_to_DT += 1
        if SVCmodel.predict([model_example]) == LRmodel.predict([model_example]):
            SVC_to_LR += 1
        model_example = evade_model(DTmodel, instance)
        if DTmodel.predict([model_example]) == LRmodel.predict([model_example]):
            DT_to_LR += 1
        if DTmodel.predict([model_example]) == SVCmodel.predict([model_example]):
            DT_to_SVC += 1
        model_example = evade_model(LRmodel, instance)
        if LRmodel.predict([model_example]) == DTmodel.predict([model_example]):
            LR_to_DT += 1
        if LRmodel.predict([model_example]) == SVCmodel.predict([model_example]):
            LR_to_SVC += 1



    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {DT_to_LR} of them transfer to LR.")
    print(f"-> {DT_to_SVC} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {LR_to_DT} of them transfer to DT.")
    print(f"-> {LR_to_SVC} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {SVC_to_DT} of them transfer to DT.")
    print(f"-> {SVC_to_LR} of them transfer to LR.")



###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)

    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])



if __name__ == "__main__":
    main()


