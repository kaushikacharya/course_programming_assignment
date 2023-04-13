# Gradient Boosting with XGBoost
1. [Introduction](#introduction)
2. [XGBoost Basics](#xgboost-basics)
3. [Cross-Validation](#cross-validation)
4. [Storing Boosters](#storing-boosters)
5. [XGBoost Classifier](#xgboost-classifier)
6. [XGBoost Regressor](#xgboost-regressor)
7. [Feature Importance](#feature-importance)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Model Persistence](#model-persistence)
10. Quiz

## Introduction
- [XGBoost Python Package](https://xgboost.readthedocs.io/en/latest/python/index.html)
    - A library for highly efficient boosted decision trees.

- ### A. XGBoost vs. scikit-learn
    - XGBoost makes use of [gradient boosted decision trees](https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting)

- ### B. Gradient boosted trees
    - Drawback of regular decision trees:
        - Often not complex enough to capture the intricacies of many large datasets.
        - How about continuously increase the maximum depth of a decision tree to fit larger datasets?
            - But decision trees with many nodes tend to overfit the data.
    - Grzadient boosting starts off with a single decision tree and iteratively adds more decision trees to the overall model to correct the model's errors on the training dataset.

## XGBoost Basics
- ### Chapter Goals
    - Learn about the XGBoost data matrix.
    - Train a ```Booster``` object in XGBoost.

- ### A. Basic data structures
    - [DMatrix](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix)
        - Basic data structure for XGBoost.
        - Represents a data matrix.
        - Can be used for training and using a [Booster](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster).
    - [train](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train):
        - The function to train a ```Booster``` with a specified set of parameters.
    - [Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)

- ### B. Using a Booster

- ### Time to Code!

## Cross-Validation
- ### Chapter Goals
    - Learn how to cross-validate parameters in XGBoost.

- ### A. Choosing parameters
    - [cv](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.cv)
        - Function to perform cross-validation for a set of parameters on a given training dataset.

## Storing Boosters
- ### Chapter Goals
    - Learn how to save and load ```Booster``` models in XGBoost.

- ### A. Saving and loading binary data
    - [save_model](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.save_model) function
        - Saves the model's binary data into an input file.
    - [load_model](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.load_model) function
        - Restores a ```Booster``` from a binary file.
            - Requires initialization of an empty ```Booster``` and load the file's data into it.

## XGBoost Classifier
- ### Chapter Goals
    - Learn how to create a scikit-learn style classifier in XGBoost.

- ### A. Following the scikit-learn API
    - [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
        - XGBoost wrapper model for classification.

## XGBoost Regressor
- ### Chapter Goals
    - Learn how to create a scikit-learn style regression model in XGBoost.

- ### A. XGBoost linear regression
    - [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)

## Feature Importance
- ### Chapter Goals
    - Understand how to measure each dataset feature's importance in making model predictions.
    - Use the matplotlib pyplot API to save a feature importance plot to a file.

- ### A. Determining important features
    - ```feature_importances_``` property of the model.

- ### B. Plotting important features
    - ```plot_importance``` function
        - By default, it uses feature weight as the importance metric, i.e. how often the feature appears in the boosted decision tree.
        - Parameter: ```importance_gain```
            - 'gain': Use [information gain](https://en.wikipedia.org/wiki/Information_gain_ratio) as the importance metric.

## Hyperparameter Tuning
- ### Chapter Goals
    - Apply grid search cross-validation to an XGBoost model.

- ### A. Using scikit-learn's GridSearchCV
    - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)

## Model Persistence
- ### Chapter Goals
    - Save and load XGBoost models with joblib API.

- ### A. The joblib API
    - Since ```XGBClassifier``` and ```XGBRegressor``` models follow the same format as scikit-learn models, we can save and load them using the [joblib](https://joblib.readthedocs.io/en/latest/) API.
