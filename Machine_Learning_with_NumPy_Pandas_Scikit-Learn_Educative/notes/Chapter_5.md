# Data Modeling with scikit-learn
1. [Introduction](#introduction)
2. [Linear Regression](#linear-regression)
3. [Ridge Regression](#ridge-regression)
4. [LASSO Regression](#lasso-regression)
5. [Bayesian Regression](#bayesian-regression)
6. [Logistic Regression](#logistic-regression)
7. [Decision Trees](#decision-trees)

## Introduction
- In this chapter:
    - Create a variety of models for linear regression and classifying data.
    - Perform hyperparameter tuning and model evaluation through cross-validation.

- ### A. Creating models for data
    - When creating these models, data scientists need to figure out the optimal [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter) to use.

## Linear Regression
- ### Chapter Goals
    - Create a basic linear regression model based on input data and labels.

- ### A. What is linear regression?
    - One of the main objectives in both machine learning and data science is finding an equation or distribution that best fits a given dataset.
    - Since finding an optimal model for a dataset is difficult, we instead try to find a good approximating distribution.
    - The term *linear regression* refers to using a linear model to represent the relationship between a set of independent variables and a dependent variable.
    - $y = a*x_1 + b*x_2 + c*x_3 + d$
        - The coefficients a,b,c and intercept d determine the model's fit.

- ### B. Basic linear regression
    - [Least squares regression](https://en.wikipedia.org/wiki/Least_squares):
        - Simplest form of linear regression
        - Minimizes the [sum of squared residuals](https://en.wikipedia.org/wiki/Residual_sum_of_squares) between the model's predictions and actual values for the dependent variable.
        - Scikit-learn's [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
        - [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination):
            - R<sup>2</sup> value tells how good of a fit the model is for the data.
            - Though traditional value is between 0 and 1, but in scikit-learn ranges from -inf to 1.
            - The closer the value is to 1, the better the model's fit on the data.

- ### Time to Code!

## Ridge Regression
- ### Chapter Goals
    - Learn about regularization in linear regression
    - Learn about hyperparameter tuning using cross-validation
    - Implement a cross validated ridge regression model in scikit-learn

- ### A.
    - Drawback of ordinary least squares regression
        - Assumption: Dataset's features are uncorrelated.
        - If many of the dataset features are linearly correlated, it makes the least squares regression model highly sensitive to noise in the data.
        - As per [Multicollinearity wiki page](https://en.wikipedia.org/wiki/Multicollinearity):
            - In multicollinearity situation, the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data.
    
    - Regularization:
        - We combat the above issue by performing regularization.
        - Goal: To not only minimize the sum of squared residuals, but to do this with coefficients as small as possible.
            - Sum of alpha * L2 norm of weights + Squared residuals
        - The smaller the coefficients, the less susceptible they are to random noise in the data.
        - Most commonly used form of regularization: [ridge regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization)
        - Side by side plots of ordinary least squares regression and ridge regression shows that ordinary least squares regression is much more susceptible to being influenced by the added noise.
        
- ### B. Choosing the best alpha
    - Scikit-learn's
        - [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
        - [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV)
            - Cross-validated ridge regression

- ### Time to Code!

## LASSO Regression
- ### Chapter Goals
    - Learn about sparse linear regression via LASSO

- ### A. Sparse regularization
    - [Wiki page](https://en.wikipedia.org/wiki/Lasso_(statistics))
    - Weight penalty term uses:
        - [L1 norm](http://mathworld.wolfram.com/L1-Norm.html)
            - In contrast, ridge regression uses L2 norm.
    - LASSO regularization tends to prefer linear models with fewer parameter values.
    - [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV):
        - Cross-validated version

- ### Time to Code!

## Bayesian Regression
- ### Chapter Goals
    - Learn about Bayesian regression techniques.

- ### A. Bayesian techniques
    - Alternative to using cross-validation for hyperparameter optimization:
        - [Bayesian](https://en.wikipedia.org/wiki/Bayesian_inference) techniques

- ### B. Hyperparameter priors
    - Bayesian ridge regression model has two hyperparameters:
        - $\alpha$
        - $\lambda$
            - Acts as precision of the model's weights.
            - Smaller its value, the greater the variance (??) between the individual weight values.
    - Both the $\alpha$ and $\lambda$ hyperparameters have gamma distribution priors.
    - Gamma probability distribution is defined by
        - a shape parameter
        - a scale parameter

- ### C. Tuning the model
    - When finding the optimal weight settings of a Bayesian ridge regression model for an input dataset, we also concurrently optimize the $\alpha$ and $\lambda$ hyperparameters based on theior prior distributions and the input data.
    - [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge)

- ### Time to Code!

## Logistic Regression
- ### Chapter Goals
    - Learn about logistic regression for linearly separable datasets.

- ### A. Classification
    - Performs regression on [logits](https://en.wikipedia.org/wiki/Logit)
        - This then allows us to classify the data based on model probability predictions.
    - Scikit-learn's [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)

- ### B. Solvers
    - Scikit-learn's [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

- ### C. Cross-validated model
    - Cross-validated [logistic regression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)

- ### Time to Code!

## Decision Trees
- ### Chapter Goals
    - Learn about decision trees and how they are constructed.
    - Learn how decision trees are used for classification and regression.

- ### A. Making decisions
    - [Wiki page](https://en.wikipedia.org/wiki/Decision_tree_learning)
    - Scikit-learn's:
        - [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
        - [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)

- ### B. Choosing features
    - Criteria for feature selection at each node:
        - Classification:
            - [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
        - Regression:
            - MSE (Mean squared error)
            - MAE (Mean absolute error)

- ### Time to Code!
