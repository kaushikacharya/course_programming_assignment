# A Simple Classification Problem
1. [Spoliers](#spoilers)
2. [Classification Problems](#classification-problems)
3. [Model for Classification Problems](#model-for-classification-problems)
4. [Sigmoid and Logistic Regression](#sigmoid-and-logistic-regression)
5. Quiz
6. [Loss](#loss)
7. [Binary Cross-Entropy Loss in PyTorch](#binary-cross-entropy-loss-in-pytorch)
8. [Imbalanced Dataset](#imbalanced-dataset)
9. [Model Configuration, Training, and Predictions for Classfication](#model-configuration-training-and-predictions-for-classfication)
10. [Decision Boundary](#decision-boundary)
11. [Classification Threshold and Confusion Matrix](#classification-threshold-and-confusion-matrix)
12. [Metrics](#metrics)
13. [Trade-offs and Curves](#trade-offs-and-curves)
14. [Best + Worst Curves and Models](#best--worst-curves-and-models)
15. [Putting It All Together](#putting-it-all-together)
16. [Recap](#recap)
17. [Quiz](#quiz-1)
18. [Challenge 5 - A Simple Classification Problem](#challenge-5---a-simple-classification-problem)
19. [Solution Review: A Simple Classification Problem](#solution-review-a-simple-classification-problem)

## Spoilers
- ### What to expect from this chapter
    - Build a model for **binary classification**
    - Understand the concept of **logits** and how it is related to probabilities.
    - Use **binary cross-entropy loss** to train a model.
    - Use the loss function to handle **imbalanced datasets**.
    - Understand the concepts of **decision boundary** and **separability**.
    - Learn how the **choice of a classification threshold** impacts evaluation metrics.
    - Build **ROC** and **Precision-Recall** curves.

- ### Imports
    - For this chapter, we will need the following imports:
        ```
        import numpy as np

        import torch
        import torch.optim as optim
        import torch.nn as nn
        import torch.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        from sklearn.datasets import make_moons
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, roc_curve, \
        precision_recall_curve, auc

        from stepbystep.v0 import StepByStep
        ```

- ### Jupyter notebook
    - A Jupyter notebook containing the entire code will be available at the end of the chapter.


## Classification Problems
- ### Intorduction to classification problems
    - Important: In a classification model, the output is the predicted probability of the positive class.

- ### Data generation
    - Generate a toy dataset using Scikit-Learn's [make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
        - Two features x1, x2
    - Train-validation split using  Scikit-Learn's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        - Remember, the split should always be the first thing we do.
    - Next, standardize the features using Scikit-Learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        - Only use the training set to fit the ```StandardScaler```.
        - Then use its ```transform``` method to apply the preprocessing step to all datasets:
            - training
            - validation
            - test

- ### Data preparation

## Model for Classification Problems
- ### Model
    - Logistic regression is one of the most straightforward models for classification problem.
        - Instead of simply presenting it, we are going to build up to it.
        - Twofold rationale:
            - To clarify why this algorithm is called logistic regression.
            - To have a clear understanding of what a logit is.
    
    - #### Linear Regression    
    
    - #### Logits
        - z = b + w1*x1 + w2*x2
            - This equation is strikingly similar to the original linear regression model.
        - z is called logit.

        - Questions:
            - Does it mean a logit is the same as linear regression?
                - Not quite.
                - One fundamental difference between them:
                    - There is no error term (epsilon) in the logit equation above.
            - If there is no error term, where does the uncertainty come from?
                - That is the role of probability.
                - Instead of assigning a dsta point to a discrete label, we will compute the probability of a dsta point belonging to the positive class.
    
    - #### Probabilities

    - #### Odds ratio
        - The ratio between the probability of success (p) and the probability of failure (q):
        - odd ratio = p/q

        - Figures plotted:
            - Odds ratio (y axis) vs Probability (x axis)
                - Log odds ratio is symmetrical.
    
        - Explanation for the need of symmetrical plot:
            - If the function were not symmetrical, different choices for the positive class would produce models that are not equivalent.

    - #### Log odds ratio
        - Advantages:
            - Symmetrical function
            - Maps probabilities into real numbers instead of only positive ones.
                - ?? What exactly do we gain from this?
                    - Is it that this property resembles similarity with linear regresssion i.e., logit is a real number.
        - Probabilities that add up tp 100 % correspond to log odd ratios that are same in absolute value.
            - KA: This is shown with an empirical example.
            - KA: Mathematical derivation:
                ```
                p = probability of data point belonging to a positive class.
                q = probability of data point belonging to a negative class.
                q = 1-p

                log_odd_ratio(p) = log (p/(1-p))
                log_odd_ratio(q) = log_odd_ratio(1-p)
                                 = log((1-p)/(1-(1-p)))
                                 = log((1-p)/p)
                                 = log(1-p) - log(p)
                                 = (-)(log(p) - log(1-p))
                                 = (-)log(p/(1-p))
                ```
        - Log odds ratio and probability
            - Figure 1: Probability maps into a log odds ratio
                - x axis: Probability
                - y axis: Log odds ratio
            - Figure 2: By flipping the horizontal and vetical axes, we invert the function.
                - Maps log odds ratio into a probability.
                - x axis: Log odds ratio
                - yaxis: Probability


## Sigmoid and Logistic Regression

- ### From logits to probabilities
    - $b + w1*x1 + w2*x2 = z = log (p/(1-p))$
    - $e^z$ = $p/(1-p)$
        - Leads to sigmoid function
        - $p = 1/(1 + e^{-z})$

- ### Sigmoid
    - Two different ways of using a sigmoid in PyTorch:
        - [torch.sigmoid](https://pytorch.org/docs/master/generated/torch.sigmoid.html)
            - Simple function
        - [nn.Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
            - Full fledged class inherited from ```nn.Module```.
            - For all intents and purposes, a model on its own.
                - Simple and straightforward model, only implements a ```forward``` method.
    - Why do we need a model for a sigmoid function?
        - Models can be used as layers of another larger model. That is exactly what we are going to do with the sigmoid class.

    - #### Sigmoid, nonlinearities and activation functions
        - The sigmoid function is nonlinear. Can be used to map logits into probabilities.
        - Nonlinear functions play a fundamental role in neural networks. Usual name: activation functions.
        - Timeline:
            - sigmoid => hyperbolic tangent (tanh) => Rectified Linear Unit (ReLU)
    
- ### Logistic regression
    - #### A note on notation
        - Notation for vectorized features.

## Quiz

## Loss
- ### Defining the appropriate loss
    - A binary classification problem calls for the binary cross-entropy (BCE) loss, sometimes known as **log loss**.
    - BCE loss computation requires:
        - Predicted probabilites (as returned by the sigmoid function)
        - True labels (y)
    - Case: Data point belongs to the positive class (y=1):
        - error: $log(P(y=1))$
    - Case: Data point belongs to the negative class (y=0):
        - error: $log(1 - P(y=1))$
        - The model outputs the probability of a point belonging to the positive, not the negative class.

- ### Binary cross-entropy loss
    - For the binary cross-entropy loss, we simply take the average of the errors and invert its sign.
    - Shows how the same equation changes from where positive and negative classes are explicitly calculated to a single expression equation.

## Binary Cross-Entropy Loss in PyTorch
- ### BCELoss
    - PyTorch implements [```nn.BCELoss```](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
        - A higher-order function that returns the actual loss function.
    - Two optional arguments:
        - ```reduction```
            - Default: ```mean```
        - ```weight```
            - Default: ```none```
                - Every data point has equal weight.
    - Important: Make sure to pass the predictions first and then the labels to the loss function.

- ### BCEWithLogitsLoss
    - Comparison with BCELoss:
        - BCELoss function takes probabilites as an argument (along with labels).
        - BCEWithLogitsLoss function takes logits as an argument instead of probabilities.
    - What does that mean in practical terms?
        - It means sigmoid should not be added as last layer of the model when using BCEWithLogitsLoss function.
        - This loss function combines both the sigmoid layer and the former cross-entropy loss into one.
        - As per [PyTorch's documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), this loss function is more numerically stable than using a plain ```Sigmoid``` followed by a ```BCELoss```.
    - Using the right combination of model and loss function:
        - IMHO the wording for this section is confusing.
        - Since ```nn.BCEWithLogitsLoss``` combines sigmoid layer and cross-entropy loss, one should not use ```nn.Sigmoid``` as part of model's final layer.
        - Whereas ```nn.BCELoss``` should be used with model producing probabilies as output. For that ```nn.Sigmoid``` can be used as last layer in the model.
    - It is a higher-order function and takes three optional arguments:
        - ```reduction```
        - ```weight```
        - ```pos_weight```
            - weight of positive examples
    - We are dealing with
        - single label binary classification (only one label per point)
    - Author warns to not get confused with **class number** mentioned in the documentation.
        - Class number, c corresponds to the number of different labels associated with a data point.
    
## Imbalanced Dataset
- ### Introduction to the imbalanced dataset
- ### The pos_weight argument
    - To compensate for the imbalance, one can set the weight equals the ratio of negstive to positive examples.
        - pos_weight = #points in negative class/ #points in positive class.
- ### Weighted average
    - For weighted average, one needs to do the following
        - a) ```reduction``` = ```sum```
            - This would return only the output sum without any division.
        - b) Then divide by the weighted count manually.

## Model Configuration, Training, and Predictions for Classfication
- ### Model configuration
    - We only need to define:
        - a model
        - an appropriate loss function
        - an optimizer
    
- ### Model training
    - When validation loss can be lower than training loss?
        - Blog referred [PyImageSearch](https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/)
            - Reasons are provided by Aurélien Géron.
        - Reasons:
            - #1: Regularization applied during training, but not during validation/testing.
                - Regularization methods often sacrifice training accuracy to improve validation/tesing accuracy.
            - #2: Training loss is measured during each epoch while validation loss is measured after each epoch.
            - #3: The validation set may be easier than the training set (or there may be leaks).

- ### Making predictions
    - The points where the logits (z) equal zero determine the boundary between posiitive and negative examples.

- ### Jupyter notebook
    - [Notebook](../code/Training_Predictions_Classfication.ipynb)

## Decision Boundary
- ### Decision boundary for logistic regression
    - A logistic regression always separates two classes with a straight line.
    - The more separable the classes are, the lower the loss will be.

- ### Validation dataset decision boundary

- ### Are my data points separable?

- ### Effect of increasing dimensions
    - Kernel: The function to create additional dimensions.
        - Kernel trick of Support Vector Machines (SVMs) mentioned.
    - Why are we talking about SVMs in deep learning course?
        - Neural networks may also increase the dimensionality.
            - Adding a hidden layer with more units than the number of features.
    - Activation functions:
        - Functions that introduce non-linearity into neurons.
        - Example: Sigmoid

## Classification Threshold and Confusion Matrix
- ### Using different thresholds for classification
    - Visualize the probabilities
        - Contour plot of the probabilities and the decision boundary as a straight line.
        - Probabilities on a line.
    - Probability line:
        - Plot the positive class points below the probability line and negaive class points above the probability line.
            - This will place the points in different quadrants which reflect the quadrants of confusion matrix.

- ### Confusion matrix
    - Scikit-Learn's
        - [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
        - [confusion matrix's display](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay)
    
    - Conventions:
        - There are various conventions to show the confusion matrix.
        - This course will stick to Scikit-Learn's convention.

## Metrics
- ### Most commonly used metrics
    - Lot of metrics are constructed using TN, FP, FN, TN
    - #### True and false-positive rates
        - $TPR = TP/(TP + FN)$
        - TPR also called **recall**.
        - Example scenario for false negative being really bad:
            - Machine failed to detect an actual threat.
        - The trade-off between TPR and FPR
            - KA: It is a trade-off between FP and FN.
    
    - #### Precision and recall
        - $Recall = TP/(TP + FN)$
        - $Precision = TP/(TP + FP)$
        - Trade-off between precision and recall.
    
    - #### Accuracy
        - $Accuracy = (TP + TN)/(TP + TN + FP + FN)$
        - Higher the accuracy, the better.
        - For imbalanced dataset, relying on accuracy can be misleading.
        - Accuracy may be misleading because it does not involve a trade-off with another metric like the above ones.

## Trade-offs and Curves
- ### Introduction to ROC and PR curves
    - ROC (Receiver operating characteristic) curve
        - x axis: False positive rate (FPR)
        - y axis: True positive rate (TPR)
    - PR (Precision-recall) curve
        - x axis: Recall
        - y axis: Precision

- ### Low threshold
    - Loose threshold: Since model is not required to be very confident to consider a data point to be positive.

- ### High threshold

- ### ROC and PR curves
    - Scikit-Learn's
        - [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
        - [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
    - **Doubt**:
        - KA: Not sure how precision = 1 for the edge case of threshold=1 (i.e. every prediction is negative) for the formula provided in [Metrics](#metrics) lesson.
            - FP = TP = 0
            - This leads to zero denominator for the precision metrics calculation.
        - KA: Scikit-Learn's [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) uses the parameter ```zero_dvision``` to handle this case.
    
- ### The precision quirk
    - In general, raising the threshold:
        - Reduces FP (false positives) => Increases precision
    - But, along the way, we may lose some of the true positives, which will temporarily reduce precision.

## Best + Worst Curves and Models
- ### Best and worst curves
    - **Doubt**:
        - KA: How is precision=1 for the best precision-recall curve.

- ### Comparing models
    - Best curve => Best model
    - Measure [area under curve (auc)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)

- ### Further reading
    - Scikit-Learn's:
        - [Receiver Operating Characteristic (ROC)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
        - [Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

## Putting It All Together
- ### Overall view of the result
    - #### Data preparation
    - #### Model configuration
    - #### Model training
    - #### Model evaluation

- ### Jupyter notebook
    - [Notebook](../code/Putting_together_part4.ipynb)

## Recap
- ### General overview
    - We have covered:
        - Defining a binary classification problem.
        - Generating and preparing a toy dataset using Scikit-Learn's ```make_moons``` method.
        - Defining logits as the result of a linear combination of features.
        - Understanding what odds ratio and log odds ratio are.
        - Figuring we can interpret logits as log odds ratio.
        - Mapping logits into probabilities using a sigmoid function.
        - Defining a logistic regression as a simple neural network with a sigmoid function in the output.
        - Understanding the binary cross-entropy loss and its PyTorch implementation ```BCELoss```.
        - Understanding the difference between the ```BCELoss``` and ```BCEWithLogitsLoss```.
        - Highlighting the importance of choosing the correct combination of the last layer and loss function.
        - Using PyTorch's loss functions` arguments to handle imbalanced datasets.
        - Configuring the model, loss function, and optimizer for a classification problem.
        - Training a model using the ```StepByStep``` class.
        - Understanding that the validation loss may be smaller than the training loss.
        - Making predictions and mapping predicted logits to probabilities.
        - Using a classification threshold to convert probabilities into classes.
        - Understanding the definition of decision boundary.
        - Understnading the concept of separability of classes and how it is related to dimensionality.
        - Exploring different classification thresholds and its effect on the confusion matrix.
        - Reviewing typical metrics for evaluating classification algorithms like true and false positive rates, precision and recall.
        - Building ROC and precision-recall curves out of metrics computed for multiple thresholds.
        - Understanding the reason behind the quirk of losing precision while raising the classification threshold.
        - Defining the best and worst possible ROC and PR curves.
        - Using the area under curve to compare different models.

- ### Jupyter notebook
    - [Notebook](../code/Chapter06.ipynb)

## Quiz
- Q8: Is there trade-off between these two matrices:
    - One of the option: TPR and Precision
    - KA: TPR is also called recall. And there's trade off between precision and recall. So IMHO, this option should also be correct.

## Challenge 5 - A Simple Classification Problem
- ### Challenge
- ### Jupyter notebook
    - [Notebook - Question](../code/Challenges05_question.ipynb)
    - [Notebook - Answer](../code/Challenges05_answer.ipynb)

## Solution Review: A Simple Classification Problem
- ### Solution
    - [Notebook](../code/Challenges05.ipynb)
