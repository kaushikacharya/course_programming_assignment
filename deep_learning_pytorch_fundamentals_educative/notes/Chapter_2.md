Visualizing Gradient Descent
============================

- Gradient Descent Steps:
  ----------------------
    - Initialising parameters
        - Random initialization
    - Forward pass
        - Compute model's predictions using current values of the parameters/weights.
    - Computing error and loss
    - Computing gradients
    - Updating parameters


- Gradient Descent:
  ----------------
    - Introduction to gradient descent
    - Why visualize gradient descent?
        - To understand how the following impacts speed of the training:
            - characteristics of data
            - hyper-parameters (e.g. mini batch, training rate)
    - Model used in this chapter: Linear Regression
    
- Data Generation:
  ---------------
    - Synthetic data generation
        - Using numpy.random
    - Train, Test, Validation split

- Compute the Loss:
  ----------------
    - Loss vs Error:
        - **Error**: Difference between the actual value (label) and the predicted value computed for a single data point.
        - **Loss**: Aggregation of errors for a set of data points.
    - Stability vs Speed:
        - Compute loss for a subset of N data points.
    - Gradient descent types:
        - Batch
        - Stochastic
        - Mini-batch
    - Computing loss for regression:
        - Mean Squared Error (MSE): Average of the squared errors.

- Computing the Loss Surface
  --------------------------
    - Loss Surface
        - Additional resource to understand **meshgrid**:
            - https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
                - Explains the purpose of using meshgrid
                    - Replace [slow python loops](https://realpython.com/numpy-array-programming/) by faster vectorized operations available in mumpy.
        - Computing the predictions
            - e.g. In linear regression, meshgrid enables prediction computation for all the combinations of w, b.
        - Computing the errors
        - Computing the Mean Squared Error
            - Grid of losses
            - Loss surface
            - Contour plot
        - Cross-sections
            - Purpose: Get effect on the loss of changing a single parameter while keeping everything else constant. This is called **gradient**.

- Compute the Gradients
  ---------------------
    - Introduction to gradients
        - Gradient = Partial derivative
        - Partial derivatives of MSE wrt a) b b) w
            - Chain rule applied for computing the gradients.
    - Visualizing gradients
        - In the example, gradient wrt b is higher compared to the one wrt w. This is reflected in the cross-section plot (steeper curves).
        - Geometrically explained with an example
            - Partial gradient computed at a certain value.
                - Also explained that geometric computation of the gradient would approximate true partial gradient as delta_b, delta_w approaches zero.
    - Backpropagation:
        - "Chained" gradient descent

- Update the Parameters
  ---------------------
    - ### Learning rate:
        - Multiplicative factor applied to the gradient.

- Learning Rate
  -------------
    - **Section Objective***: Learn how the choice of learning rate affects the model.

    - ### Introduction to the learning rate
        - https://cs231n.github.io/neural-networks-3/#loss-function
            - Refers to the graph (epcoh vs loss plot) mentioned here which shows the impact of learning rate.
        - Leaning rate: Step size in the gradient direction.

    - ### Choosing a learning rate
        - Small learning rate:
            - Time is money, especially on using GPU in the cloud.
                - Incentive to try bigger learning rates.
        - Big learning rate:
            - Moving back and forth from one side to another side of the minimum.
        - Very big learning rate:
            - Shows that not only we might move from one side to another but also end up with higher loss.
            - Also shows that need not be the case for each of the parameters.
            - **Highlights** the problem with a single learning rate for each of the parameters.

- Scaling the Dataset
  -------------------
    - ### Objective
        - Learn how scaling the dataset can have a meaningful impact on gradient descent.

    - ### Overview of learning rate results
        - Conclusion drawn from results in previous section:
            - Ideal scenario: All the curves are equally steep, so the learning rate is closer to optimal for all of them.

    - ### Achieving equally steep curves
        - How?
            - Correctly scale the dataset.
        - Bad dataset
            - Size of the learning rate is limited by the steepest curve.
        - Scaling / standardizing / normalizing
            - scikit-learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
                - Transforms a feature to end up with zero mean and unit standard deviation.
                - Commonly referred as normalization
                - Technical term: Standardization
                - Apply StandardScaler fit on train set only.
                - Apply transform to train, validation and test sets.
            - An important preprocessing step for PCA too.
            - Centering the features at zero
                - ?? Handles **vanishing gradients**.

- Rinse and Repeat
  ================
    - ### Introduction to Epcoh
        - Definition
        - Updates and gradient descent
            - The number of updates depends on the type of gradient descent used.
    
    - ### Restarting the process
        - Number of epochs
            - Simple model: Large number of epochs can be afforded.
            - Complex model: Couple of dozen epochs may be enough.
    
    - ### The path of gradient descent
        - Factors:
            - Learning rate
            - Shape of the loss surface
            - Number of points used to compute the loss.

- Recap
  =====
    - ### General overview
        - Recap:
            - Defining a simple linear regression model.
            - Generating synthetic data for it.
            - Performing a train-validation split on our dataset.
            - Randomly initializing the parameters of our model.
            - Performing a forward pass; that is, making predictions using our model.
            - Computing the errors associated with our predictions.
            - Aggregating the errors into a loss (MSE).
            - Learning that the number of points used to compute the loss defines the kind of gradient descent we’re using: batch (all), mini-batch, or stochastic (1).
            - Visualizing an example of a loss surface and using its cross-sections to get the loss curves for individual parameters.
            - Learning that a gradient is a partial derivative, and it represents how much the loss changes if one parameter changes a little bit.
            - Computing the gradients for our model’s parameters using equations, code, and geometry.
            - Learning that larger gradients correspond to steeper loss curves.
            - Learning that backpropagation is nothing more than “chained” gradient descent.
            - Using the gradients and a learning rate to update the parameters.
            - Comparing the effects on the loss of using small, big, and very big learning rates.
            - Learning that loss curves for all parameters should be, ideally, similarly steep.
            - Visualizing the effects of using a feature with a larger range, making the loss curve for the corresponding parameter much steeper.
            - Using Scikit Learn’s StandardScaler to bring a feature to a reasonable range, and thus making the loss surface more bowl-shaped and its cross-sections similarly steep.
            - Learning that preprocessing steps like scaling should be applied after the train-validation split to prevent leakage.
            - Figuring that performing all steps (forward pass, loss, gradients, and parameter update) makes one epoch.
            - Visualizing the path of gradient descent over many epochs and realizing it is heavily dependent on the kind of gradient descent used: batch, mini-batch, or stochastic.
            - Learning that there is a trade-off between the stable and smooth path of batch gradient descent and the fast and somewhat chaotic path of stochastic gradient descent, making the use of mini-batch gradient descent a good compromise between the other two.

    - ### Jupyter notebook
        - [Notebook](../code/Chapter02.ipynb)

- Challenge 1 - Visualizing Gradient Descent
  ==========================================

    - Challenge
      ---------

    - Jupyter Notebook
      ----------------
        - [Jupyter Notebook](../code/Challenges01_question.ipynb)

- Soution Review - Visualizing Gradient Descent
  =============================================

    - Solution
      --------
