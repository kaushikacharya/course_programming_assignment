# Deep Learning with TensorFlow
1. [Introduction](#introduction)
2. [Model Initialization](#model-initialization)
3. [Logits](#logits)
4. [Metrics](#metrics)
5. [Optimization](#optimization)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Linear Limitations](#linear-limitations)
9. [Hidden Layer](#hidden-layer)
10. [Multiclass](#multiclass)
11. [Softmax](#softmax)

## Introduction
- ### Chapter Goals
    - Learn about one of the most essential neural networks used in deep learning: the multilayer perceptron.
    - Learn how to use the TensorFlow framework to manipulate this neural network model.

- ### A. Multilayer perceptron
    - Learn:
        - How to code your own multilayer perceptron (MLP).
        - Apply it to the task of classifying 2-D points in the Cartesian plane.
        - The basics of *computation graph* - the structure of a neural network.
    - The number of hidden layers represents how "deep" a model is.

- ### B. TensorFlow
    - [TensorFlow](https://www.tensorflow.org/)
    - The name is derived from *tensors*, which are basically multidimensional (i.e. generalized) vectors/matrices.
    - When writing code, it may be easier to think of anything with numeric value as being a tensor.
    - In TensorFlow, we first create the computation graph structure and then train and evaluate the model with input data and labels.
    - ```import tensorflow as tf```

## Model Initialization
- ### Chapter Goals
    - Define a class for an MLP model.
    - Initialize the model.

- ### A. Placeholder
    - [tf.compat.v1.placeholder](https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder)
        - When building the computation graph of model, it acts as a "placeholder" for the input data and labels.
        - [type](https://www.tensorflow.org/programmers_guide/tensors#data_types)

- ### B. Shapes and dimensions
    - ```shape``` argument:
        - 1. Number of data points passed (Referred as *batch size* when training the model).
        - 2. Number of features in the dataset.
    - Each data point also has a label, which is used to identify and categorize the data.

- ### C. Different amounts of data
    - ```None``` for dimension size:
        - Allows dimension of any size.
        - Useful because we will use neural network on input data with different input sizes.

- ### Time to Code!

## Logits
- ### Chapter Goals
    - Build a single fully-connected layer.
    - Output logits, AKA log-odds.

- ### A. Fully-connected layer
    - The single fully-connected layer:
        - The input layer i.e. ```self.inputs``` is directly connected to the output layer.
        - Each of the ```input_size``` neurons in the input layer has a connection to each neuron in the output layer, hence the fully-connected layer.
    - Implemented using [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

- ### B. Weighted connections
    - Weight on a connection from neuron A into neuron B:
        - Tells how strongly A affects B.
        - Whether that effect is positive or negative i.e. direct vs inverse relationship.

- ### C. Logits
    - In classification problems they represent log-odds:
        - Maps a probability between 0 and 1 to a real number.

- ### D. Regression
    - For regression we would have our model directly output the logits rather than convert them to probabilities.

- ### Time to Code!

## Metrics
- ### Chapter Goals
    - Convert logits to probabilities.
    - Obtain model predictions from probabilities.
    - Calculate prediction accuracy.

- ### A. Sigmoid
    - Logits represent real number mappings from probabilities.
    - [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
        - Inverse mapping to obtain the original probabilities.
    - ```tf.math.sigmoid```
        - Sigmoid function in TensorFlow to extract probabilities from the logits.
    - Binary classification:
        - ```output_size = 1```
        - Sigmoid represents the probability the model gives to the input data being labeled 1.

- ### B. Predictions and accuracy
    - [tf.math.reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)
        - Function to produce the overall mean of a tensor's numeric values.
        - Can use this to calculate prediction accuracy as the mean number of correct predictions across all input data points.

- ### Time to Code!

## Optimization
- ### Chapter Goals
    - Know the relationship between training, weights and loss.
    - Understand the intuitive definition of loss.
    - Obtain the model's loss from logits.
    - Write a training operation to minimize the loss.

- ### A. What is training?
    - In [lesson 3](#logits), we discussed the weights associated with connections between neurons.
    - These weights are *trainable variables*:
        - We need to train neural network to find the optimal weights for each connection.
    - Training involves setting up a *loss* function.
        - The loss function tells us how bad the neural network's output is compared to the actual labels.

- ### B. Loss as error
    - Regression problems:
        - Common loss functions
            - L1 norm
            - L2 norm

- ### C. Cross entropy
    - [Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
        - Also referred to as *log loss*.
    - We want a loss function that is
        - Small: When the probability is close to the label.
        - Large: When the probability is far from the label.

- ### D. Optimization
    - [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
        - Model updates its weights based on a *gradient* of the loss function until it reaches the minimum loss.
    - [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
        - Finds optimal gradient for the model.
        - [tf.compat.v1.train.GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/GradientDescentOptimizer)
    - Learning rate:
        - *Large*: The model could potentially reach the minimum loss quicker, but could also overshoot the minimum.
        - *Small*: More likely to reach the minimum, but may take longer.
        - [Adam](https://arxiv.org/abs/1412.6980)
            - Popular and effective optimization method.
            - [tf.compat.v1.train.AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer)

- ### Time to Code!
    - ```tf.cast```
    - ```tf.nn.sigmoid_cross_entropy_with_logits```
        - Computes sigmoid cross entropy given logits.
        - Argument ```labels```: A tensor of the same type and shape as argument ```logits```.
            - Hence the usage of ```tf.cast```.
    - ```tf.math.reduce_mean```
        - Loss = Mean(cross entropy errors)
    - Additional info (KA): [AdamOptimizer methods](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer#methods_2)
        - Describes the various methods.
        - ```minimize```
            - Add operations to minimize loss by updating list of variables.
            - This method combines calls:
                 - ```compute_gradients()```
                    - Compute gradients of loss for the variables in the variable list.
                 - ```apply_gradients()```
                    - Apply gradients to variables.

## Training
- ### Chapter Goals
    - Learn how to feed data values into a neural network.
    - Understand how to run a neural network using input values.
    - Traing the neural network on batched input data and labels.

- ### A. Running the model
    - Run model on input data using:
        - [tf.compat.v1.Session](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session)
        - ```tf.compat.v1.placeholder```
    - Code written in previous lessons was to build computation graph of the neural network i.e. layers and operations.
    - [run](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session#run) function
        - To train or evaluate the model on real input data.

- ### B. Using run
    - ```feed_dict``` argument:
        - To pass values into certain tensors in the computation graph.

- ### C. Initializing variables
    - [tf.compat.v1.global_variables_initializer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables_initializer)

- ### D. Training logistics
    - Choosing the batch size is a speed-precision tradeoff.

- ### Time to Code!

## Evaluation
- ### Chapter Goals
    - Evaluate model performance on a test set.

- ### A. Evaluating using accuracy
    - ```accuracy``` metric defined in [Metrics](#metrics) lesson.

- ### B. Different amounts of data
    - It is good practice to split up a dataset into three sets:
        - *Training set* (~80% of dataset)
            - Used for model training and optimization.
        - *Validation set* (~10% of dataset)
            - Used to evaluate the model in between training runs, e.g. when tweaking model parameters like batch size.
        - *Test set* (~10% of dataset)
            - used to evaluate the final model, usually through some metric.

- ### Time to Code!

## Linear Limitations
- ### Chapter Goals
    - Understand the limitations of a single layer perceptron model.

- ### A. Linear decision boundary
    - Example to illustrate limitation of single layer perceptron
        - Predict whether 2-D points are inside or outside a circle.
            - Non-linear decision boundary.

## Hidden Layer
- ### Chapter Goals
    - Add a hidden layer to the model's layers.
    - Understand the purpose of non-linear activations.
    - Learn about the ReLU activation function.

- ### A. Why a single layer is limited
    - Equation of the circle boundary:
        - $x^2 + y^2 = r^2$
    - This can't be modeled by a single linear combination.

- ### B. Hidden layers

- ### C. Non-linearity
    - Non-linearities to model is added through activation functions.
    - The three most common activation functions used in deep learning:
        - [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function#Tanh)
        - [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
        - [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)

- ### ReLU
    - $ReLU(x) = mainx(0,x)$
    - Shows an example to prove that with enough linear combinations and ReLU activations, a model can learn quadratic transformation.
        - $f(x) = ReLU(x) + ReLU(-x) + ReLU(2x-2) + ReLU(-2x-2)$
        - This represents somewhat like the quadratic function, $f(x) = x^2$
    - What makes ReLU work well in general purpose situations?
        - Simplicity
            - Simplicity w.r.t. its gradient, allows it to avoid [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).
        - Mapping negative values to 0 helps the model train faster and avoid overfitting.

- ### Time to Code!

## Multiclass
- ### Chapter Goals
    - Learn about multiclass classification.
    - Understand the purpose of multiple hidden layers.
    - Learn the pros and cons of adding hidden layers.

- ### A. Multiclass classification

- ### B. One-hot

- ### C. Adding hidden layers
    - Adding an extra hidden layer may decrease the number of training iterations needed for convergence compared to maintaining a single hidden layer.
    - General rule of thumb:
        - Usually up to 3 hidden layers.
            - Unless it is a complicated problem which would more hidden layer.
                - Example: Google's Alpha Go needed more than a dozen layers.

- ### D. Overfitting
    - The more hidden layers or neurons a neural network has, the more prone it is to overfitting the training data.

- ### Time to Code!

## Softmax
- ### Chapter Goals
    - Update the model to use the softmax function.
    - Perform multiclass classification.

- ### A. The softmax function
    - [Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
        - Generalization of the sigmoid function as we move from binary classification to multiclass classification.
    - The softmax function takes in a vector of numbers (logits for each class), and converts the numbers to a probability distribution.
    - Each class's individual probability is based on how large its logit is relative to the sum of all classes's logits.
    - [tf.nn.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)
    - Sigmoid cross entropy for binary classification.
    - Softmax cross entropy for multiclass classification.
        - Cross entropy is calculated *for each class* and then averaged at the end.

- ### B. Predictions
    - [tf.math.argmax](https://www.tensorflow.org/api_docs/python/tf/math/argmax)
        - To return the index with the maximum probability.

- ### Time to Code!
    - Softmax cross entropy:
        - [tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/softmax_cross_entropy_with_logits_v2)
