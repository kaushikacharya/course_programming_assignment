# Deep Learning with Keras
1. [Introduction](#introduction)
2. [Sequential Model](#sequential-model)
3. [Model Output](#model-output)
4. [Model Configuration](#model-configuration)
5. [Model Execution](#model-execution)
6. Quiz
7. [Course Conclusion](#course-conclusion)

## Introduction
- ### Chapter Goals
    - Learn how to use the Keras API, a simple and compact API for creating neural networks.
    - Use Keras to build a multilayer perceptron model for multiclass classification.

- ### A. The Keras API
    - Downside of TensorFlow:
        - The code can be bit complex, especially when setting up a model for training or evaluation.
    - [Keras](https://keras.io/)
        - A simpler alternative to TensorFlow.
        - Keras is often run on top of TensorFlow, acting as a wrapper API to make the coding simpler.

- ### B. Multilayer perceptron
    - This chapter will focus on the Keras implementation of an MLP.
    - Specific details on the MLP model is covered in [previous chapter](./Chapter_8.md).

## Sequential Model
- ### Chapter Goals
    - Initialize an MLP model in Keras.

- ### A. Building the MLP
    - [Sequential](https://keras.io/models/sequential/):
        - In Keras, every neural network is an instance of the ```Sequential``` object.
        - Acts as the container of the neural network, allowing us to build the model by stacking multiple layers inside the ```Sequential``` object.
    - [Dense](https://keras.io/api/layers/core_layers/dense/)
        - Most commonly used Keras neural network layer.
        - Represents a fully-connected layer in the neural network.
        - The last two sections in the Keras documentation explains the arguments: ```units```, ```input_dim```:
            - Input shape
            - Output shape

- ### Time to Code!

## Model Output
- ### Chapter Goals
    - Add the final layers to the MLP for multiclass classification.

- ### A. Final layer activation
    - In my opinion, an inaccurate information provided in the course:
        - In Keras, the cross-entropy loss functions only calculate cross-entropy, without applying the sigmoid/softmax function to the MLP output.
        - Therefore, we can have the model directly output class probabilities instead of logits (i.e. we apply sigmoid/softmax activation to the output layer).
    - As per [BinaryCrossEntropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) documentation:
        - Contains argument ```from_logits```

- ### Time to Code!

## Model Configuration
- ### Chapter Goals
    - Learn how to configure a Keras model for training.

- ### A. Configuring for training
    - [compile](https://keras.io/api/models/model_training_apis/) function
        - In Keras, a single call to compile function allows to set up all the training requirements for the model.
        - Required argument: Optimizer
        - Two main keyword arguments:
            - ```loss```
                - Binary classification: ```binary_crossentropy```
                    - Binary cross-entropy function.
                - Multiclass classification: ```categorical_crossentropy```
                    - Multiclass cross-entropy function.
            - ```metrics```
    - [List of optimizers](https://keras.io/optimizers/)

- ### Time to Code!

## Model Execution
- ### Chapter Goals
    - Understand the facets of model execution for Keras models.

- ### A. Training
    - [fit](https://keras.io/api/models/model_training_apis/#fit-method) function:
        - Use ```Sequential``` model's ```fit``` function to train the model on input data and labels.
        - Input data type:
            - TensorFlow: tensor objects for any sort of data.
            - Keras: Simply use NumPy arrays.
        - Output:
            - [History](https://keras.io/callbacks/#history) object which records the training metrics.

- ### B. Evaluation
    - [evaluate](https://keras.io/api/models/model_training_apis/#evaluate-method) function
        - We use the ```Sequential``` model's ```evaluate``` function.

- ### C. Predictions
    - [predict](https://keras.io/api/models/model_training_apis/#predict-method) function

## Course Conclusion
- Topics covered include the following:
    - Data analysis/visualization
    - Feature engineering
    - Supervised learning
    - Unsupervised learning
    - Deep learning

- Industry-standard frameworks learned how to use:
    - NumPy
    - pandas
    - scikit-learn
    - XGBoost
    - TensorFlow
    - Keras