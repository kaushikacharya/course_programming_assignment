# Rethinking the Training Loop
1. [Spoilers](#spoilers)
2. [Introducing Higher-Order Functions](#introducing-higher-order-functions)
3. [Rethinking the Training Loop](#rethinking-the-training-loop-1)
4. [Building a Dataset](#building-a-dataset)
5. [DataLoader](#dataloader)
6. Quiz
7. [Mini-Batch Inner Loop and Training Split](#mini-batch-inner-loop-and-training-split)
8. [Evaluation](#evaluation)
9. [TensorBoard](#tensorboard)
10. [SummaryWriter](#summarywriter)

- ## Spoilers
    - ### What to expect from this chapter
        - Build a function to perform training steps.
        - Implement our own dataset class.
        - Use data loaders to generate mini-batches.
        - Build a function to perform mini-batch gradient descent.
        - Evaluate our model.
        - Integrate TensorBoard to monitor model training.
        - Save/checkpoint our model to disk.
        - Load our model from disk to resume training or to deploy.
    - ### Imports
    - ### Jupyter notebook
        - Available at the end of the chapter.

- ## Introducing Higher-Order Functions
    - ### Using different optimizers, loss, and models.
        - Higher-order function is very useful in reducing boilerplate.
    
    - ### Higher-order functions.
        - Example: Exponentiation builder

    - ### Practice

- ## Rethinking the Training Loop
    - ### Training step
        - The higher-order function that builds a training step function takes the key elements of the training loop:
            - model
            - loss
            - optimizer
        - The actual training step function to be returned:
            - Arguments:
                - features
                - labels
            - Return:
                - Corresponding loss value
        - #### Creating the higher-order function for training step

    - ### Updating model configuration code

    - ### Updating model training code
        - [notebook](../code/Train_V1.ipynb)

    - ### What comes next?
        - For now, let us give our training loop a rest and focus on our data for a while.

- ## Building a Dataset
    - Objective:
        - Build datasets using built-in Dataset and TensorDataset classes in PyTorch.
    
    - ### The Dataset class
        - Think of it as a list of tuples:  
            - Each tuple corresponding to one point (features, label).
        - Most fundamental methods that needs to be implemented:
            - __ init __(self)
            - __ get_item__(self, index)
                - Allows dataset to be indexed.
                - Memory efficient: Loading on demand.
            - __ len__(self)
    
        - #### Building custom dataset

    - ### TensorDataset
        - Useful when dataset is nothing more than a couple of tensors.

- ## DataLoader
    - ### Introduction to DataLoader
        - Need of mini-batch gradient descent:
            - Make our work more efficient and less computationally expensive.
        - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        - shuffle=True for training set
            - Improves the performance of gradient descent.
    
    - ### Choosing the mini-batch size
        - Typical to use powers of two:
            - 16, 32, 64, 128

    - ### Changes in our implementation
        - [Data preparation code](../code/data_preparation/v1.py) updated:
            - Building a dataset of tensors.
            - Building a data loader that yields mini-batches.
    
        - Changes in [model training](../code/model_training/v2.py):
            - Added an inner loop to handle the mini-batches produced by DataLoader.
            - Only one mini-batch sent to the device as opposed to sending the whole training set.
            - Performed a train_step on a mini-batch.
    
        - Current state of development:
            - Data preparation: [V1](../code/data_preparation/v1.py)
            - Model configuration: [V1](../code/model_configuration/v1.py)
            - Model training: [V2](../code/model_training/v2.py)
    
        - [Notebook](../code/DataLoader.ipynb)

    - ### Taking more training time

- ## Mini-Batch Inner Loop and Training Split
    - ### The inner loop
        - The inner loop depends on three elements:
            - The **device** where data is being sent to.
            - A **data loader** to draw mini-batches from.
            - A **step function**, returning the corresponding loss.

        - Current state of development:
            - Data preparation: [V1](../code/data_preparation/v1.py)
            - Model configuration: [V1](../code/model_configuration/v1.py)
            - Model training: [V3](../code/model_training/v3.py)

        - [Notebook](../code/Mini-batch.ipynb)
    
    - ### Random split
        - Data preparation: [V2](../code/data_preparation/v2.py)
            - Make tensors out of the full dataset (before split).
            - Perform train-validation split in PyTorch.
            - Create data loader for the validation set.
    
- ## Evaluation
    - ### How to evaluate the model
        - How can we evaluate the model?
            - Compute the validation loss i.e. how wrong the model's predictions for unseen data.
        - Steps:
            - Use the model to compute predictions.
            - Use the loss function to compute the loss, given our predictions and the true labels.
        - Model's eval() method needs to be used.
            - Adjusts model's behavior when it has to perform some operaitons like Dropout.
            - Explanation the importance of setting this mode.
                - In dropout, the weights are randomly set to zero.
                    If this happens during eval mode, then that would lead to inconsistency of prediction even for the same input.
        - Update model configuration code: [V2](../code/model_configuration/v2.py):
            - Adding val_step function for the model and loss function.
        - Wrap validation loop with **context manager**
            - ```with torch.no_grad():```
                - Using no_grad as context manager to prevent gradient computation.
            - Referred: https://www.geeksforgeeks.org/context-manager-in-python/
                - Explained with:
                    - File management
                    - Database connection management
        - Current state of development:
            - Data preparation: [V2](../code/data_preparation/v2.py)
            - Model configuration: [V2](../code/model_configuration/v2.py)
            - Model training: [V4](../code/model_training/v4.py)
    
    - ### Plotting losses
        - [Notebook](../code/Evaluation.ipynb)
    
    - ### Fixing training step function
        - Naive approach: Storing losses in a list.

- ## TensorBoard
    - ### TensorBoard introduction
        - TensorBoard: A very useful visualization tool from Tensorflow (PyTorch's competing framework)
        - Can be used with PyTorch as well
    
    - ### Running TensorBoard inside a notebook
        - Load TensorBoard's [extension for Jupyter](https://ipython.readthedocs.io/en/stable/config/extensions/index.html):
            ```
            %load_ext tensorboard
            ```
        - Run Tensorboard using the newly available magic:
            ```
            %tensorboard --logdir runs
            ```
        - [Official guide](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks)

    - ### Testing TensorBoard
        - [Notebook](../code/Tensorboard.ipynb)

- ## SummaryWriter
    - ### Overview of Summary Writer
        - Creation of SummaryWriter
            ```
            writer = SummaryWriter('runs/test)
            ```
        - Naming convention for the folder discussed.

    - ### SummaryWriter methods
        - The SummaryWriter class implements several methods to allow us to send information to TensorBoard.
        - Also allows two other methods for effectively writing data to disk:
            - flush
            - close
        - #### The add_graph method
            ```
            writer.add_graph(model)
            ```
            - This would throw error as we need to send some inputs together with model.
            ```
            dummy_x, dummy_y = next(iter(train_loader))
            writer.add_graph(model, dummy_x.to(device))
            ```
            - [Notebook](../code/Tensorboard_graphs.ipynb)
        - #### The add_scalars method
            - What about sending the loss values to TensorBoard?
                - Use *add_scalars* method to send multiple scalar values at once.
            - Requires following three arguments:
                - main_tag
                    - Parent name of the tag or the "group tag"
                - tag_scalar_dict
                    - Dictionary containing the *key: value* pairs for the scalars we want to keep track of.
                    - In our case, training and validation losses.
                - global_step
                    - Step value or the index associated with the values sent in the dictionary.
                    - In our case, its epoch as losses are computed for each epoch.
            ```
            writer.add_scalars(
                main_tag='loss',
                tag_scalar_dict={'training': loss,
                                 'validation': val_loss},
                global_step=epoch
            )
            ```
            - [Notebook](../code/Tensorboard_scalars.ipynb)
                - This shows losses for only the final epoch.
                - To ensure that we display losses for all the epochs, we need to incorporate the scalars into our model configuration and model training codes.

            - Current state of development post last update of model configuration and training parts:
                - Data preparation: [V2](../code/data_preparation/v2.py)
                - Model configuration: [V3](../code/model_configuration/v3.py)
                - Model training: [V5](../code/model_training/v5.py)

            - [Notebook](../code/Tensorboard_training.ipynb)
                - This shows losses over sequence of epochs.
