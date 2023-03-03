A Simple Regression Problem
===========================

- Spoilers
  --------
  - ### What to expect from this chapter
    - Briefly review the steps of gradient descent (optional).
    - Use gradient descent to implement a linear regression in Numpy.
    - Create tensors in PyTorch (finally!).
    - Understand the difference between CPU and GPU tensors.
    - Understand PyTorch’s main feature, **autograd**, to perform automatic differentiation.
    - Visualize the dynamic computation graph.
    - Create a loss function.
    - Define an **optimizer**.
    - Implement our own model class.
    - Implement **nested** and **sequential** models using PyTorch’s layers.
    - Organize code into three parts:
        - Data preparation
        - Model configuration
        - Model training

  - ### Imports

  - ### Jupyter Notebook

- Reviewing the Steps of Gradient Descent
  ---------------------------------------
  - This lesson is a review for previous chapter.

  - ### Simple Linear Regression

  - ### Data Generation
    - Synthetic data generation
    - Splitting data
    
  - ### Gradient Descent

- Linear Regression in Numpy
  --------------------------

  - ### Implementing linear regression using Numpy
  - ### Implementing linear regression using Scikit-Learn

- PyTorch Tensors
  ---------------

  - ### Diving into tensors

  - ### Tensor
    - Naming convention:
      - Scalar: zero dimension (or dimensionless)
      - Vector: One dimension
      - Matrix: Two dimensions
      - Tensor: Three or more dimensions

    - To keep things simple, it is commonplace to call vectors and matrices tensors as well.

    - Functions:
      - tensor(): Can create a scalar as well as a tensor.
      - shape:
        - tensor.size()
        - tensor.shape
      - Reshaping tensor:
        - Methods:
          - view():
            - Returns a tensor that shares the underlying data with the original tensor.
            - **Does not** creates a new, independent tensor!
            - Example shows how changing value in view() created matrix also makes corresponding change in the original tensor.
          - reshape():
            - Weird behavior: May or may not create a copy!
          - clone()
            - Preferred method for copying tensor which should be followed up by detach().
              - detach() removes the tensor from the computation graph.

  - ### Practice

- Loading Data, Devices and CUDA
  ------------------------------

  - ### Conversions between Numpy and PyTorch
    - The "as_tensor" method
      - Preserves the type of the array.
      - Both as_tensor() and from_numpy() return tensor that shares the underlying data with the original numpy array.
      - as_tensor() vs torch.tensor():
        - torch.tensor() makes a copy of the data instead of sharing the underlying data with the numpy array.
    - The numpy method
      - Transforms PyTorch tensor back to a numpy array.

  - ### GPU Tensors
    - CPU tensors:
      - Data in the tensor is stored in the computer's main memory.
      - CPU handles the operation performed on it.
    - GPU tensors:
      - Tensors store their data in the graphics card's memory.
      - Operations on top of them are performed by the GPU.
    - CPU vs GPU:
      - https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/
    - CUDA:
      - Compute Unified Device Architecture
      - PyTorch supports the use of the GPUs for model training using CUDA.
    - Availability of GPU:
      - torch.cuda.is_available()
      - torch.cuda.device_count()
      - torch.cuda.get_device_name(0) # default GPU: 0

  - ### Turning tensor into GPU tensor
    - torch.as_tensor(x_train).to(device)
      - Sends a tensor to the specified device.

  - ### Putting it all together
    - How to tell if the tensor is a CPU tensor or a GPU tensor?
      - Python's in-built function: type() returns class 'torch.Tensor' for both CPU and GPU tensor.
      - PyTorch's type function: (usage: <x_tensor>.type()):
        - CPU: torch.FloatTensor
        - GPU: torch.cuda.FloatTensor
    - Turn GPU tensor back into Numpy
      - Numpy can not handle GPU tensors.
      - Steps:
        - Convert GPU tensors into CPU tensors using cpu().
        - Convert into numpy using numpy()
      - Command: x_tensor.cpu().numpy()

  - ### Practice

- Creating Parameters
  -------------------

  - ### Tensors requiring gradients
    - What distinguishes between these two tensors?
      - (a) Tensor used for training data
      - (b) Tensor used as a trainable parameter/weight

    - Tensor (b) requires the computation of gradients, so that their values can be updated.
    - Argument: requires_grad=True

  - ### First attempt
    - Works well for CPU only.

  - ### Second attempt
    - Succeeds sending tensor to GPU but gradients are lost since there is no more requires_grad=True.

  - ### Third attempt
    - In PyTorch method ending with underscore makes chnages in-place i.e., modify the underlying variable.
    - We first send our tensors to the device and then use the requires_grad_(). This approach works fine but it requires a lot of work though.

  - ### Final attempt
    - Recommended approach:
      - Assign tensors to a device at the moment of their creation.

  - ### What if GPU isn't present?
    - PyTorch generates different sequence of numbers in different devices (CPU and GPU).

- Quiz
  ----
  - If CUDA is not properly configured, PyTorch will not be able to use the power of a graphics card's GPU, even if the hardware is available.

- Autograd
  --------

  - ### Introduction to autograd
    - Autograd: PyTorch's automatic differentiation package.

  - ### The backward method
    - Compute gradients for all (requiring gradient) tensors.
    - loss.backward()
      - backward() needs to be invoked by the loss variable.
    - #### Tensors handled by backward
      - Tensors going to be handled by the backward() method applied to the loss:
        - b and w: Obvious as requires_grad=True has been set for these variables.
        - yhat, error:
          - yhat: Since both b and w are used for computing yhat.
          - error: error is computed from yhat.
      - Pattern:
        - If a tensor requiring gradient is used to compute another tensor, the latter also requires gradient.
        - Dynamic computation graph tracks these dependencies.

  - ### The grad method
    - The gradients are accumulated.
    - [Jupyter notebook](../code/Autograd.ipynb)

    - #### Gradient accumulation
      - Why does PyTorch accumulate gradients by default?
        - To circumvent hardware limitations.
          - Mini-batch may be too big to fit in a GPU's memory.
          - Mini-batch can be split into sub batches.
            Compute the gradients for theese sub batches.
            Accumulate them to achieve the same result of computing the gradients on the full mini-batch.
            - ?? Does the parameter update happens only after complete processing of the mini-batches?

  - ### The zero_ method
    - Every time we use the gradients to update the parameters, we need to zero the gradients afterwards.

- ## Updating Parameters
  - ### All attempts summed up
  - ### First attempt
  - ### Second attempt
  - ### Third attempt
    - #### The no_grad method
      - **Understand in depth**:
        - Why we need to put the parameter updation under no_grad to avoid disrupting dynamic computation graph.
          - To keep the update out of the gradient computation.
  - ### Summary
    - Another usecase of no_grad will be discussed in the chapter: rethinking the training loop.

- ## Dynamic Computation Graphs
  - ### Introduction to dynamic computation graph
    - Package: [PyTorchViz](https://github.com/szagoruyko/pytorchviz)
    - make_dot(variable) method: Visualize a graph associated with a given Python variable involved in the gradient computation.

  - ### Plotting graphs of tensors
    - **Observation**: make_dot() behaves differently depending on whether it is run in jupyter notebook or in a script.
      - Jupyter Notebook:
      ```
      make_dot()
      ```

        - This plots the graph.
      - Script:
          - make_dot() returns Digraph. It doesn't plot the graph.
          - Refer: [make_dot](https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py)
          - Plot is done by [render function](https://graphviz.readthedocs.io/en/latest/api.html#graphviz.Digraph.render). 
      ```
      dot = make_dot()
      dot.render(outfile=<outfile_name.png>)
      ```
      - One can replace filename extension with other permitted extensions.
      - **Explanation**:
        - The difference in the behaviors are mentioned in issue [#5](https://github.com/szagoruyko/pytorchviz/issues/5) and [#3](https://github.com/szagoruyko/pytorchviz/issues/3#issuecomment-373873219).
        - [graphviz's manual](https://graphviz.readthedocs.io/en/stable/manual.html#jupyter-notebooks) mentions the reasoning for the rendering of graph in jupyter notebook.

    - Computation graph plot only shows gradient computing tensors are its dependencies.
      - *Blue box*: Tensors whose gradients are to be computed.
      - *Gray box*: Either involving gradient-computing tensor operations or their dependencies.
      - *Green box*: Tensor that is used as a starting point for the computation of gradients.

    - #### Plotting without gradients
      - Modification of above graph due to assigning of variable b as no gradient variable.

    - #### Complex dynamic computation graphs

  - ### Jupyter notebook
    - [Computation Graph notebook](../code/Computation_Graph.ipynb)

- ## Optimizer
  - ### Intorduction to optimizers
    - Different optimizers use different mechanics for updating the parameter, but they all achieve the same goal through (literally) different paths.

  - ### The step and zero_grad methods

- ## Loss
  - ### Introduction to loss functions
    - [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
    - nn.MSELoss is a [higher-order function](https://www.geeksforgeeks.org/higher-order-functions-in-python/).

  - ### Using the created loss function

  - ### Converting loss tensor to Numpy array
    - Unlike data tensors, the loss tensor is actually computing gradients.
      - To use *numpy()*, we need to *[detach()](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)* the tensor from the computation graph first.
        ```
        loss.detach().cpu().numpy()
        ```
      - *detach()*: Returns a new tensor, detached from the current graph.

- ## Quiz
  - Ans 10: Accumulating gradients is a serious problem since we only need the gradients corresponding to the current loss to perform the parameter update.

- ## Model
  - ### Introduction to models
    - model class inherits from [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    - Fundamental methods a model class needs to implement:
      - __ init __(self):
        - Include super().__ init__() first to execute __ init __ (self) of parent class (nn.Module)
      - forward(self, x):
        - Performs actual computation i.e. outputs a prediction given the input x.
    - Weird advice:
      - Whole model (model(x)) should be called instead of foward pass.
      - Reason:
        - Call to whole model involves extra steps, namely, handling forward and backward (?? hooks).
    - Hooks:
      - A very useful mechanism that allows retrieving intermediate values in deeper models.

  - ### The Parameter and parameters methods
    - Parameters in model class are defined using the Parameter() class.
      - This lets PyTorch consider these are parameters of the model class.
    - Advantage:
      - To retrieve an iterator over all model's parameters, including parameters of nested models.
  
  - ### The state_dict method
    - A Python dictionary that maps each attribute/parameter to its corresponding tensor.
    - Only learnable parameters are included.
      - Purpose: To keep track of parameters that are going to be updated by the optimizer.
    - Another usecase:
      - Checkpointing a model.
  
  - ### Device
    - Model needs to be on the same device where the data is.
  
  - ### Forward pass
    - Avoid model.forward(x). Instead call model(x).
    - Otherwise, model's hooks will not work (if present).
  
  - ### The train method
    - model.train():
      - Only purpose is to set the model to training mode.
    - Certain mechanisms like Dropout, have distinct behaviors during training and evaluation phases.

- ## Model Types
  - ### Nested models
    - [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)
    - parameters() vs state_dict()
      - state_dict() outputs parameter values together with their names.
  
  - ### Sequential models
    - Convention:
      - Call any internal model a layer.
    
  - ### Practice

- ## Layers
  - ### Introduction to Layers
    - A Linear model can be seen as a layer in a neural network.
    - Example neural network:
      - Netowrk architecture:
        - Input nodes: 3
        - Hidden nodes: 5
        - Output node: 1
        ```
        model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 1)).to(device)
        ```
  - ### Naming layers
    - If sequential model does not have attribute names, state_dict() uses numeric prefixes.
      - *0.weight, 0.bias, 1.weight, 1.bias* and so on.
    - We can use model's *add_module()* method to be able to name the layers.
  
  - ### Types of layers
    - Mentioned a few different layers that can be used in PyTorch.
      - Convolution layers
      - Pooling layers
      - Padding layers
      - Non-linear activations
      - Normalization layers
      - Recurrent layers
      - Transformer layers
      - Linear layers
      - Dropout layers
      - Sparse layers (embeddings)
      - Vision layers
      - DataParallel layers (multi-GPU)
      - Flatten layer

- ## Putting it All Together
  - ### Linear regression using PyTorch complete steps
    - Organize our code into three fundamental parts:
      - Data preparation (not data generation)
      - Model configuration
      - Model training

  - ### Data preparation
    - Next chapter: Use **Dataset** and **DataLoader** classes.
    - [IPython magic commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
      - Commands utilized:
        - %%writefile
        - %run
          - -i option: Works exactly as if code is copied from the files to the cell and executed.
    
  - ### Model configuration
    - Model configurations shown till now:
      - Defining parameters b and w.
      - Wrapping parameters using the Module class.
      - Using layers in a Sequential model.

    - Following elements included in the model configuration:
      - model
      - loss function (chosen according to the model)
      - optimizer
    
  - ### Model training
    - Training loops over the gradient descent steps:
      - Step 1: Compute the model's predictions.
      - Step 2: Compute the loss.
      - Step 3: Compute the gradients.
      - Step 4: Update the parameters.

    - Random initialization step:
      - As we are not manually creating parameters anymore, the initialization is handled inside each layer during model creation.
  
    - [Code](../code/model_training/v0.py)

  - ### Jupyter notebook
    - [Notebook](../code/Putting_together_part1.ipynb)
  
  - ### Effect of using different optimizers, loss or models

- ## Recap
  - ### General Overview
    - Covered in this 3rd chapter:
      - Implementing a linear regression in Numpy using gradient descent.
      - Creating tensors in PyTorch, sending them to a device, and making parameters out of them.
      - Understanding PyTorch's main feature, autograd, to perform automatic differentiation, using its associated properties and methods like backward, grad, zero_ and no_grad.
      - Visualizing the Dynamic Computation Graph associated with a sequence of operations.
      - Creating an optimizer to simultaneously update multiple parameters, using its step and zero_grad methods.
      - Creating a loss function by using PyTorch's higher-order function.
      - Understanding PyTorch's Module class and creating your own models:
        - Implementing __ init __ and forward methods.
        - Making use of built-in parameters and state_dict methods.
      - Transforming the original Numpy implementation into a PyTorch one using the elements above.
      - Realizing the importance of including model.train() inside the training loop.
      - Implementing nested and sequential models using PyTorch's layers.
      - Putting it all together into neatly organized code divided into three distinct parts:
        - data preparation
        - model configuration
        - model training

  - ### Jupyter notebook
    - [Notebook](../code/Chapter03.ipynb)
      - Covers the code for the entire Chapter #3.

- ## Quiz

- ## Challenge 2 - A Simple Regression Problem
  - ### Challenge
  - ### Jupyter notebook
    - [Notebook](../code/Challenges02_question.ipynb)
