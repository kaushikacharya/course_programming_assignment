# Going Classy
1. [Spoliers](#spoliers)
2. [Going Classy](#going-classy-1)
3. [Functions](#functions)
4. [Different Method Types](#different-method-types)
5. [Training Methods](#training-methods)
6. [Saving and Loading Methods](#saving-and-loading-methods)
7. [Finalising the Methods needed for going Classy](#finalising-the-methods-needed-for-going-classy)
8. [Classy Pipeline](#classy-pipeline)
9. [Model Training and Predictions](#model-training-and-predictions)
10. [Checkpointing](#checkpointing)
11. [Putting It All Together](#putting-it-all-together)
12. [Recap](#recap)
13. [Quiz](#quiz)
14. [Challenge 4 - Going Classy](#challenge-4---going-classy)
15. [Solution Review - Going Classy](#solution-review---going-classy)

- ## Spoliers
    - ### What to expect from this chapter
        - Define a class to handle model training.
        - Implement the **constructor** method.
        - Understand the difference between **public**, **protected** and **private** methods of a class.
        - Integrate the code we have developed so far into our class.
        - Instantiate our class, and use it to run a **classy** pipeline.
    
    - ### Imports
        - For this chapter, we'll need the following imports:
            ```
            import numpy as np
            import datetime

            import torch
            import torch.optim as optim
            import torch.nn as nn
            import torch.functional as F
            from torch.utils.data import DataLoader, TensorDataset, random_split
            from torch.utils.tensorboard import SummaryWriter

            import matplotlib.pyplot as plt
            %matplotlib inline
            plt.style.use('fivethirtyeight')
            ```
    - ### Jupyter notebook
        - A Jupyter notebook containing the entire code will be available to you at the end of the chapter.

- ## Going Classy
    - ### Building classes for model training
        - So far, the ```%%writefile``` magic heled us to organize the code into three distinct parts:
            - data preparation
            - model configuration
            - model training
        - At the end of previous chapter, we bumped into some of its limitations:
            - e.g. being unable to choose a different number of epochs without editing the model training code.

    - ### The class
        - A simple empty class definition:
            ```
            class StepByStep(object):
	            pass
            ```
            - Either we do not specify a parent class or we inherit it from the fundamental ```object``` class.
    
    - ### The constructor
        - The constructor defines the parts (attributes) that make up the class:
            - **Arguments** provided by the user.
            - **Placeholders** for other objects that are not available at the moment of creation (pretty much like delayed arguments).
            - **Variables** we may want to keep track of.
            - **Functions** that are dynamically built using some of the arguments and higher-order functions.

        - #### Arguments
            - Learning: Assignment of device:
                - a) Automatically check if there is a GPU available and fall back to CPU if there is not.
                - b) Give the user a chance to use a different device.

                ```
                class StepByStep(object):
                    def __init__(self, model, loss_fn, optimizer):
                        ...
                        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        # Let's send the model to the specified device right away
                        self.model.to(self.device)
                    def to(self, device):
                        # This method allows the user to specify a different device
                        # It sets the corresponding attribute (to be used later in
                        # the mini-batches) and sends the model to the device
                        self.device = device
                        self.model.to(self.device)   
                ```
        - #### Placeholders
            ```
            class StepByStep(object):
                    def __init__(self, model, loss_fn, optimizer):
                        ...
                        # These attributes are defined here, but since they are
                        # not available at the moment of creation, we keep them None
                        self.train_loader = None
                        self.val_loader = None
                        self.writer = None
            ```

            - The validsation data loader is not required (although it is recommended), and the summary writer is definitely optional.
            - The class should, therefore, implement methods to allow the user to inform those at a later time.

            ```
            def set_loaders(self, train_loader, val_loader=None):
                # This method allows the user to define which train_loader 
                # (and val_loader, optionally) to use
                # Both loaders are then assigned to attributes of the class
                # So they can be referred to later
                self.train_loader = train_loader
                self.val_loader = val_loader

            def set_tensorboard(self, name, folder='runs'):
                # This method allows the user to create a SummaryWriter to 
                # interface with TensorBoard
                suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                self.writer = SummaryWriter('{}/{}_{}'.format(
                    folder, name, suffix
                ))
            ```
        - #### Variables
            - Typical examples:
                - Number of epochs
                - Tranining loss
                - Validation loss
            
                ```
                class StepByStep(object):
                        def __init__(self, model, loss_fn, optimizer):
                            ...
                            # These attributes are going to be computed internally
                            self.losses = []
                            self.val_losses = []
                            self.total_epochs = 0
                ```

            - Best practice:
                - Define all attributes of a class in the constructor method.
                - Otherwise, as classes grow more complex, it may lead to problems.

- ## Functions
    - ### Creating function attributes
        - Sometimes it is useful to create attributes that are functions, which will be called somewhere else inside the class.
        - In our case, we can create both ```train_step``` and ```val_step``` by using the higher-order functions we defined in the [previous chapter](./Chapter_4.md).

- ## Different Method Types
    - ### Different types of methods
        - #### Public, protected and private methods
            - Some programming languages like Java have three kinds of methods:
                - public
                    - Can be called by the user.
                - protected
                    - Should not be called by the user.
                    - Supposed to be called internally or by the child class.
                        - The child class can call a protected method from its parent class.
                - private
                    - Supposed to be called exclusively internally.
                    - They should be invisible even to a child class.
            - Java:
                - Above rules are strictly enforced.
            - Python:
                - Takes a more relaxed approach: all methods are public.
                - But we can suggest the appropriate usage by prefixing the method with
                    - a single underscore (for protected methods)
                    - a double underscore (for private methods)
                - This way, the user is aware of the programmer's intention.
            
    - ### What is ```setattr```?
        - *Warning*: Should not use in regular code.
            - Using ```setattr``` to build a class by appending methods to it incrementally serves educational purposes only.
            - Using ```setattr``` is a hack.
        - The ```setattr``` function sets the value of the specified attribute of a given object.
        - As methods are also attributes, we can use this function to "attach" to an existing class and all its existing instances in one go.
        - My thought on the example code that is mentioned to show that ```setattr``` modifies class instances:
            - The code flow should be:
            ```
            class Dog(object):
                def __init__(self, name):
                    self.name = name

            rex = Dog('Rex')
            print(rex.name)

            def bark(self):
                print('{} barks: "Woof!"'.format(self.name))

            setattr(Dog, 'bark', bark)
            rex.bark()
            ```

            Whereas in the lesson,
            ```
            setattr(Dog, 'bark', bark)
            ```
            is executed before creating the ```Dog``` instance
            ```
            rex = Dog('Rex')
            ```

- ## Training Methods
    - ### Updating min-batch
        - Case: If the user decides not to provide a validation loader:
            - It will retain its initial ```None``` value from the constructor method.
            - We do not have a corresponding loss to compute, and it returns ```None``` instead.
        - [PyTorch's guidelines](https://pytorch.org/docs/stable/notes/randomness.html) on reproducibility:
            - Steps to limit the number of sources of nondeterministic behavior:
                - Controlling sources of randomness
                - Avoid using nondeterministic algorithms for some operations.

            ```
            def set_seed(self, seed=42):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False    
                torch.manual_seed(seed)
                np.random.seed(seed)
                
            setattr(StepByStep, 'set_seed', set_seed)
            ```

    - ### Updating the training loop

- ## Saving and Loading Methods
    - ### Saving and loading
        - N.B. The model is set to training mode after loading the checkpoint.
    
    - ### Making predictions

- ## Finalising the Methods needed for going Classy
    - ### Visualization methods
        - ```add_graph```
            - Use training loader to fetch a single mini-batch.
            - Use TensorBoard to build the model graph.
    
    - ### Full code of StepByStep

- ## Classy Pipeline
    - ### Pipeline steps
        - In the [previous chapter](./Chapter_4.md) our pipeline was composed of three steps:
            - data preparation [V2](../code/data_preparation/v2.py)
            - model configuration: [V3](../code/model_configuration/v3.py)
            - model training: [V5](../code/model_training/v5.py)
        
        - The last step, model training, has already been integrated into our ```StepByStep``` class.
    
    - ### Data preparation
        - As per the author, we can keep the data preparation exactly the way it was.
        - IMHO (In my humble opinion), we can parameterize:
            - Train, validation split ratio.
            - batch size
    
    - ### Model configuration
        - Some of its code has already been integrated into our class:
            - Both the ```train_step``` and ```val_step``` functions
            - ```SummaryWriter```
            - Adding the model graph

        - Now we keep only the elements needed to pass as arguments to our ```StepByStep``` class:
            - model
            - loss function
            - optimizer
        
        - N.B. We do not send the model to the device anymore since this will be handled by our class constructor.
    
    - ### Jupyter notebook
        - [Notebook](../code/Classy.ipynb)

- ## Model Training and Predictions
    - ### Starting steps
        - Start by instantiating the ```StepByStep``` class with the corresponding arguments.
            ```
            sbs = StepByStep(model, loss_fn, optimizer)
            ```
        - Next, set its loaders using the appropriately named function ```set_loaders```.
            ```
            sbs.set_loaders(train_loader, val_loader)
            ```
        - Set up an interface with TensorBoard.
            ```
            sbs.set_tensorboard('classy')
            ```
        - ```model``` attribute of the ```sbs``` object is the same object as the ```model``` variable creted in the model configuration.
            ```
            print(sbs.model == model)
            ```
            returns true.
    
    - ### Training the model

    - ### Making predictions
        - Let us make up some data points for our feature ```x```, and shape them as a single column matrix.
            ```
            new_data = np.array([.5, .3, .7]).reshape(-1, 1)
            ```
            - Explanation of -1 in reshape is provided in DuttaA's answer in [this Stackoverflow thread](https://stackoverflow.com/questions/36384760/transforming-a-row-vector-into-a-column-vector-in-numpy).

    - ### Jupyter notebook
        - [Notebook](../code/Training_Predictions.ipynb)

- ## Checkpointing
    - ### Saving checkpoints
        - To checkpoint the model to resume training later, we can use the ```save_checkpoint``` method, which handles the state dictionaries for us and saves them to a file.
    
    - ### Resuming training

    - ### Jupyter notebook
        - [Notebook](../code/Checkpointing.ipynb)

- ## Putting It All Together
    - ### Overall view of the result
        - #### Data preparation V2
        - #### Model configuration V4
        - #### Model training
    
    - ### Jupyter noptebook
        - [Notebook](../code/Putting_together_part3.ipynb)

- ## Recap
    - ### General overview
        - We have covered:
            - Defining our ```StepByStep``` class.
            - Understanding the purpose of the constructor (__ init__) method.
            - Defining the arguments of the constructor method.
            - Defining class' attributes to store arguments, placeholders and variables that we need to keep track of.
            - Defining functions as attributes using higher-order functions and the class' attributes to build functions that perform training and validation steps.
            - Understand the difference between public, protected and private methods; and Python's "relaxed" approach.
            - Creating methods to set data loaders and TensorBoard integration.
            - (re)implementing training methods: ```_mini_batch``` and ```train```.
            - Implementing saving and loading methods: ```save_checkpoint``` and ```load_checkpoint```.
            - Implementing a method for making predictions that takes care of all boilerplate code regarding Numpy to PyTorch conversion and back.
            - Implementing methods to plot losses, and add the model's  graph to TensorBoard.
            - Instantiating our ```StepByStep``` class and running a classy pipeline: configuring the model, loading the data, training the model, making predictions, checkpointing, and resuming training. The whole nine yards!

    - ### Jupyter notebook
        - [Notebook](../code/Chapter05.ipynb)

- ## Quiz
    - In Python, the interpreter rewrites the name of the attributes having double underscore prefixes.
        -  This is done so that the variable does not get overridden and collides with variables in subclasses.
        -  Hence, therefore we get an AttributeError when we try to access __< variable name > since according to the interpreter, the object does not has a __< variable name > attribute.

- ## Challenge 4 - Going Classy
    - ### Challenge

    - ### Jupyter notebook
        - [Question notebook](../code/Challenges04_question.ipynb)
        - [Answer notebook](../code/Challenges04_answer.ipynb)

- ## Solution Review - Going Classy
    - ### Solution
        - [Notebook](../code/Challenges04.ipynb)
