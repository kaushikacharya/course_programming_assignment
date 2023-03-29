# Data Manipulation with NumPy
1. [Introduction](#introduction)
2. [NumPy Arrays](#numpy-arrays)
3. [NumPy Basics](#numpy-basics)
4. [Math](#math)
5. [Random](#random)

## Introduction
- A. Data processing
- B. NumPy

## NumPy Arrays
- ### Chapter Goals
    - Learn about NumPy arrays and how to initialize them.
    - Write code to create several NumPy arrays.
- ### A. Arrays
- ### B. Copying
    - Similar to Python lists, a reference to a NumPy array doesn't create a different array.
    - To get around this, use array's inherent [copy](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html) function.

- ### C. Casting
    - [astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html) function
    - ```dtype``` property returns the type of an array.

- ### D. NaN
    - ```numpy.nan``` acts as a placeholder.
    - ```numpy.nan``` cannot take on an integer type.

- ### E. Infinity
    - ```numpy.inf``` cannot take on an integer type.

- ### Time to Code!

## NumPy Basics
- ### Chapter Goals
    - Learn some basic NumPy operations.

- ### A. Ranged data
    - [numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
    - [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
        - Unlike ```arange```, end of the range is inclusive.

- ### B. Reshaping data
    - [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
    - Special value -1 is allowed in at most one dimension of the new shape.
    - ```flatten```: Flattens an array using the inherent function.
        - Alternatively, ```numpy.reshape``` can be used.

- ### C. Transposing
    - [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
        - ```axes``` parameter explained.
    - Visual explanation of transposing in a 3D array:
        - [Alex Riley's answer](https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array) in StackOverflow thread
    
- ### D. Zeros and ones
    - [numpy.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
    - [numpy.ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)
    - Creating array with same shape as another array:
        - [numpy.zeros_like](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html)
        - [numpy.ones_like](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html)
        - ?? Aren't these functions an overkill? One can extract the shape of another array and use zeros, ones.

- ### Time to Code!

## Math
- ### Chapter Goals
    - Perform math operation in NumPy

- ### A. Arithmetic
    - One of the main purposes of NumPy is to perform multi-dimensional arithmetic.

- ### B. Non-linear functions
    - ```nnumpyp.exp```
    - ```np.exp2```
    - ```np.log```
    - ```np.log2```
    - ```np.log10```
    - [numpy.power](https://numpy.org/doc/stable/reference/generated/numpy.power.html)
        - To do regular power operation with any base.
            - 1st argument: base
            - 2nd argument: power
    - [List of mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html)
    
- ### C. Matrix multiplication
    - [numpy.matmul](https://numpy.org/doc/stable/reference/routines.math.html)
        - Input: Two vector/matrix arrays
        - Produces a dot product or matrix multiplication

- ### Time to Code!

## Random
- ### Chapter Goals
    - Learn about random operations in NumPy.
    - ```numpy.random``` submodule

- ### A. Random integers
    - [numpy.random.randint](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)

- ### B. Utility functions
    - [numpy.random.seed](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html)
    - [numpy.random.shuffle](https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html)
        - Shuffling happens in-place.
        - Shuffling multi-dimensional array only shuffles the first dimension.

- ### C. Distributions
    - Using ```numpy.random``` we can aslo draw samples from probability distributions.

- ### D. Custom sampling
    - [numpy.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)

- ### Time to Code!

## Indexing

