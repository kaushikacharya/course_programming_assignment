# Data Processing with scikit-learn
1. [Introduction](#introduction)
2. [Standardizing data](#standardizing-data)
3. [Data Range](#data-range)
4. [Robust Scaling](#robust-scaling)
5. [Normalizing Data](#normalizing-data)
6. [Data Imputation](#data-imputation)
7. [PCA](#pca)
8. [Labeled Data](#labeled-data)

## Introduction
- ### A. ML engineering vs. data science
    - Importing scikit-learn library in Python:
        ```
        import sklearn
        ```

## Standardizing data
- ### A. Standard data format
    - Standard format:
        - Mean = 0
        - Standard deviation = 1
    - Process of converting data into this format:
        - data standardization

- ### B. Numpy and scikit-learn
    - Module:
        - ```sklearn.preprocessing```
        - ```scale``` function:
            - Applies data standardization to a given axis of a Numpy array.
    - Usually the data is standardized independently across each feature of the data array.

- ### Time to Code!

## Data Range
- ### Chapter Goals
    - Learn how to compress data values to a specified range.

- ### A. Range scaling
    - Two step process:
        - First compute proportion of the value wrt min ($d_{min}$) and max ($d_{max}$) of the data.
            - $x_{prop} = (x - d_{min})/(d_{max} - d_{min})$
        - Use the proportion of the value to scale to the specified range: $[r_{min}, r_{max}]$
            - $x_{scale} = x_{prop}*(r_{max} - r_{min}) + r_{min}$

- ### B. Range compression in scikit-learn
    - [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) transformer

- ### Time to Code!

## Robust Scaling
- ### Chapter Goals
    - Learn how to scale data without being affected by outliers.

- ### A. Data outliers
    - [Interquartile Range (IQR)](https://en.wikipedia.org/wiki/Interquartile_range)

- ### B. Robust scaling with scikit-learn
    - [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)

- ### Time to Code!

## Normalizing Data
- ### Chapter Goals
    - Learn how to apply L2 normalization to data.

- ### L2 normalization
    - In certain cases, we want to scale the individual dsta observations (i.e. rows).
        - Usecase: Fo clustering data we need to apply L2 normalization to each row, in order to calculate cosine similarity scores.
    - Divide each value in the row by the row's L2 norm.
    - Transformer module in Scikit-Learn:
        - [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer)

- ### Time to Code!

## Data Imputation
- ### Chapter Goals
    - Learn different methods for imputing data

- ### A. Data imputation methods
    - In real life, we often have to deal with data that contains missing values.
    - [Data imputation wiki](https://en.wikipedia.org/wiki/Imputation_(statistics))
    - Scikit-Learn's [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer)
        - Univariate imputer for completing missing values with simple strategies.
        - Four data imputation methods:
            - Using the mean value (default)
            - Using the median value
            - Using the most frequent value
            - Filling in missing values with a constant

- ### B. Other imputation methods
    - Advanced imputation methods:
        - K-Nearest Neighbors
            - Filling in missing values based on similarity scores from the kNN algorithm.
        - [MICE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)
            - Applying multiple chained imputations, assuming the missing values are randomly distributed across observations.

## PCA
- ### Chapter Goals
    - Learn about principal component analysis and why it's used.

- ### A. Dimensionality reduction
    - PCA extracts the *principal components* of the dataset, which are an uncorrelated set of [latent variables](https://en.wikipedia.org/wiki/Latent_variable) that encompass most of the information from the original dataset.

- ### B. PCA in scikit-learn
    - Scikit-Learn's [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) module

- ### Time to Code!

## Labeled Data
- ### Chapter Goals
    - Learn about labeled datasets.
    - Separate principle component data by class label.

- ### A. Class labels
    - Dataset:
        - [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset
        - ```from sklearn.datasets import load_breast_cancer```

- ### Time to Code!
