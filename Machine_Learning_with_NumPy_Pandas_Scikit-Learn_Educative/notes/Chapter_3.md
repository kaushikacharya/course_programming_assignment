# Data Analysis with pandas
1. [Introduction](#introduction)
2. [Series](#series)
3. [DataFrame](#dataframe)
4. [Combining](#combining)
5. [Indexing](#indexing)
6. [File I/O](#file-io)
7. [Grouping](#grouping)
8. [Features](#features)
9. [Filtering](#filtering)
10. [Sorting](#sorting)
11. [Metrics](#metrics)
12. [Plotting](#plotting)
13. [To Numpy](#to-numpy)
14. Quiz

## Introduction
- In this chapter, we'll use pandas to analyze Major League Baseball (MLB) data.
- Dataset:
    - Contains statistics for every player, manager and team in MLB history.
    - [Download location](https://www.seanlahman.com/baseball-archive/statistics/)

- ### A. Data analysis
    - Data analysis allows us:
        - To understand the dataset
        - Find potential outlier values
        - Figure out which features of the dataset are most important to our application.

- ### B. pandas
    - [Pandas website](https://pandas.pydata.org/pandas-docs/stable/)

- ### C. Matplotlib and pyplot
    - [Matplotlib website](https://matplotlib.org/index.html)

## Series
- ### Chapter Goals
    - Learn about the pandas Series object and its basic utilities.

- ### A. 1-D data
    - For 1-D data, we use the [pandas.Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) objects.

- ### B. Index
    - Each individual index element is referred to as a label.
    - The values in the ```index``` list can be any hashable type (e.g. integer, float, string).

- ### C. Dictionary input

- ### Time to Code!

## DataFrame
- ### Chapter Goals
    - Learn about the pandas DataFrame object and its basic utilities.

- ### A. 2-D data
    - Since tabular data contains rows and columns, it is 2-D.
    - When we use a Python dictionary for initialization, the DataFrame takes the dictionary's keys as its column labels.

- ### B. Upcasting
    - Upcasting occurs on a per-column basis.

- ### C. Appending rows
    - [append](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html) function
        - pandas documentation recommends concat() as append has been deprecated now.
    
- ### D. Dropping data
    - [drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html) function
        - Returns the modified DataFrame but doesn't actuaally change the original.
    - Two ways to drop rows/columns from a DataFrame:
        - Using ```labels``` keyword argument alongside ```axis``` keyword argument (default=0)
        - Directly use ```index``` or ```columns``` keyword arguments to specify the labels of the rows or columns directly.
            - ```axis``` keyword argument not required.
    - Errata:
        ```
        df.drop(index='r2', columns='c2')
        ```
        - Observation: Drops only the column without dropping the row.
        - Chapter mentions
            - Note that when using ```labels``` and ```axis```, we can't drop both rows and columns from the DataFrame.
            - IMHO, a wrong combination of keyword arguments are mentioned in the above statement. This is different from the one used in the command.

- ### Time to Code!

## Combining
- ### Chapter Goals
    - Understand the methods used to combine DataFrame objects.

- ### A.
    - [pandas.concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html)
        - Concatenates multiple DataFrames along either rows or columns.
    - Works similar to [concatenation in NumPy](./Chapter_2.md#aggregation).

- ### B. Merging
    - [pandas.merge](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html)
    - KA: Additional info
        - As per [docs](https://pandas.pydata.org/docs/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging):
            - ```merge``` is also available as a ```DataFrame``` instance method ```merge()```, with the calling ```DataFrame``` being implicitly considered the left object in the join.

- ### Time to Code!

## Indexing
- ### Chapter Goals
    - Learn how to index a DataFrame to retrieve rows and columns.

- ### A. Direct indexing
    - We can treat DataFrame as a dictionary of Series objects, where each column represents a Series.
    - Retrieving single row based on its label throws KeyError.
        - Reason: DatFrame treats the label as a column label.

- ### B. Other indexing
    - Properties:
        - [loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
            - Uses row labels instead of integer indexes.
            - Can perform column indexing along with row indexing.
            - Can set new values in a DataFrame for specific rows and columns.
        - [iloc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html)
            - Access rows based on integer index.

- ### Time to Code!

## File I/O
- ### Chapter Goals
    - Learn how to handle file input/output using pandas.

- ### A. Reading data
    - [File formats for pandas](http://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)

    - ### CSV
        - ```index_col```: Keyword argument to specify which column we want to use as the row labels.

    - ### Excel
        - Similar to CSV in its usage of rows and columns.
        - However an Excel workbook can contain multiple spreadsheets.

    - ### JSON
        - ```pandas.read_json```
            - Treats each outer key of the JSON data as a column label and each inner key as a row label.
            - ```outer='index'``` reverses the above labels.

- ### B. Writing to files
    - ### CSV
        - ```to_csv```
    
    - ### Excel
        ```
        with pd.ExcelWriter('data.xlsx') as writer:
            mlb_df1.to_excel(writer, index=False, sheet_name='NYY')
            mlb_df2.to_excel(writer, index=False, sheet_name='BOS')
        ```

    - JSON
        - ```to_json```

## Grouping
- ### Chapter Goals
    - Learn how to group DataFrames by columns.

- ### A. Grouping by column
    - [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) function
    - DataFrame groups' functions:
        - ```sum```
        - ```mean```
        - ```filter```

- ### B. Multiple columns
    - To specify grouping by multiple columns, we can use a list of column labels.

- ### Time to Code!

## Features
- ### Chapter Goals
    - Understand the difference between quantitative and categorical features.
    - Learn methods to manipulate features and add them to a DataFrame.

- ### A. Quantitative vs. categorical
    - Columns of a DataFrame are often referred as the *features* of the dataset that it represents.

- ### B. Quantitative features
    - Two of the most important functions:
        - [sum](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html)
        - [mean](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html)

- ### C. Weighted features
    - [multiply](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.multiply.html)
        - Takes in a required argument:
            - A constant
            - Or a list of weights
        - ```axis``` parameter:
            - Unlike ```sum``` and ```mean```, default axis for ```multiply``` is the column axis.

- ### Time to Code!

## Filtering
- ### Chapter Goals
    - Understand how to filter a DataFrame based on filter conditions.

- ### A. Filter conditions
    ```
    cruzne02 = df['playerID'] == 'cruzne02'
    ```

- ### B. Filters from functions
    - Apart from relation operations, pandas provides various functions for creaitng specific filter conditions.
    - For columns with string values, we can use:
        - ```str.startswith```
        - ```str.endswith```
        - ```str.contains```
    - [isin](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html)
    - Missing value in Series or DataFrame is represented by ```NaN```.
        - Equivalent to ```numpy.nan```.
    - Similar to Numpy, we cannot use a relation operation to create a filter condition for ```Nan``` values.
        - Instead use:
            - [isna](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isna.html)
            - [notna](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.notna.html)

- ### C. Feature filtering
    - Filter a DataFrame's rows based on filter conditions.

- ### Time to Code!

## Sorting
- ### Chapter Goals
    - Learn how to sort a DataFrame by its features.

- ### A. Sorting by feature
    - [sort_values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html) function

- ### Time to Code!

## Metrics
- ### Chapter Goals
    - Understand the common metrics used to summarize numeric data

- ### A. Numeric metrics
    - [describe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) function
        - Returns a summary of metrics for each of the DataFrame's numeric features.

- ### B. Categorical features
    - [pandas.Series.value_counts](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) function
        - To obtain the frequency counts for each category in a column feature.
    - [pandas.Series.unique](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html) function

- ### Time to Code!

## Plotting
- ### Chapter Goals
    - Learn how to plot DataFrames using the pyplot API.

- ### A. Basics
    ```
    import matplotlib.pyplot as plt

    df.plot(kind='line',x='yearID',y='HR')
    plt.show()
    ```

- ### B. Other plots
    - [plot](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot) function

- ### C. Multiple features
    - Plot multiple features on the same graph.
        ```
        ax = plt.gca()

        df.plot(kind='line',x='yearID',y='H',ax=ax)
        df.plot(kind='line',x='yearID',y='BB', color='red', ax=ax)
        plt.show()
        ```

## To NumPy
- ### Chapter Goals
    - Learn how to convert a DataFrame to a NumPy matrix.

- ### A. Machine learning
    - Most machine learning frameworks (e.g. TensorFlow) work directly with Numpy data.
    - The NumPy data used as input to machine learning models must solely contain quantitative values.
    - So even the categorical features of a DataFrame, such as gender and birthplace, must be converted to quantitative values.

- ### B. Indicator features
    - Convert each categorical feature into a set of indicator features for each of its categories.
    - The indicator feature for a specific category represents whether or not a given data sample belongs to that category.
        - Indicator feature:
            - 1: The row has that particular category.
            - 0: The row does not has that particular category.

- ### C. Converting to indicators
    - [get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)
        - Converts each categorical feature of a DataFrame to indicator features.

- ### D. Converting to NumPy
    - [values](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values) property
        - IMHO, this is a property and not a function as mentioned in the course.

- ### Time to Code!
    - [dropna](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) function
