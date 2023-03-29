# What you'll learn from this course
1. [Overview](#overview)

## Overview
- ### A. What is machine learning?
    - Most machine learning tasks generalize to one of the following two learning types:
        - **Supervised learning**
            - Using labeled data to train a model.
        - **Unsupervised learning**
            - Using *unlabeled* data to allow a model to learn relationships between data observations and pick on underlying patterns.

- ### B. ML vs AI vs data science
    - Machine Learning:
        - Subset of Artificial Intelligence (AI).
            - One of the main techniques used to create artificial intelligence.
            - Other non-ML techniques (e.g. alpha-beta pruning, rule based systems) are also widely used in AI.
        - Overlaps heavily with data science.

- ### C. 7 steps of the machine learning process
    1. Data Collection
    2. Data Processing and Preparation
    3. Feature Engineering
    4. Model Selection
        - Based on the dataset, choose which model architecture to use.
    5. Model Training and Data Pipeline
        - Create a continuous stream of batched data observations to efficiently train the model.
    6. Model Validation
    7. Model Persistence
        - Post training and validating the model's performance: 
            - Properly save the model weights and
            - Possibly push the model to production.
        - This means setting up a process with which new users can easily use your pre-trained model to make predictions.

- ### D. What this course will provide
    - Take a raw dataset and process it for a given task
        - Dealing with missing data and outliers.
        - Normalizing and transforming features.
        - Figuring out most relevant features for the task.
        - Picking out the best combination of features to use.
    - Pick the correct model architecture to use based on the data
    - Code a machine learning model and train it on processed data
        - Validate the model's performance on held-out data and understand tenchniques to improve model's performance.
