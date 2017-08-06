import sys
import os
import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split   # In the new version this has been moved into model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DigitRecognizer:
    def __init__(self, data_folder="data", train_file="train.csv", test_file="test.csv"):
        # data files
        self.data_folder = data_folder
        self.train_file = train_file
        self.test_file = test_file

        self.train_df = None
        self.test_df = None

        # train_file is being split into train, validation sets
        self.train_data_df = None
        self.validation_data_df = None
        self.train_label_array = None
        self.validation_label_array = None

    def load_train_data(self):
        with open(os.path.join(self.data_folder, self.train_file), "r") as fd:
            self.train_df = pd.read_csv(fd)

    def load_test_data(self):
        with open(name=os.path.join(self.data_folder, self.test_file), mode="r") as fd:
            self.test_file = pd.read_csv(fd)

    def split_train_validation(self, train_fraction=0.7):
        """ Split train.csv into train and validation set
        """
        # separate the label from the features
        data_df = self.train_df.drop("label", axis=1)
        label_array = np.array(self.train_df.ix[:, "label"])

        # Dimensionality reduction using PCA
        # TODO check how to predict pca on validation set
        # TODO normalize each of the columns
        pca = PCA(n_components=50)
        data_df = pca.fit_transform(data_df)

        train_index_array, validation_index_array = train_test_split(range(0, len(self.train_df.index)),
                                                                     train_size=train_fraction)
        # PCA fit output is of type numpy.ndarray
        if isinstance(data_df, pd.DataFrame):
            self.train_data_df = data_df.ix[train_index_array, ]
            self.validation_data_df = data_df.ix[validation_index_array, ]

            # re-index train, validation dataframes
            # https://stackoverflow.com/questions/19609631/python-changing-row-index-of-pandas-data-frame
            self.train_data_df.index = range(len(train_index_array))
            self.validation_data_df.index = range(len(validation_index_array))
        else:
            self.train_data_df = data_df[train_index_array,]
            self.validation_data_df = data_df[validation_index_array,]

        # store the labels
        self.train_label_array = label_array[train_index_array]
        self.validation_label_array = label_array[validation_index_array]


class BackPropagation:
    def __init__(self, train_df, validation_df, train_label_array, validation_label_array, learning_rate=0.05):
        # converting train/validation dataframe to ndarray
        if isinstance(train_df, pd.DataFrame):
            train_df = np.ndarray(train_df)
        if isinstance(validation_df, pd.DataFrame):
            validation_df = np.ndarray(validation_df)
        self.train_df = train_df
        self.validation_df = validation_df
        self.train_label_array = train_label_array
        self.validation_label_array = validation_label_array
        # hidden layers/nodes
        self.n_hidden_layers = 1
        # self.n_input_nodes = len(train_df.columns)
        self.n_input_nodes = train_df.shape[1]
        self.n_output_nodes = 10  # digits 0 - 9
        self.n_hidden_nodes = None
        # weights
        self.theta = dict()
        self.learning_rate = learning_rate
        # activation and z values for the levels
        # output layer
        self.output_layer_activation_array = np.zeros(self.n_output_nodes)
        self.output_layer_z_array = np.zeros(self.n_output_nodes)
        # hidden layer
        # List of arrays. Length of list = number of hidden layers
        self.hidden_layer_activation_array = list()
        self.hidden_layer_z_array = list()
        # input layer
        self.input_layer_array = None
        # output layer
        self.output_layer_array = None

        # initialize functions
        self.set_hidden_nodes()
        self.initialize_hidden_layer()

    def set_hidden_nodes(self):
        """ Compute number of nodes for hidden layer
            Method #1: average of input and output nodes
            Method #2: formula which considers number of training samples
        """
        # Method #1
        self.n_hidden_nodes = int((self.n_input_nodes + self.n_output_nodes) / 2)

    def initialize_hidden_layer(self):
        """ Initializing y and z for the hidden layers with zeros
            Assumption: There's equal number of nodes in each of the hidden layers
        """
        assert self.n_hidden_nodes is not None, "count of hidden nodes should be done by calling set_hidden_nodes()"
        for i in range(0, self.n_hidden_layers):
            self.hidden_layer_activation_array.append(np.zeros(self.n_hidden_nodes))
            self.hidden_layer_z_array.append(np.zeros(self.n_hidden_nodes))

    def get_number_of_nodes(self, level):
        if level == 0:
            return self.n_input_nodes
        elif level == 1:
            return self.n_hidden_nodes
        elif level == 2:
            return self.n_output_nodes
        else:
            assert False, "Current implementation has a single hidden layer"

    def initialize_weights(self):
        """ Randomly initialize weights
            Notation: theta[(level_i,node_i)][(level_j,node_j)] = weight
                    This represents the weight of the edge connecting node_i of level_i to node_j of level_j
                    We consider i for lower level and j for the upper level.
                    Example: i=0 for input level and j=1 for first hidden layer.
        """
        for level_i in range(0, self.n_hidden_layers+1):
            n_nodes_lower_level = self.get_number_of_nodes(level_i)
            level_j = level_i + 1
            n_nodes_upper_level = self.get_number_of_nodes(level_j)
            for node_i in range(0, n_nodes_lower_level):
                self.theta[(level_i, node_i)] = dict()
                for node_j in range(0, n_nodes_upper_level):
                    # TODO: what should be the range of values for weight initialization to be small
                    # current range: (0,1)
                    self.theta[(level_i, node_i)][(level_j, node_j)] = random.random()

    def compute_cross_entropy_cost(self):
        cross_entropy_cost = 0.0
        for k in range(len(self.output_layer_array)):
            cross_entropy_cost += self.output_layer_array[k] * np.log(self.output_layer_activation_array[k])
        cross_entropy_cost *= -1.0
        return cross_entropy_cost

    @staticmethod
    def compute_sigmoid(z):
        y = 1.0/(1.0 + np.exp(-1*z))
        return y

    def compute_hidden_layer(self, cur_level):
        """ Compute (a) weighted sum of inputs (z)  (b) activation (y) of z
            Notation: input level is considered level 0. First hidden layer is level 1 and so on.
        """
        if cur_level == 1:
            # case: previous level is input level
            # Before calling this function ensure that self.input_layer_array has been assigned current training sample
            # by using the function assign_input_layer
            y_array = self.input_layer_array

            # For each of the hidden nodes of current hidden layer compute its z
            z_array = np.zeros(self.n_hidden_nodes)
            for j in range(0, self.n_hidden_nodes):
                # extract the weight array based on the edges connecting the nodes from previous layer to the j'th node
                # of current hidden layer
                weight_array = np.zeros(self.n_input_nodes)
                for i in range(0, self.n_input_nodes):
                    weight_array[i] = self.theta[(0, i)][(cur_level, j)]

                z_array[j] = np.dot(y_array, weight_array)

            # Now compute the y array for this hidden layer
            y_array = np.zeros(self.n_hidden_nodes)
            for j in range(0, self.n_hidden_nodes):
                y_array[j] = self.compute_sigmoid(z_array[j])

            self.hidden_layer_z_array[0] = z_array
            self.hidden_layer_activation_array[0] = y_array
        else:
            # case: previous level is another hidden layer
            pass

    def compute_output_layer(self):
        """ Compute (a) weighted sum of inputs (z)  (b) activation (y) of z for the output layer
        """
        y_array = self.hidden_layer_activation_array[self.n_hidden_layers-1]

        # For each of the nodes of the output layer compute its z
        z_array = np.zeros(self.n_output_nodes)
        for k in range(0, self.n_output_nodes):
            # extract the weight array
            weight_array = np.zeros(self.n_hidden_nodes)
            for j in range(0, self.n_hidden_nodes):
                weight_array[j] = self.theta[(self.n_hidden_layers, j)][(self.n_hidden_layers+1, k)]

            z_array[k] = np.dot(y_array, weight_array)

        y_array = np.zeros(self.n_output_nodes)
        for k in range(0, self.n_output_nodes):
            y_array[k] = np.exp(z_array[k])
        # For softmax y values needs to be normalized
        sum_y_output = np.sum(y_array)
        for k in range(0, self.n_output_nodes):
            y_array[k] /= sum_y_output

        self.output_layer_z_array = z_array
        self.output_layer_activation_array = y_array

    # TODO Handle weight update for multi hidden layers
    def update_weights(self):
        """ Current implementation: Online learning
            Assumption: single hidden layer
        """
        # Update of the weights connecting the top most hidden layer to the output layer
        level_k = self.n_hidden_layers + 1
        level_j = self.n_hidden_layers
        for node_k in range(0, self.n_output_nodes):
            yk_minus_tk = self.output_layer_activation_array[node_k] - self.output_layer_array[node_k]
            for node_j in range(0, self.n_hidden_nodes):
                y_j = self.hidden_layer_activation_array[self.n_hidden_layers-1][node_j]
                self.theta[(level_j, node_j)][(level_k, node_k)] -= self.learning_rate * yk_minus_tk * y_j

        # Update of the weights connecting the input layer to the single hidden layer
        # For multiple hidden layers, equation will be different
        level_i = 0  # input layer
        for node_j in range(0, self.n_hidden_nodes):
            y_j = self.hidden_layer_activation_array[0][node_j]
            for node_i in range(0, self.n_input_nodes):
                y_i = self.input_layer_array[node_i]
                error_derivative_ij = 0.0
                for node_k in range(0, self.n_output_nodes):
                    w_jk = self.theta[(level_j, node_j)][(level_k, node_k)]
                    yk_minus_tk = self.output_layer_activation_array[node_k] - self.output_layer_array[node_k]
                    error_derivative_ij += w_jk * yk_minus_tk * y_j * (1.0 - y_j) * y_i
                # update the weight connecting node_i to node_j using the cost error derivative
                self.theta[(level_i, node_i)][(level_j, node_j)] -= self.learning_rate * error_derivative_ij

    def train_online(self):
        # random initialization of weights
        self.initialize_weights()

        for train_i in range(len(self.train_df)):
            if train_i % 10 == 9:
                print "train_i: ", train_i
            # assign input layer
            self.input_layer_array = self.train_df[train_i, ]
            # assign output layer
            self.output_layer_array = np.zeros(self.n_output_nodes)
            self.output_layer_array[self.train_label_array[train_i]] = 1.0

            # forward pass
            for level in range(1, self.n_hidden_layers+1):
                self.compute_hidden_layer(level)
            self.compute_output_layer()
            cross_entropy_cost = self.compute_cross_entropy_cost()
            print "train_i: {0} :: cost: {1}".format(train_i, cross_entropy_cost)
            # backward pass
            self.update_weights()

    def validate(self):
        for validate_i in range(len(self.validation_df)):
            # assign input layer
            self.input_layer_array = self.validation_df[validate_i]
            # forward pass
            for level in range(1, self.n_hidden_layers + 1):
                self.compute_hidden_layer(level)
            self.compute_output_layer()
            true_class = self.validation_label_array[validate_i]
            predicted_class = np.argmax(self.output_layer_activation_array)
            print "validate_i: {0} :: true class: {1} : (prob) ({2}) :: predicted class: {3} : (prob): ({4})"\
                .format(validate_i, true_class, self.output_layer_activation_array[true_class],
                        predicted_class, self.output_layer_activation_array[predicted_class])

if __name__ == "__main__":
    digit_recognizer_obj = DigitRecognizer("data", "train_trial.csv")
    digit_recognizer_obj.load_train_data()
    print "row count of train file: ", len(digit_recognizer_obj.train_df.index)
    print "columns count of data: ", len(digit_recognizer_obj.train_df.columns)-1  # One column represents label
    digit_recognizer_obj.split_train_validation()
    back_propagation_obj = BackPropagation(train_df=digit_recognizer_obj.train_data_df,
                                           validation_df=digit_recognizer_obj.validation_data_df,
                                           train_label_array=digit_recognizer_obj.train_label_array,
                                           validation_label_array=digit_recognizer_obj.validation_label_array
                                           )
    back_propagation_obj.train_online()
    back_propagation_obj.validate()

"""
TODO:
    - Split train into (a) train, (b) validation using random split with seed
    - Dimensionality reduction using PCA, random projection
    - Plot t-sne from scikit-learn

Current implementation:
    - single hidden layer, softmax backpropagation with cross entropy as cost function
    - no bias
    - online training

Resource:
    Number of hidden nodes:
        https://www.quora.com/How-do-I-decide-the-number-of-nodes-in-a-hidden-layer-of-a-neural-network
        https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

    Normalization of input data:
        https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network
        https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks

    Weight Update in mini-batch:
        https://stats.stackexchange.com/questions/266968/how-does-minibatch-gradient-descent-update-the-weights-for-each-example-in-a-bat
"""
