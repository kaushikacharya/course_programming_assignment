import sys
import os
import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split   # In the new version this has been moved into model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


class DigitRecognizer:
    def __init__(self, data_folder="data", train_file="train.csv", test_file="test.csv"):
        # data files
        self.data_folder = data_folder
        # train_file is being split into train, validation sets
        self.train_file = train_file
        self.test_file = test_file

        self.train_features_data = None
        self.validation_features_data = None

        # The following contains the data after dimensionality reduction
        self.train_features_reduced_data = None
        self.validation_features_reduced_data = None

        self.train_label_array = None
        self.validation_label_array = None

    def load_train_data(self):
        """ Load train_file data
            This data is later split into train,validation sets
        """
        # alternate is using numpy.loadtxt whose return type is ndarray
        # But then label column would be needed to be moved into label array
        with open(os.path.join(self.data_folder, self.train_file), "r") as fd:
            train_df = pd.read_csv(fd)
        return train_df

    def load_test_data(self):
        with open(name=os.path.join(self.data_folder, self.test_file), mode="r") as fd:
            test_df = pd.read_csv(fd)
        return test_df

    def split_train_validation(self, file_data_df, train_fraction=0.7):
        """ Split train.csv into train and validation set
            Model(s) are trained on train set and validated on validation set

            Parameters
            ----------
            file_data_df Each row has both feature and label
                         This is the dataframe loaded from train_file
        """
        label_array = np.array(file_data_df.ix[:, "label"])

        train_index_array, validation_index_array = train_test_split(range(len(file_data_df.index)),
                                                                     train_size=train_fraction)

        # store the labels
        self.train_label_array = label_array[train_index_array]
        self.validation_label_array = label_array[validation_index_array]

        # Now split the features dataframe into train,validation
        # Separate the label from the features
        features_data_df = file_data_df.drop("label", axis=1)
        # https://stackoverflow.com/questions/17682613/how-to-convert-a-pandas-dataframe-subset-of-columns-and-rows-into-a-numpy-array
        # (a) convert pandas dataframe into numpy ndarray  (b) converting pixel intensities from int to float
        self.train_features_data = features_data_df.loc[train_index_array, ].values.astype(float)
        self.validation_features_data = features_data_df.loc[validation_index_array, ].values.astype(float)

    # TODO Allow other methods for dimensionality reduction e.g. random projection
    def dimensionality_reduction(self, proportion_variance_explained_threshold=0.9):
        """ Dimensionality reduction of train, validation data
            Currently PCA is used.
        """
        assert self.train_label_array is not None, "pre-requisite: split the data using the function: split_train_validation()"
        assert self.validation_label_array is not None, "pre-requisite: split the data using the function: split_train_validation()"

        # min-max normalization
        # TODO move min-max normalization into separate function
        # TODO assert if data type is int
        self.train_features_data /= 255
        self.validation_features_data /= 255

        pca = PCA(n_components=proportion_variance_explained_threshold)
        # Fit PCA on train set
        self.train_features_reduced_data = pca.fit_transform(self.train_features_data)
        # Now use the PCA model fitted on train set to transform validation set
        self.validation_features_reduced_data = pca.transform(self.validation_features_data)
        assert self.train_features_reduced_data.shape[1] == self.validation_features_reduced_data.shape[1], \
            "Both training and validation set should be reduced to same number of dimensions after dimensionality" \
            " reduction"
        print "dimensions post PCA: {0} using proportion variance explained={1}".format(
            self.train_features_reduced_data.shape[1], proportion_variance_explained_threshold)


class BackPropagation:
    def __init__(self, train_data, validation_data, train_label_array, validation_label_array, learning_rate=0.05,
                 batch_size=500, n_epoch=5, velocity_decay_rate=0.9, delta_local_gain=0.05, min_local_weight_gain=0.1,
                 max_local_weight_gain=10):
        """ Initialize BackPropagation

            Parameters
            ----------
            train_data : numpy.ndarray, shape(n_train_samples, n_features)
            validation_data : numpy.ndarray, shape(n_validation_samples, n_features)
        """
        assert isinstance(train_data, np.ndarray), "train_data type: numpy.ndarray"
        assert isinstance(validation_data, np.ndarray), "validation_data type: numpy.ndarray"

        self.train_data = train_data
        self.validation_data = validation_data
        self.train_label_array = train_label_array
        self.validation_label_array = validation_label_array
        # hidden layers/nodes
        self.n_hidden_layers = 1
        self.n_input_nodes = train_data.shape[1]
        self.n_output_nodes = 10  # digits 0 - 9
        self.n_hidden_nodes = None

        # list of numpy ndarray; index=i represent connections between level_i and level_i+1
        # weights
        self.theta = list()
        # error derivative wrt weights
        self.error_derivative_wrt_weight = list()
        self.sign_error_derivative_wrt_weight = list()
        self.local_weight_gain = list()
        # previous weight change
        self.prev_theta_update = list()  # list of numpy ndarray

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.velocity_decay_rate = velocity_decay_rate  # used by momentum
        self.delta_local_gain = delta_local_gain  # used for individual learning rates
        self.min_local_weight_gain = min_local_weight_gain
        self.max_local_weight_gain = max_local_weight_gain
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
        """Get number of nodes for a given level
            Assumption: Same number of nodes at hidden layers.
        """
        if level == 0:
            return self.n_input_nodes
        elif level == self.n_hidden_layers+1:
            return self.n_output_nodes
        elif level > 0 & level <= self.n_hidden_layers:
            return self.n_hidden_nodes
        else:
            assert False, "levels present: {0} - {1}".format(0, self.n_hidden_layers+1)

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
            theta_level_i_to_j = np.zeros((n_nodes_lower_level, n_nodes_upper_level))
            for node_i in range(0, n_nodes_lower_level):
                for node_j in range(0, n_nodes_upper_level):
                    # TODO: what should be the range of values for weight initialization to be small
                    # current range: (0,1)
                    theta_level_i_to_j[node_i, node_j] = random.random()
            # Now append the weight matrix for the connections between nodes of level_i with nodes of level_j
            self.theta.append(theta_level_i_to_j)

    def initialize_local_weight_gain(self):
        """Initialize local weight gain for each connection as 1
        """
        for level_i in range(self.n_hidden_layers+1):
            level_j = level_i + 1
            n_nodes_lower_level = self.get_number_of_nodes(level_i)
            n_nodes_upper_level = self.get_number_of_nodes(level_j)
            local_weight_gain_i_to_j = np.ones((n_nodes_lower_level, n_nodes_upper_level))
            assert len(self.local_weight_gain) == level_i, "expected len(self.local_weight_gain) = level_i." \
                                                           " This initialization function needs to be called only once"
            self.local_weight_gain.append(local_weight_gain_i_to_j)

    def initialize_error_derivatives(self):
        """Initialize error derivatives to zero
            Before every mini-batch weight update initialize error derivatives to zeros.
        """
        for level_i in range(self.n_hidden_layers+1):
            n_nodes_lower_level = self.get_number_of_nodes(level_i)
            level_j = level_i + 1
            n_nodes_upper_level = self.get_number_of_nodes(level_j)
            error_derivative_ij = np.zeros((n_nodes_lower_level, n_nodes_upper_level))
            if len(self.error_derivative_wrt_weight) > level_i:
                self.error_derivative_wrt_weight[level_i] = error_derivative_ij
            else:
                self.error_derivative_wrt_weight.append(error_derivative_ij)

    def initialize_previous_weight_update(self):
        """Stores weight update in previous step. Required for using momentum.
        """
        for level_i in range(self.n_hidden_layers+1):
            level_j = level_i + 1
            n_nodes_lower_level = self.get_number_of_nodes(level_i)
            n_nodes_upper_level = self.get_number_of_nodes(level_j)
            weight_update_level_i_to_j = np.zeros((n_nodes_lower_level, n_nodes_upper_level))
            assert len(self.prev_theta_update) == level_i, "expected len(self.prev_weight_update) = level_i"
            self.prev_theta_update.append(weight_update_level_i_to_j)

    def compute_cross_entropy_cost(self):
        """ Cross entropy cost for a single training sample
        """
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
            for node_j in range(0, self.n_hidden_nodes):
                # extract the weight array based on the edges connecting the nodes from previous layer to the j'th node
                # of current hidden layer
                weight_array = np.zeros(self.n_input_nodes)
                for node_i in range(0, self.n_input_nodes):
                    weight_array[node_i] = self.theta[0][node_i, node_j]

                z_array[node_j] = np.dot(y_array, weight_array)

            # Now compute the y array for this hidden layer
            y_array = np.zeros(self.n_hidden_nodes)
            for node_j in range(0, self.n_hidden_nodes):
                y_array[node_j] = self.compute_sigmoid(z_array[ node_j])

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
                weight_array[j] = self.theta[self.n_hidden_layers][j, k]

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
    def update_weights_online(self):
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
                self.theta[level_j][node_j, node_k] -= self.learning_rate * yk_minus_tk * y_j

        # Update of the weights connecting the input layer to the single hidden layer
        # For multiple hidden layers, equation will be different
        level_i = 0  # input layer
        for node_j in range(0, self.n_hidden_nodes):
            y_j = self.hidden_layer_activation_array[0][node_j]
            for node_i in range(0, self.n_input_nodes):
                y_i = self.input_layer_array[node_i]
                error_derivative_ij = 0.0
                for node_k in range(0, self.n_output_nodes):
                    w_jk = self.theta[level_j][node_j, node_k]
                    yk_minus_tk = self.output_layer_activation_array[node_k] - self.output_layer_array[node_k]
                    error_derivative_ij += w_jk * yk_minus_tk * y_j * (1.0 - y_j) * y_i
                # update the weight connecting node_i to node_j using the cost error derivative
                self.theta[level_i][node_i, node_j] -= self.learning_rate * error_derivative_ij

    # TODO Handle error derivatives wrt weight for multi hidden layers
    def update_error_derivatives(self, batch_size):
        """Update error derivatives wrt weight
            For mini-batch we average error derivatives over the mini-batch size.
            This function is called online i.e. for each of the training samples of the mini-batch
            Once its called for each of the samples on the mini-batch, we update the weights
            Assumption: Single hidden layer
        """
        # Error derivatives wrt the weights connecting the top most hidden layer to the output layer
        level_k = self.n_hidden_layers + 1
        level_j = self.n_hidden_layers
        for node_k in range(0, self.n_output_nodes):
            yk_minus_tk = self.output_layer_activation_array[node_k] - self.output_layer_array[node_k]
            for node_j in range(0, self.n_hidden_nodes):
                y_j = self.hidden_layer_activation_array[self.n_hidden_layers - 1][node_j]
                self.error_derivative_wrt_weight[level_j][node_j, node_k] += yk_minus_tk * y_j / batch_size

        # Update of the weights connecting the input layer to the single hidden layer
        # For multiple hidden layers, equation will be different
        level_i = 0  # input layer
        for node_j in range(0, self.n_hidden_nodes):
            y_j = self.hidden_layer_activation_array[0][node_j]
            for node_i in range(0, self.n_input_nodes):
                y_i = self.input_layer_array[node_i]
                # sum over the error derivative on the connections from node_j to the nodes in its upper layer i.e. k
                error_derivative_ij = 0.0
                for node_k in range(0, self.n_output_nodes):
                    w_jk = self.theta[level_j][node_j, node_k]
                    yk_minus_tk = self.output_layer_activation_array[node_k] - self.output_layer_array[node_k]
                    error_derivative_ij += w_jk * yk_minus_tk * y_j * (1.0 - y_j) * y_i
                # update the weight connecting node_i to node_j using the cost error derivative
                self.error_derivative_wrt_weight[level_i][node_i, node_j] += error_derivative_ij / batch_size

    def update_local_weight_gain(self, prev_error_derivative_wrt_weight):
        """Individual learning rates implemented using local weight gain
            Based on the idea whether direction of error derivatives are same or has changed
            If same: local weight gain is increased additively.
            If different: local weight gain is decreased multiplicatively.
        """
        # case: error derivative wrt weight hasn't been computed yet.
        if len(prev_error_derivative_wrt_weight) == 0:
            return

        for level_i in range(self.n_hidden_layers+1):
            level_j = level_i + 1
            n_nodes_level_i = self.get_number_of_nodes(level_i)
            n_nodes_level_j = self.get_number_of_nodes(level_j)
            for node_j in range(n_nodes_level_j):
                for node_i in range(n_nodes_level_i):
                    sign_prev_error_derivative = np.sign(prev_error_derivative_wrt_weight[level_i][node_i, node_j])
                    sign_cur_error_derivative = np.sign(self.error_derivative_wrt_weight[level_i][node_i, node_j])

                    if sign_prev_error_derivative * sign_cur_error_derivative == 1:
                        self.local_weight_gain[level_i][node_i, node_j] += self.delta_local_gain
                        if self.max_local_weight_gain < self.local_weight_gain[level_i][node_i, node_j]:
                            self.local_weight_gain[level_i][node_i, node_j] = self.max_local_weight_gain
                    elif sign_prev_error_derivative * sign_cur_error_derivative == -1:
                        self.local_weight_gain[level_i][node_i, node_j] *= (1.0 - self.delta_local_gain)
                        if self.local_weight_gain[level_i][node_i, node_j] < self.min_local_weight_gain:
                            self.local_weight_gain[level_i][node_i, node_j] = self.min_local_weight_gain
                    else:
                        pass

    # TODO Handle weight update for multi hidden layers
    def update_weights(self):
        """Update weights for the mini-batch after error derivatives have been computed for each of the samples of the mini-batch
        """
        for level_i in range(self.n_hidden_layers+1):
            level_j = level_i + 1
            n_nodes_level_i = self.get_number_of_nodes(level_i)
            n_nodes_level_j = self.get_number_of_nodes(level_j)
            for node_j in range(n_nodes_level_j):
                for node_i in range(n_nodes_level_i):
                    weight_update = self.velocity_decay_rate * self.prev_theta_update[level_i][node_i, node_j] - \
                                    self.learning_rate * self.local_weight_gain[level_i][node_i, node_j] *\
                                    self.error_derivative_wrt_weight[level_i][node_i, node_j]
                    self.theta[level_i][node_i, node_j] += weight_update
                    # save this update for usage in next iteration of update at [level_i][node_i, node_j] connection
                    self.prev_theta_update[level_i][node_i, node_j] = weight_update

    # TODO create mini-batch randomly instead of only in sequence
    #       as shown in "mini-batch gradient descent" section of http://ruder.io/optimizing-gradient-descent/
    def train_mini_batch(self):
        # random initialization of weights
        self.initialize_weights()
        self.initialize_previous_weight_update()
        # initialize local weight gain for each connection as 1
        self.initialize_local_weight_gain()

        # TODO instead of fixed number of epochs, better to have a stopping criterion and max number of epochs
        n_mini_batch = len(self.train_data)/self.batch_size
        for epoch_i in range(self.n_epoch):
            print "epoch # ", epoch_i
            for mini_batch_i in range(n_mini_batch):
                print "mini batch # ", mini_batch_i
                # create a mini-batch using the range [train_i, train_j)
                train_i = mini_batch_i*self.batch_size
                if mini_batch_i < n_mini_batch-1:
                    train_j = (mini_batch_i+1)*self.batch_size
                else:
                    # in case train data size is not exactly divisible to batch_size, we take till the end of train_data
                    # for the last mini-batch
                    train_j = len(self.train_data)

                # error derivatives wrt weight calculated for the previous mini-batch
                prev_error_derivative_wrt_weight = self.error_derivative_wrt_weight
                # set error derivatives to zero
                self.initialize_error_derivatives()

                # iterate over each of the training samples in the mini-batch and compute error derivatives
                # we average the error derivatives and update weights using the average
                mini_batch_size = train_j - train_i
                cross_entropy_mini_batch_cost = 0.0
                while train_i < train_j:
                    # assign input layer
                    self.input_layer_array = self.train_data[train_i, ]
                    # assign output layer
                    self.output_layer_array = np.zeros(self.n_output_nodes)
                    self.output_layer_array[self.train_label_array[train_i]] = 1.0

                    # forward pass
                    for level in range(1, self.n_hidden_layers + 1):
                        self.compute_hidden_layer(level)
                    self.compute_output_layer()

                    # backward pass: update error derivatives
                    self.update_error_derivatives(mini_batch_size)

                    cross_entropy_mini_batch_cost += self.compute_cross_entropy_cost()
                    train_i += 1

                """
                # print the error derivatives
                for level_i in range(self.n_hidden_layers + 1):
                    level_j = level_i + 1
                    print [",".join([str(x) for x in self.error_derivative_wrt_weight[level_i][i, ]])
                           for i in range(self.error_derivative_wrt_weight[level_i].shape[0])]
                """

                # TODO Also mention the accuracy
                print "\t average cost for mini train batch: ", cross_entropy_mini_batch_cost/mini_batch_size
                # Now update the weight based on error derivatives of the current mini-batch
                if self.delta_local_gain != 0:
                    self.update_local_weight_gain(prev_error_derivative_wrt_weight)
                self.update_weights()

    # TODO Should evaluate accuracy on training set also.
    def evaluate(self):
        incorrect_prediction_count = 0
        for validate_i in range(len(self.validation_data)):
            # assign input layer
            self.input_layer_array = self.validation_data[validate_i]
            # forward pass
            for level in range(1, self.n_hidden_layers + 1):
                self.compute_hidden_layer(level)
            self.compute_output_layer()
            true_class = self.validation_label_array[validate_i]
            predicted_class = np.argmax(self.output_layer_activation_array)
            print "validate_i: {0} :: true class: {1} : (prob) ({2}) :: predicted class: {3} : (prob): ({4})"\
                .format(validate_i, true_class, self.output_layer_activation_array[true_class],
                        predicted_class, self.output_layer_activation_array[predicted_class])
            if true_class != predicted_class:
                incorrect_prediction_count += 1

        print "incorrect prediction: {0} %".format(incorrect_prediction_count*100/len(self.validation_data))


class CNN:
    """Based on Deep Learning with Python by Jason Brownlee (www.machinelearningmastery.com)
        Keras version used: 1.1.1
    """
    def __init__(self, train_data, validation_data, train_label_array, validation_label_array, channels_image=1, width_image=28, height_image=28):
        """Initialize Convolutional Neural Network

        Parameters
        ----------
        train_data : numpy.ndarray, shape(n_train_samples, n_features)
        validation_data : numpy.ndarray, shape(n_validation_samples, n_features)
        """
        # TODO check for data type of train/validation data. Should be of type float
        # reshape to [samples][channels][width][height]
        self.train_X = train_data.reshape(train_data.shape[0], channels_image, width_image, height_image)
        self.validation_X = validation_data.reshape(validation_data.shape[0], channels_image, width_image, height_image)
        self.train_label_matrix = np_utils.to_categorical(train_label_array)
        self.validation_label_matrix = np_utils.to_categorical(validation_label_array)
        assert self.train_label_matrix.shape[1] == 10, "classes: 0-9 expected"
        self.channels_image = channels_image
        self.width_image = width_image
        self.height_image = height_image
        # min-max normalization
        self.train_X /= 255
        self.validation_X /= 255

    def create_model(self):
        model = Sequential()
        # https://github.com/fchollet/keras/issues/2558 (Added dim_ordering="th" which represents single channel)
        model.add(Convolution2D(32, 5, 5, border_mode="valid", input_shape=(1, self.width_image, self.height_image),
                                activation="relu", dim_ordering="th"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        # compile model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit_model(self):
        model = self.create_model()
        model.fit(self.train_X, self.train_label_matrix, validation_data=(self.validation_X, self.validation_label_matrix), nb_epoch=2, batch_size=300, verbose=2)
        scores = model.evaluate(self.validation_X, self.validation_label_matrix, verbose=0)
        print "CNN model error: %.2f" % (100-scores[1]*100)


if __name__ == "__main__":
    digit_recognizer_obj = DigitRecognizer("data", "train_trial.csv")
    train_file_data_df = digit_recognizer_obj.load_train_data()
    print "row count of train file: ", len(train_file_data_df.index)
    print "columns count of data: ", len(train_file_data_df.columns)-1  # One column represents label
    digit_recognizer_obj.split_train_validation(train_file_data_df)

    method_chosen = 0
    if method_chosen == 0:
        digit_recognizer_obj.dimensionality_reduction()
        # without momentum: use velocity_decay_rate=0
        # with individual learning rates: use delta_local_gain=0
        back_propagation_obj = BackPropagation(train_data=digit_recognizer_obj.train_features_reduced_data,
                                               validation_data=digit_recognizer_obj.validation_features_reduced_data,
                                               train_label_array=digit_recognizer_obj.train_label_array,
                                               validation_label_array=digit_recognizer_obj.validation_label_array,
                                               batch_size=30, velocity_decay_rate=0
                                               )
        back_propagation_obj.train_mini_batch()
        back_propagation_obj.evaluate()
    else:
        cnn_obj = CNN(train_data=digit_recognizer_obj.train_features_data,
                      validation_data=digit_recognizer_obj.validation_features_data,
                      train_label_array=digit_recognizer_obj.train_label_array,
                      validation_label_array=digit_recognizer_obj.validation_label_array)
        cnn_obj.fit_model()

"""
TODO:
    - Split train into (a) train, (b) validation using random split with seed
    - Dimensionality reduction using PCA, random projection
    - Plot t-sne from scikit-learn
    - Create confusion matrix
    - In online weight update version, think on how many samples should we calculate the cost. Only computing cost for
        the current sample doesn't seems to be good enough.
    - Gradient checking: (a) http://cs231n.github.io/neural-networks-3/
                         (b) In Andrew Ng's programming assignment
    - config file: use it as now there's too many parameters

Current implementation:
    - single hidden layer, softmax backpropagation with cross entropy as cost function
    - no bias

Resource:
    Number of hidden nodes:
        https://www.quora.com/How-do-I-decide-the-number-of-nodes-in-a-hidden-layer-of-a-neural-network
        https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

    Normalization of input data:
        https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network
        https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks

    Weight Update in mini-batch:
        https://stats.stackexchange.com/questions/266968/how-does-minibatch-gradient-descent-update-the-weights-for-each-example-in-a-bat
        https://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent ** (nice explanation)
        Suggestion for tuning hyper-parameters: (a) learning rate  (b) batch size
        http://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

    Applying PCA to test data:
        https://stats.stackexchange.com/questions/144439/applying-pca-to-test-data-for-classification-purposes
        Advice on using the same transformation for test set which is learnt over train set:
            https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/

    Explanation of fit, fit_transform in scikit:
        https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models

    pandas:
        https://stackoverflow.com/questions/19609631/python-changing-row-index-of-pandas-data-frame
"""
