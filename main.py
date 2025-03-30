import time
import matplotlib.pyplot
import numpy as np


class DNN:
    def __init(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        input_layer = sizes[0]
        hidden_layer_1 = sizes[1]
        hidden_layer_2 = sizes[2]
        output_layer = sizes[3]

        self.params = {
            # 128x784
            'W1': np.random.randn(hidden_layer_1, input_layer) * np.sqrt(1./hidden_layer_1),

            # 64x128
            'W2': np.random.randn(hidden_layer_2, hidden_layer_1) * np.sqrt(1./hidden_layer_2),

            # 10x64
            'W3': np.random.randn(output_layer, hidden_layer_2) * np.sqrt(1./output_layer)
        }

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.xp(-x)+1)**2)
        return 1/(1+np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x-x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1-exps / np.sum(exps, axis=0))
        return exps/np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train  # 784x1

        # input_layer to hidden_layer_1
        params['Z1'] = np.dot(params['W1'], params['A0'])  # 128x1
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden_layer_1 to hidden_layer_2
        params['Z2'] = np.dot(params['W2'], params['A1'])  # 128x1
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden_layer_2 to output_layer
        params['Z3'] = np.dot(params['W3'], params['A2'])  # 128x1
        params['A3'] = self.softmax(params['Z3'])

        return params['Z3']

    def backward_pass(self):
        pass

    def compute_accuracy(self):
        pass


train_file = open("/home/suito/Documents/Projet/CNN_MNIST/train.csv")
train_list = train_file.readlines()
train_file.close

test_file = open("/home/suito/Documents/Projet/CNN_MNIST/test.csv")
test_list = test_file.readlines()
test_file.close

dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, lr=0.001)
