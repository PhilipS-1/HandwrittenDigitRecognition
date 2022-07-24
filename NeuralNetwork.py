import numpy as np

class LinearLayer:
    def __init__(self, n_input, n_output):
        self.weights = np.random.normal(loc=0,
                                        scale=0.01,
                                        size=(n_input + 1, n_output))
        self.last_input = None

    def forward(self, x_input):
        self.last_input = np.column_stack((x_input, np.ones(x_input.shape[0])))
        return np.dot(self.last_input, self.weights)
    
    def update_weights(self, error_gradient, learning_rate):
        weight_grad = np.dot(np.transpose(self.last_input), error_gradient)
        self.weights -= learning_rate * weight_grad

class Perceptron:
    def __init__(self, n_input, n_output, activation_function=None):
        self.linear_layer = LinearLayer(n_input, n_output)
        if activation_function:
            self.activation_function = activation_function()
        else:
            self.activation_function = None

    def forward(self, x_input):
        result = self.linear_layer.forward(x_input)
        if self.activation_function:
            return self.activation_function.forward(result)
        return result

    def classify(self, x_input):
        result = self.forward(x_input)
        return np.array(np.argmax(result, 1))

    def update_weights(self, error_gradient, learning_rate):
        if self.activation_function:
            error_gradient = self.activation_function.backward(error_gradient)  
        self.linear_layer.update_weights(error_gradient, learning_rate)


class NNLayer:
    
    def __init__(self):
        pass
    
    def forward(self, x_input):
        raise NotImplementedError()

    def __call__(self, x_input):
        return self.forward(x_input)
    
    def backward(self, top_gradient):
        raise NotImplementedError()

    def apply_update(self, learning_rate, momentum):
         pass

class SigmoidLayer(NNLayer):

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    
    def __init__(self):
        self.sig_results = None 

    def forward(self, x_input):
        result = self.sigmoid(x_input)
        self.sig_results = result
        return result

    def backward(self, top_gradient):
        sig = self.sig_results
        sig_dy = sig * (1 - sig)
        return top_gradient * sig_dy

class EuclideanLoss:
    
    def __init__(self):
        self.y_label = None
        self.y_pred = None

    def __call__(self, y_pred, y_label):
        return self.forward(y_pred, y_label)
    

    def forward(self, y_pred, y_label):
        self.y_label = y_label
        self.y_pred = y_pred
        return np.sum(np.mean(np.square(y_label - y_pred), axis=1), axis=0)

    def gradient(self):
        return 2 / self.y_label.shape[1] * (self.y_pred - self.y_label)
