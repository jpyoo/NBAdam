import torch
import torch.nn as nn
import numpy as np
import pickle

class LayerDense:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1. / output_size)
        self.biases = np.random.randn(output_size, 1)
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.z = (np.dot(self.weights, input.T) + self.biases).T
        self.activation.forward(self.z)
        self.a = self.activation.output

    def backward(self, dvalue):
        self.activation.backward(dvalue)
        self.dweights = np.dot(self.input.T, self.activation.dinput).T
        self.dbiases = np.sum(self.activation.dinput, axis=0, keepdims=True).T
        self.dinputs = np.dot(self.activation.dinput, self.weights)

    def balance(self, incoming_weights, incoming_biases, b_type = 'Layer', include_bias = True):
        iw = np.hstack((incoming_weights, incoming_biases)) if include_bias else incoming_weights
        ow = np.hstack((self.weights, self.biases)) if include_bias else self.weights
        incoming_norm = np.linalg.norm(iw)
        outgoing_norm = np.linalg.norm(ow)
        lambda_scale = (outgoing_norm / incoming_norm)
        incoming_weights *= lambda_scale
        if include_bias: incoming_biases *= lambda_scale
        return incoming_weights, incoming_biases

    def balance_neurons(self, incoming_weights, incoming_biases, b_type = 'Individual', include_bias = True):
        for i in range(self.weights.shape[0]):
            iw = np.append(incoming_weights[i, :], incoming_biases[i]) if include_bias else incoming_weights[i, :]
            ow = self.weights[:, i]
            incoming_norm = np.linalg.norm(iw)
            outgoing_norm = np.linalg.norm(ow)
            lambda_scale = (outgoing_norm / incoming_norm) if incoming_norm != 0 else 1
            incoming_weights[i, :] *= lambda_scale
            if include_bias: incoming_biases[i] *= lambda_scale
        return incoming_weights, incoming_biases
    
    def load(self, loaded_w, loaded_b):
      self.weights = loaded_w
      self.biases = loaded_b

class ReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalue):
        self.dinput = dvalue.copy()
        self.dinput[self.input <= 0] = 0

class Tanh:
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, dvalues):
        self.dinput = dvalues * (1 - np.square(self.output))
        return self.dinput

def forward_pass(layers, X):
    for i in range(len(layers)):
        layers[i].forward(X)
        X = layers[i].a
    return X

def backward_pass(y, iteration, model_layers, loss_function, optimizer, balance_type=None, include_bias=True, dreg=[]):
    loss_function.backward(loss_function.softmax, y, dreg)
    dvalue = loss_function.dinput
    for i in reversed(range(len(model_layers))):
        model_layers[i].backward(dvalue)
        dvalue = model_layers[i].dinputs
        if dreg:
          model_layers[i].dweights += dreg[i]
    optimizer.update(model_layers, iteration, balance_type, include_bias)

# Load the best model parameters
def load_model(file_path):
    with open(file_path, 'rb') as f:
        saved_params = pickle.load(f)
    return saved_params

def set_model_params(model_layers, saved_params):
    for layer, params in zip(model_layers, saved_params):
        layer.weights, layer.biases = params

def create_model(layers_structure, activation_functions):
    torch.manual_seed(0)
    layers = []
    for i in range(len(layers_structure) - 1):
        layers.append(LayerDense(layers_structure[i], layers_structure[i+1], activation_functions[i]))
    return layers
