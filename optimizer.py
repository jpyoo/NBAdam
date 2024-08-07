import numpy as np

class NBAdamOptimizer:
    def __init__(self, learning_rate=0.00122, decay=0, epsilon=4e-7, beta_1=0.9, beta_2=0.99):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update(self, layers, iterations, balance_type = None, include_bias=True):
        lr = self.learning_rate * (1. / (1. + self.decay * iterations))
        for layer in layers:
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)
            # add balancing, get new dweights
            layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
            weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (iterations + 1))
            layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
            weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (iterations + 1))
            bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (iterations + 1))
            layer.weights -= lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases -= lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        if balance_type == 'None' or balance_type == None:
          pass
        elif balance_type == 'Layer':
          for i in reversed(range(1, len(layers)-1)):
              try:
                layers[i-1].weights, layers[i-1].biases = layers[i].balance(layers[i-1].weights, layers[i-1].biases, balance_type, include_bias)
              except Exception as e:
                print(e)
        elif balance_type == 'Individual':
          for i in reversed(range(1, len(layers)-1)):
              try:
                layers[i-1].weights, layers[i-1].biases = layers[i].balance_neurons(layers[i-1].weights, layers[i-1].biases, balance_type, include_bias)
              except Exception as e:
                print(i, e)