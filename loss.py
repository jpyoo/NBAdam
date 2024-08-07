import torch
import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, input, y_true, reg_loss=0):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.softmax = probabilities
        softmax_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        cross_entropy_loss = -np.log(np.sum(softmax_clipped * y_true, axis=1))
        self.loss = np.mean(cross_entropy_loss) + reg_loss  # Add reg_loss
        return self.loss

    def backward(self, dvalue, y_true, dreg=0):
        samples = len(dvalue)
        self.dinput = dvalue.copy()
        self.dinput[range(samples), np.argmax(y_true, axis=1)] -= 1
        self.dinput /= samples
        return self.dinput
    
def lp_regularization(model, p=0, lambda_reg=0.01):
    reg_loss = 0
    if p != 0:
        for layer in model:
            reg_loss += torch.norm(torch.tensor(layer.weights), p)**p
    return lambda_reg * reg_loss

def regularization_gradients(layers, lambda_reg, p):
    grads = []
    for layer in layers:
        if p == 2:
            grad = 2 * lambda_reg * layer.weights  # Gradient for L2
        elif p == 1:
            grad = lambda_reg * np.sign(layer.weights)  # Gradient for L1
        elif p == 0:
            grad = np.zeros_like(layer.weights) # Gradient for No regularization
        else:
            grad = lambda_reg * p * np.abs(layer.weights) ** (p-1) * np.sign(layer.weights)  # Gradient for Lp
        grads.append(grad)
    return grads