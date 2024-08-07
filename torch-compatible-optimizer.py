import numpy as np
import torch
from torch.optim import Adam
from typing import List, Optional, Tuple, Union
from torch import tensor

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class NBAdam(Adam):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, balance_type='Layer', balance_cycle=100):
        super(NBAdam, self).__init__(model.parameters(), lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, amsgrad=amsgrad)
        self.model = model
        self.balance_type = balance_type
        self.balance_cycle = balance_cycle
        self.last_balance = 0
    def step(self, closure=None, epoch=1):
        # If a closure is provided, recompute the output.
        if closure is not None:
            loss = closure()
        else:
            loss = None

        # Standard Adam update
        loss = super(NBAdam, self).step(closure)

        # Balancing every `balance_cycle` epochs
        if (epoch + 1) % self.balance_cycle == 0:
            self.last_balance = epoch
            child_list = [p for name, p in self.model.named_parameters() if 'weight' in name]
            reversed_children = list(reversed(child_list))

            for i in range(len(reversed_children) - 1):
                top = reversed_children[i]
                bottom = reversed_children[i + 1]
                self.apply_balancing(bottom, top, self.balance_type)

        return loss

    def apply_balancing(self, in_weights, out_weights, balance_type):
        if balance_type == 'Layer':
            self.balance_layers(in_weights, out_weights)
        elif balance_type == 'Individual':
            self.balance_individual_neurons(in_weights, out_weights)

    def balance_layers(self, incoming_weights, outgoing_weights):
      with torch.no_grad():
        incoming_norm = torch.linalg.vector_norm(nn.ReLU(incoming_weights.data))
        outgoing_norm = torch.linalg.vector_norm(nn.ReLU(outgoing_weights.data))
        incoming_weights.data *= (outgoing_norm / incoming_norm)

    def balance_individual_neurons(self, incoming_weights, outgoing_weights):
        incoming_weights = incoming_weights.data
        outgoing_weights = outgoing_weights.data
        for i in range(outgoing_weights.shape[1]):
            iw = incoming_weights[i, :]
            ow = outgoing_weights[:, i]
            incoming_norm = torch.linalg.vector_norm(iw)
            outgoing_norm = torch.linalg.vector_norm(ow)
            scaling_factor = (outgoing_norm / incoming_norm) if incoming_norm != 0 else 1.0
            incoming_weights[i, :] *= scaling_factor

## Example Usage:

# net = Net()
# # Define loss function and optimizer
# criterion = SoftmaxCrossEntropy()
# optimizer = NBAdam(net, lr=0.001)
# best_loss, best_acc, best_model_norm = float('inf'), 0, 0
# # Training loop
# for epoch in range(num_epochs):
#     epoch_loss, epoch_acc = 0.0, 0.0
#     for batch in range(100):
#         batch_images = torch.tensor(train_images[480*batch:480*(batch+1)], dtype = torch.float32)
#         batch_labels = torch.tensor(train_labels[480*batch:480*(batch+1)], dtype = torch.float32)
#         predictions = net(batch_images)
#         optimizer.zero_grad()
#         loss = criterion(predictions, batch_labels)
#         loss.backward()
#         epoch_acc += compute_accuracy(predictions, batch_labels)
#         epoch_loss += loss.item()
#         optimizer.step(epoch=epoch)