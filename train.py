from utils import *
from model import *
from optimizer import *
from loss import *

def train_model(layers_structure, activation_functions, train_images, train_labels, val_images, val_labels, num_epochs=900, balance_type=None, include_bias = False, balance_schedule = 100, decay = 0, p = 0, show = True):
    balance_schedule = balance_schedule
    np.random.seed(42)
    layers = create_model(layers_structure, activation_functions)
    loss_function = SoftmaxCrossEntropy()
    optimizer = NBAdamOptimizer(decay = decay)
    train_acc_history, train_loss_history = [], []
    val_acc_history, val_loss_history = [], []
    best_val_acc = 0.0
    best_model_params = None
    lambda_reg = 0.01

    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = 0.0, 0.0
        regularization_grads = regularization_gradients(layers, lambda_reg, p)
        for batch in range(100):
            batch_images = train_images[480*batch:480*(batch+1)]
            batch_labels = train_labels[480*batch:480*(batch+1)]
            predictions = forward_pass(layers, batch_images)

            reg_loss = lp_regularization(layers, p, lambda_reg)
            loss = loss_function.forward(predictions, batch_labels, reg_loss)
            epoch_loss += loss
            if (epoch+1) % balance_schedule == 0:
              backward_pass(batch_labels, epoch + 1, layers, loss_function, optimizer, balance_type, include_bias, regularization_grads)
            else:
              backward_pass(batch_labels, epoch + 1, layers, loss_function, optimizer, None, include_bias, regularization_grads)

            epoch_acc += compute_accuracy(loss_function.softmax, batch_labels)

        val_predictions = forward_pass(layers, val_images)
        val_loss = loss_function.forward(val_predictions, val_labels)
        val_acc = compute_accuracy(loss_function.softmax, val_labels)

        train_acc_history.append(epoch_acc / 100)
        train_loss_history.append(epoch_loss / 100)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = [(layer.weights.copy(), layer.biases.copy()) for layer in layers]
        if epoch % 50 == 0:
            print(f'Epoch {epoch} - Train Accuracy: {epoch_acc / 100:.2f}%, Train Loss: {epoch_loss / 100:.4f}, Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}')
    
    if show == True:
        # Plotting results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
        ax1.plot(range(num_epochs), train_loss_history, label='Train Loss')
        ax1.plot(range(num_epochs), val_loss_history, label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax2.plot(range(num_epochs), train_acc_history, label='Train Accuracy')
        ax2.plot(range(num_epochs), val_acc_history, label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return best_model_params, (train_loss_history, val_loss_history, train_acc_history, val_acc_history)