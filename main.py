from train import *

# Set random seed for reproducibility
np.random.seed(42)

root_path = os.path.join(os.getcwd(),"data")
# Load dataset
train_images = load_images(os.path.join(root_path,"train-images.idx3-ubyte"))
train_labels = load_labels(os.path.join(root_path,"train-labels.idx1-ubyte"))
test_images = load_images(os.path.join(root_path,"t10k-images.idx3-ubyte"))
test_labels = load_labels(os.path.join(root_path,"t10k-labels.idx1-ubyte"))

# Preprocess data
train_images = flatten_and_normalize(train_images)
test_images = flatten_and_normalize(test_images)
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)
train_images, val_images = split_data(train_images)
train_labels, val_labels = split_data(train_labels)
    
model_path = os.path.join(os.getcwd(),"saved_models")
plot_path = os.path.join(os.getcwd(),"saved_plots")

num_neurons = 32
num_epochs = 900
num_layers = 2
balance_schedule = 100
layers_structure = [784] + [num_neurons for _ in range(num_layers+1)] + [10]
activation_functions = [ReLU() for _ in range(num_layers+2)]

lambda_reg = 0.01
regularization = True
decay = 0
ps = [0,1,2,3,4]

# Train model without balancing
balance_type = 'None'

for i, p in enumerate(ps):
  best_model_params_nodecay, lapd = train_model(layers_structure, activation_functions, train_images, train_labels, val_images, val_labels, balance_type=None, num_epochs=num_epochs, balance_schedule = balance_schedule, p = p)
  # Save best model parameters
  with open(os.path.join(model_path, f'best_model_params_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl'), 'wb') as f:
      pickle.dump(best_model_params_nodecay, f)
  # Save loss-accuracy plot data
  lapd_file_path = os.path.join(plot_path, f'lapd_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl')
  with open(lapd_file_path, 'wb') as f:
      pickle.dump(lapd, f)

# Train model Layer balancing
balance_type = 'Layer'

for i, p in enumerate(ps):
  best_model_params_nodecay, lapd = train_model(layers_structure, activation_functions, train_images, train_labels, val_images, val_labels, balance_type=None, num_epochs=num_epochs, balance_schedule = balance_schedule, p = p)
  # Save best model parameters
  with open(os.path.join(model_path, f'best_model_params_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl'), 'wb') as f:
      pickle.dump(best_model_params_nodecay, f)
  # Save loss-accuracy plot data
  lapd_file_path = os.path.join(plot_path, f'lapd_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl')
  with open(lapd_file_path, 'wb') as f:
      pickle.dump(lapd, f)

# Train model Inividual balancing
balance_type = 'Individual'

for i, p in enumerate(ps):
  best_model_params_nodecay, lapd = train_model(layers_structure, activation_functions, train_images, train_labels, val_images, val_labels, balance_type=None, num_epochs=num_epochs, balance_schedule = balance_schedule, p = p)
  # Save best model parameters
  with open(os.path.join(model_path, f'best_model_params_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl'), 'wb') as f:
      pickle.dump(best_model_params_nodecay, f)
  # Save loss-accuracy plot data
  lapd_file_path = os.path.join(plot_path, f'lapd_{num_neurons}x{num_layers}w{balance_type}_L{p}.pkl')
  with open(lapd_file_path, 'wb') as f:
      pickle.dump(lapd, f)