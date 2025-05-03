import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd


# Define the MLP class
# This class is a multi-layer perceptron (MLP) neural network.
# It is used to approximate the solution of a partial differential equation (PDE).
# The MLP consists of multiple layers, each with a specified number of neurons and activation functions.
class MLP(nn.Module):
    def __init__(self, hparams_set):
        """ Initialize the MLP model.
        Args:
            input_dim (int): Dimension of the input layer.
            out_dim (int): Dimension of the output layer.
            width (int): Number of neurons in each hidden layer.
            depth (int): Number of hidden layers.
            activation (str): Activation function to use ('tanh', 'sin', 'relu', 'sigmoid').
            initialization (str): Weight initialization method ('uniform' or 'normal').
        """

        input_dim, out_dim, width, depth, activation, initialization = hparams_set.values()

        super(MLP, self).__init__()

        # Want to ensure at least one hidden layer
        assert depth > 1
        self.depth = depth
        
        # Selecting the activation for hidden layers
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        # Create a list of layers and include the first hidden layer
        MLP_list = [nn.Linear(input_dim, width)]

        # Remaining hidden layers
        for _ in range(depth - 2):
            MLP_list.append(nn.Linear(width, width))

        # Output layer 
        MLP_list.append(nn.Linear(width, out_dim))

        # Adding list of layers as modules
        self.model = nn.ModuleList(MLP_list)

        # Weights initialization
        # Solution for item 5.8.1
        def init_uniform(layer):
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1, 1)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -1, 1)

        def init_normal(layer):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, mean=0.0, std=1.0)

        if initialization == 'uniform':
            self.model.apply(init_uniform)
        
        elif initialization == 'normal':
            self.model.apply(init_normal)

    # Defining forward mode of MLP model
    # Since i < depth-1, we apply this condition, activation function will run in all layers except the last one (output layer)
    def forward(self, x):

        for i, layer in enumerate(self.model):
            # Apply activation only to hidden layers
            if i < self.depth-1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
 
        return x

    def countpar(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def init_normal(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=1.0)
    
    def clear(self):
        self.model.apply(self.init_normal)
    

# Create custom dataset Class 
class CustomDataset(Dataset):
    def __init__(self, samples):
        """ Initialize CustomDataset with paired samples.

        Args:
            samples (list of tuples): A ;ist of (x, y) pairs
            representing the dataset samples. """
        # I've decided to update to a more modern version
        self.samples = torch.tensor(samples, dtype=torch.float32)

    def __len__(self):
        """
        Returns the lenght of the dataset, i.e., the number of samples."""

        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns the sample pairs corresponding to the given list of indices.
        Args: indices (list): A list of indices to retrieve samples for.
        Returns: list: A list of (x, y) pairs corresponding to the specified indices. 
        """
        # Split the tensor into input (x) and target (y)
        x = self.samples[idx, 0]  # First column is input
        y = self.samples[idx, 1]  # Second column is target
        return x, y



# This function computes the loss for the time-dependent Schrödinger equation (TDSE)
def loss_fun(xyt_train, model, lambda_bc, lambda_ic, V=0, hbar=1, mass=1):
    """
    xyt_train: Tensor of shape (N, 4) containing [x, y, z, t].
    model: Neural network predicting [Re(Ψ), Im(Ψ)].
    hbar, mass: Physical constants.
    V: Potential function (or precomputed tensor).
    lambda_bc, lambda_ic: Loss weights for BCs/ICs.
    """
    # Split inputs into spatial and time components
    x = xyt_train[:, 0:1].requires_grad_(True)
    y = xyt_train[:, 1:2].requires_grad_(True)
    z = xyt_train[:, 2:3].requires_grad_(True)
    t = xyt_train[:, 3:4].requires_grad_(True)
    
    # Forward pass: Predict real and imaginary parts of Ψ
    psi = model(torch.cat([x, y, z, t], dim=1))
    psi_real = psi[:, 0:1]
    psi_imag = psi[:, 1:2]
    
    # Time derivatives
    dpsi_real_dt = torch.autograd.grad(psi_real, t, torch.ones_like(psi_real), create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi_imag, t, torch.ones_like(psi_imag), create_graph=True)[0]
    
    # Spatial derivatives for Laplacian (Real part)
    psi_real_x = torch.autograd.grad(psi_real, x, torch.ones_like(psi_real), create_graph=True)[0]
    psi_real_xx = torch.autograd.grad(psi_real_x, x, torch.ones_like(psi_real_x), create_graph=True)[0]
    psi_real_y = torch.autograd.grad(psi_real, y, torch.ones_like(psi_real), create_graph=True)[0]
    psi_real_yy = torch.autograd.grad(psi_real_y, y, torch.ones_like(psi_real_y), create_graph=True)[0]
    psi_real_z = torch.autograd.grad(psi_real, z, torch.ones_like(psi_real), create_graph=True)[0]
    psi_real_zz = torch.autograd.grad(psi_real_z, z, torch.ones_like(psi_real_z), create_graph=True)[0]
    laplacian_real = psi_real_xx + psi_real_yy + psi_real_zz
    
    # Spatial derivatives for Laplacian (Imaginary part)
    psi_imag_x = torch.autograd.grad(psi_imag, x, torch.ones_like(psi_imag), create_graph=True)[0]
    psi_imag_xx = torch.autograd.grad(psi_imag_x, x, torch.ones_like(psi_imag_x), create_graph=True)[0]
    psi_imag_y = torch.autograd.grad(psi_imag, y, torch.ones_like(psi_imag), create_graph=True)[0]
    psi_imag_yy = torch.autograd.grad(psi_imag_y, y, torch.ones_like(psi_imag_y), create_graph=True)[0]
    psi_imag_z = torch.autograd.grad(psi_imag, z, torch.ones_like(psi_imag), create_graph=True)[0]
    psi_imag_zz = torch.autograd.grad(psi_imag_z, z, torch.ones_like(psi_imag_z), create_graph=True)[0]
    laplacian_imag = psi_imag_xx + psi_imag_yy + psi_imag_zz
    
    # Potential term (assuming V is precomputed or a function)
    V_term_real = V * psi_real
    V_term_imag = V * psi_imag
    
    # Compute TDSE residuals
    residual_real = (hbar**2 / (2 * mass)) * laplacian_real - V_term_real + hbar * dpsi_imag_dt
    residual_imag = (hbar**2 / (2 * mass)) * laplacian_imag - V_term_imag - hbar * dpsi_real_dt
    pde_loss = torch.mean(residual_real**2) + torch.mean(residual_imag**2)
    
    # Boundary conditions (example: Ψ=0 at spatial boundaries)
    bc_mask = (x == 0) | (x == L) | (y == 0) | (y == L) | (z == 0) | (z == L)
    bc_loss = torch.mean(psi_real[bc_mask]**2) + torch.mean(psi_imag[bc_mask]**2)
    
    # Initial condition (Ψ(t=0) = Ψ_0)
    ic_mask = (t == 0)
    ic_loss = torch.mean((psi_real[ic_mask] - psi_real_0)**2) + torch.mean((psi_imag[ic_mask] - psi_imag_0)**2)
    
    # Total loss
    total_loss = pde_loss + lambda_bc * bc_loss + lambda_ic * ic_loss
    
    return total_loss, pde_loss.item(), bc_loss.item(), ic_loss.item()


# Create a function to train the model
def train_model(model, peda_set, params_set, hparams_set):
    
    # Solve item 5.8.1 
    # Create the model input as in item 5.8.1
    x_train = torch.linspace(0, 1, 100).reshape(-1, 1).requires_grad_(True)
    
    # Unpack the parameters
    Pe_all, Da_all, lambda_b, l2_lambda, eta, num_epochs = params_set.values()
    input_dim, out_dim, width, depth, activation, initialization = hparams_set.values()

    # Chose the set of parameters Pe and Da
    Pe = Pe_all[peda_set]
    Da = Da_all[peda_set]

    # Define the optimizer 
    # Adam optimizer with learning rate eta and weight decay l2_lambda
    # Note: weight decay is used for L2 regularization
    # Note: Adam optimizer uses a default L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=l2_lambda)

    # Create a list to store the models and data
    outputs = {
        'General_info': [],
        'Info': [], 
        'location': [],
        'set': peda_set
    }

    # Create the information for the model
    about_models = f'Model with activation {activation}, initialization: {initialization}, width: {width}, depth: {depth},\n'
    about_models+= f'Pe: {Pe}, Da: {Da},\n'
    about_models+= f'lambda_b: {lambda_b}, Learning rate: {eta}, l2_lambda: {l2_lambda}\n'
    about_models+= f'Number of epochs: {num_epochs},\n'
    about_models+= f'Number of parameters: {model.countpar()}\n'
    about_models+= f'Input dimension: {input_dim}, Output dimension: {out_dim},\n'

    outputs['General_info'] = about_models
       
    about_model = f'Parameters: Pe = {Pe}, Da = {Da},\n'
    outputs['Info'].append(about_model)

    print(f"Training the model with parameters Pe = {Pe}, Da = {Da} ...\n")

    df = pd.DataFrame(columns=['epoch', 'R_int', 'R_bc'])
    
    for epoch in range(num_epochs):
        optimizer.zero_grad() # zero the gradients
        # Compute the interior and boundary losses
        loss, interior_loss, boundary_loss = loss_fun(x_train, model, Pe, Da, lambda_b)
        loss.backward() # backpropagation
        optimizer.step() # update the weights
      
        # Store the losses
        df.loc[(epoch+1)] = [(epoch+1), interior_loss, boundary_loss]

        # Print the losses every 1000 epochs
        if (epoch+1) % 5000 == 0:
            print(f"Epoch {epoch+1}: Interior Loss: {interior_loss}, Boundary Loss: {boundary_loss}, Total Loss: {loss.item()}")
            # Save the model every 5000 epochs

    # Save the model loss
    filetext = f"models/model_item1_Pe_{Pe}_Da_{Da}.csv"
    df.to_csv(filetext, index=False) 
    outputs['location'] = filetext

    # Print the final output results.
    print("Training completed.\n")
        
    return outputs


# Working on 5.8.4
# Plot the losses for each combination of Pe and Da
def plotter(model, inputs_from_train, params_set, data_model_location, info = False):
    """ Plot the losses for each combination of Pe and Da.
    Args:
        model (MLP): The trained MLP model.
        inputs_from_train (dict): Dictionary containing the outputs from the training function.
    """
    
    # Unpack the parameters
    ginfo, info, location, peda_set = inputs_from_train.values()
    Pe_all, Da_all, lambda_b, l2_lambda, eta, num_epochs = params_set.values()

    # Define a function to extract more information from the model. (Optional)
    if info:
        # Print the general information about the model
        print(ginfo)
        # Print the information about the model
        print(info)
        # Print the location of the model
        print(location)
        # Print the l2 penalty
        print(f"l2_lambda: {l2_lambda}")

    # Adjust for the parameter set 
    Pe = Pe_all[peda_set]
    Da = Da_all[peda_set]


    # Load the data from the files
    data = pd.read_csv(data_model_location, sep='\s', header=None)

    losses = pd.read_csv(location)

    boundary_losses = losses['R_bc'].to_numpy()
    interior_losses = losses['R_int'].to_numpy()

    model.eval()

    # Build x_test correctly as shape [N,1]
    x_np   = data[0].to_numpy().reshape(-1, 1)
    x_test = torch.tensor(x_np, dtype=torch.float32)

    # (If using GPU:)
    # device = next(model_item1.parameters()).device
    # x_test = x_test.to(device)

    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy().reshape(-1)

    # calculate the norm for the errors
    error = np.linalg.norm(u_pred - data[1].to_numpy(), 2) / np.linalg.norm(data[1].to_numpy(), 2)

    # Plot the losses and the predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f'solution of $u$ using PINN for Pe={Pe}, Da={Da}, $\lambda_b$ = {lambda_b}, $\eta$ = {eta}', fontsize=16)

    # Plot the solution
    ax1.plot(data[0], data[1], '--', lw = 4, label='Target', color='crimson')
    ax1.plot(data[0], u_pred, lw = 2, label='NN Solution', color='dodgerblue')
    ax1.plot(0,0, 'o', lw = 0, label='Error: {:.2e}'.format(error), color='black')
    ax1.legend()
    ax1.set_ylim([0, 1.02])
    ax1.set_xlim([0, 1])
    ax1.grid()
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$u(x)$')
    ax1.set_title('NN Solution vs Target')   


    # Plot the losses
    ax2.semilogy(interior_losses, label=f'Interior Loss, Pe={Pe}, Da={Da}')
    ax2.semilogy(boundary_losses, label=f'Boundary Loss, Pe={Pe}, Da={Da}')
    ax2.legend()
    ax2.set_ylim([1e-15, 1e5])
    ax2.set_xlim([0, num_epochs])
    ax2.grid()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Losses vs Epochs')   

    plt.show()