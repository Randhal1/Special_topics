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
    

# This function computes the loss for the time-dependent Schrödinger equation (TDSE)
def loss_fun(train_4d, model, boundary_conditions, initial_conditions):
    """
    train_4d: Tensor of shape (N, 4) containing [x, y, z, t].
    model: Neural network predicting [Re(Ψ), Im(Ψ)].
    hbar, mass: Physical constants.
    V: Potential function (or precomputed tensor).
    lambda_bc, lambda_ic: Loss weights for BCs/ICs.
    """
    # Unpack the boundary conditions and the initial conditions. 
    L = initial_conditions['L']
    hbar = initial_conditions['hbar']
    mass = initial_conditions['mass']
    V = initial_conditions['V']
    lambda_ic = initial_conditions['lambda_ic']
    psi_0yzt = initial_conditions['psi_0yzt']
    psi_Lyzt = initial_conditions['psi_Lyzt']
    psi_x0zt = initial_conditions['psi_x0zt']
    psi_xLzt = initial_conditions['psi_xLzt']
    psi_xy0t = initial_conditions['psi_xy0t']
    psi_xyLt = initial_conditions['psi_xyLt']

    lambda_bc = boundary_conditions['lambda_bc']
    u0 = initial_conditions['u0']
    v0 = initial_conditions['v0']

    # Split inputs into spatial and time components
    x = train_4d[:, 0:1].requires_grad_(True)
    y = train_4d[:, 1:2].requires_grad_(True)
    z = train_4d[:, 2:3].requires_grad_(True)
    t = train_4d[:, 3:4].requires_grad_(True)
    
    # Forward pass: Predict u (real) and v (imaginary)
    output = model(torch.cat([x, y, z, t], dim=1))
    u = output[:, 0:1]  # Real part of Ψ
    v = output[:, 1:2]  # Imaginary part of Ψ

    # Time derivatives -----------------------------------------------------------------
    du_dt = torch.autograd.grad(u, t, 
                              grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    dv_dt = torch.autograd.grad(v, t,
                              grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]

    # Spatial derivatives (Laplacian) --------------------------------------------------
    # First derivatives for u
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Second derivatives for u
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    laplacian_u = u_xx + u_yy + u_zz

    # First derivatives for v
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # Second derivatives for v
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    laplacian_v = v_xx + v_yy + v_zz

    # Potential terms ------------------------------------------------------------------
    V_term_u = V * u  # Real potential term
    V_term_v = V * v  # Imaginary potential term

    # TDSE Residuals (corrected signs) ------------------------------------------------
    residual_u = (hbar**2 / (2 * mass)) * laplacian_u - V_term_u - hbar * dv_dt
    residual_v = (hbar**2 / (2 * mass)) * laplacian_v - V_term_v + hbar * du_dt
    
    # PDE loss (mean squared residuals)
    pde_loss = torch.mean(residual_u**2) + torch.mean(residual_v**2)

    # Boundary conditions (Ψ = 0 at spatial boundaries) -------------------------------
    # The variable bc_mask is a boolean mask that identifies the boundary points in the domain. 
    bc_mask = (x == 0) | (x == L) | (y == 0) | (y == L) | (z == 0) | (z == L)
    bc_loss = torch.mean(u[bc_mask]**2) + torch.mean(v[bc_mask]**2)

    # Initial condition (Ψ(t=0) = Ψ_0) ------------------------------------------------
    ic_mask = (t == 0)
    ic_loss = torch.mean((u[ic_mask] - u0)**2) + torch.mean((v[ic_mask] - v0)**2)

    # Total loss ----------------------------------------------------------------------
    total_loss = pde_loss + lambda_bc * bc_loss + lambda_ic * ic_loss
    
    return total_loss, pde_loss.item(), bc_loss.item(), ic_loss.item()


# Create a function to train the model
def train_model(model, params_set, hparams_set):
    
    # Unpack the parameters
    L, Number_of_points = params_set.values()
    input_dim, out_dim, width, depth, activation, initialization, eta, l2_lambda, num_epochs = hparams_set.values()
        

    # This Neuronal Nework use a 4D input, so we need to create a 4D tensor
    # The input is a tensor of shape (N, 4) where N is the number of points in the domain
    x_train = torch.linspace(0, L, Number_of_points).reshape(-1, 1).requires_grad_(True)

    # Define the optimizer 
    # Adam optimizer with learning rate eta and weight decay l2_lambda
    # Note: weight decay is used for L2 regularization
    # Note: Adam optimizer uses a default L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=l2_lambda)
    
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