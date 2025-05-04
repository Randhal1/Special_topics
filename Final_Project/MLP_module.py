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


# Create a function to define the exact solution
# hbar = m = 1, L = 1
def exact_solution_example(r, t, L=1, plot=False):
    # r is a 3D tensor
    # t is a 1D tensor

    def En(nx, ny, nz):
        return np.pi**2 * (nx**2 + ny**2 + nz**2) / (2 * L**2)
    
    # Ap and Bp are the coefficients of the wave function 
    # The constraint is that the wave function must be normalized
    # Ap^2 + Bp^2 = 8/L^3
    Ap = np.sqrt(5 / L**3)
    Bp = np.sqrt(3 / L**3)

    # Take for example 
    nx, ny, nz = [1, 1, 1]

    # And 
    mx, my, mz = [2, 1, 1]

    # The wave function is a linear combination of the two states
    x = r[:, 0:1]
    y = r[:, 1:2]
    z = r[:, 2:3]
    t = t*torch.ones_like(x)

    # Contruct a function to compute the sin of multiple variables
    def msin(x, y, z):
        return torch.sin(np.pi * x/L) * torch.sin(np.pi * y/L) * torch.sin(np.pi * z/L)
    
    def mexp(nx, ny, nz, t):
        return torch.exp(-1j * En(nx, ny, nz) * t)

    # Construct the wave function
    values_psi = Ap * msin(x, y, z) * mexp(nx, ny, nz, t) + Bp * msin(2*x, y, z) * mexp(mx, my, mz, t)

    # Plot the exact solution
    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1, ax2 = fig.subplots(1, 2)

        ax1.plot(values_psi.real, color = 'dodgerblue', label='Real part')
        ax1.legend(loc = 'upper right')
        ax2.plot(values_psi.imag, color = 'crimson', label='Imaginary part')
        ax2.legend(loc = 'upper right')
        plt.show()
    
    return values_psi
    

# This function computes the loss for the time-dependent Schrödinger equation (TDSE)
# Define a placeholder for the loss function (if needed, implement it here or use the global one)
def loss_fun(timeq, model, boundary_conditions, initial_conditions):
    """
    Neural network predicting [Re(Ψ), Im(Ψ)].
    hbar, mass: Physical constants.
    V: Potential function (or precomputed tensor).
    lambda_bc, lambda_ic: Loss weights for BCs/ICs.
    """
    # Unpack the boundary conditions and the initial conditions. 
    mass = initial_conditions['mass']
    hbar = initial_conditions['hbar']
    
    lambda_ic = initial_conditions['lambda_ic']

    L = initial_conditions['L'] # The length of the box
    V = initial_conditions['V'] # Potential function (or precomputed tensor)
    N = initial_conditions['N'] # Number of points in the domain
    
    psi_0yzt = initial_conditions['psi_0yzt']
    psi_Lyzt = initial_conditions['psi_Lyzt']
    psi_x0zt = initial_conditions['psi_x0zt']
    psi_xLzt = initial_conditions['psi_xLzt']
    psi_xy0t = initial_conditions['psi_xy0t']
    psi_xyLt = initial_conditions['psi_xyLt']

    # Unpack the boundary conditions
    lambda_bc = boundary_conditions['lambda_bc'] # Weight for boundary conditions
    u0 = boundary_conditions['u0'] # Boundary conditions for the real part of Ψ
    v0 = boundary_conditions['v0'] # Boundary conditions for the imaginary part of Ψ
    
    # Split inputs into spatial and time components
    # Supose a perfect box of lenght L 
    space = torch.linspace(0, L, N).reshape(-1, 1)  # Spatial coordinates

    x = space[:, 0:1].requires_grad_(True)  # Spatial x coordinates
    y = space[:, 0:1].requires_grad_(True)  # Spatial y coordinates
    z = space[:, 0:1].requires_grad_(True)  # Spatial z coordinates
    t = timeq*torch.ones_like(x).requires_grad_(True)  # Time is constant for the batch
    
    # Forward pass: Predict u (real) and v (imaginary)
    output = model(torch.cat([x, y, z, t], dim=1))  # Create a grid of points in the domain

    # Split the output into real and imaginary parts
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
def train_model(model, params_set):
    
    # Unpack the parameters
    L, Number_of_points, BCs, ICs, eta, l2_lambda, num_epochs = params_set.values()
        
    # This Neuronal Nework use a 4D input, so we need to create a 4D tensor
    # The input is a tensor of shape (N, 4) where N is the number of points in the domain
    train_r = torch.linspace(0, L, Number_of_points).reshape(-1, 1).requires_grad_(True)

    # Define the optimizer 
    # Adam optimizer with learning rate eta and weight decay l2_lambda
    # Note: weight decay is used for L2 regularization
    # Note: Adam optimizer uses a default L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=l2_lambda)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad() # zero the gradients
        # Compute the interior and boundary losses
        loss, pde_loss, boundary_loss, ic_loss = loss_fun(0, model, BCs, ICs)
        loss.backward() # backpropagation
        optimizer.step() # update the weights
      
        # Print the losses every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Function Loss: {pde_loss}, Boundary Loss: {boundary_loss}, Initial_cond_loss: {ic_loss}, Total Loss: {loss}")


    # Print the final output results.
    print("Training completed.\n")