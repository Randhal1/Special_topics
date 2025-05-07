import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd

class MLP(nn.Module):
    def __init__(self, hparams_set):
        input_dim, out_dim, width, depth, activation, initialization = hparams_set.values()
        super(MLP, self).__init__()
        assert depth > 1, "Depth must be greater than 1"
        self.depth = depth
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = torch.sin
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        # Layer construction
        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, out_dim))
        self.model = nn.ModuleList(layers)

        # Weight initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                if initialization == 'uniform':
                    nn.init.uniform_(layer.weight, -1, 1)
                elif initialization == 'normal':
                    nn.init.normal_(layer.weight, std=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.model.apply(init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.model[:-1]):
            x = self.activation(layer(x))
        x = self.model[-1](x)  # No activation for output layer
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

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


def loss_fun(model, boundary_conditions, initial_conditions, t_batch):
    mass = initial_conditions['mass']
    hbar = initial_conditions['hbar']
    L = initial_conditions['L']
    V = initial_conditions['V']
    N = initial_conditions['N']
    lambda_bc = boundary_conditions['lambda_bc']
    lambda_ic = initial_conditions['lambda_ic']

    # Generate 3D spatial grid with time sampling
    coords = torch.linspace(0, L, N)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    x = x.reshape(-1, 1).requires_grad_(True)
    y = y.reshape(-1, 1).requires_grad_(True)
    z = z.reshape(-1, 1).requires_grad_(True)
    t = torch.full_like(x, t_batch).requires_grad_(True)

    inputs = torch.cat([x, y, z, t], dim=1)
    output = model(inputs)
    u, v = output[:, 0:1], output[:, 1:2]

    # PDE Residuals
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    dv_dt = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]

    # Spatial derivatives
    def laplacian(f, x, y, z):
        df_dx = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
        df_dy = torch.autograd.grad(f, y, torch.ones_like(f), create_graph=True)[0]
        df_dz = torch.autograd.grad(f, z, torch.ones_like(f), create_graph=True)[0]
        d2f_dx2 = torch.autograd.grad(df_dx, x, torch.ones_like(df_dx), create_graph=True)[0]
        d2f_dy2 = torch.autograd.grad(df_dy, y, torch.ones_like(df_dy), create_graph=True)[0]
        d2f_dz2 = torch.autograd.grad(df_dz, z, torch.ones_like(df_dz), create_graph=True)[0]
        return d2f_dx2 + d2f_dy2 + d2f_dz2

    laplacian_u = laplacian(u, x, y, z)
    laplacian_v = laplacian(v, x, y, z)

    residual_u = (hbar**2/(2*mass)) * laplacian_u - V*u - hbar*dv_dt
    residual_v = (hbar**2/(2*mass)) * laplacian_v - V*v + hbar*du_dt
    pde_loss = torch.mean(residual_u**2 + residual_v**2)

    # Boundary conditions
    bc_mask = ((x == 0) | (x == L) | (y == 0) | (y == L) | (z == 0) | (z == L))
    bc_loss = torch.mean(u[bc_mask]**2 + v[bc_mask]**2)

    # Initial conditions (t=0 only)
    if t_batch == 0:
        ic_loss = torch.mean((u - initial_conditions['psi_0'])**2 + (v - initial_conditions['psi_0'])**2)
    else:
        ic_loss = torch.tensor(0.0)

    total_loss = pde_loss + lambda_bc*bc_loss + lambda_ic*ic_loss
    return total_loss, pde_loss.item(), bc_loss.item(), ic_loss.item()

def train_model(model, params_set):
    L = params_set['L']
    N = params_set['Number_of_points']
    BCs = params_set['BCs']
    ICs = params_set['ICs']
    eta = params_set['eta']
    l2_lambda = params_set['l2_lambda']
    num_epochs = params_set['num_epochs']
    T_max = params_set.get('T_max', 1.0)  # Default max time

    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=l2_lambda)
    
    # Time sampling during training
    time_steps = torch.linspace(0, T_max, 10)  # Sample 10 time points
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for t_batch in time_steps:
            optimizer.zero_grad()
            loss, pde_loss, bc_loss, ic_loss = loss_fun(model, BCs, ICs, t_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 100 == 0:
            avg_loss = epoch_loss / len(time_steps)
            print(f"Epoch {epoch+1}: Avg Loss {avg_loss:.4e}")

    print("Training completed.\n")

# Example usage 

if __name__ == "__main__":
    hparams = {
        'input_dim': 4,  # x,y,z,t
        'out_dim': 2,    # Re(Ψ), Im(Ψ)
        'width': 64,
        'depth': 4,
        'activation': 'sin',
        'initialization': 'normal'
    }
    model = MLP(hparams)

    params_set = {
        'L': 1.0,
        'Number_of_points': 16,
        'BCs': {'lambda_bc': 100, 'u0': 0.0, 'v0': 0.0},
        'ICs': {
            'mass': 1.0,
            'hbar': 1.0,
            'V': 0.0,
            'L': 1.0,
            'N': 100,
            'lambda_ic': 100,
            'psi_0': exact_solution_example(torch.zeros(1,3), 0.0)  # Example IC
        },
        'eta': 1e-3,
        'l2_lambda': 1e-4,
        'num_epochs': 1000,
        'T_max': 1.0
    }

    train_model(model, params_set)
