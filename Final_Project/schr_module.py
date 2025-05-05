import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

# Physics constants
hbar = 1.0  # Planck's constant
m = 1.0     # particle mass

# Domain setup
L = 1.0     # spatial domain [0, L]
T = 1.0     # time domain [0, T]

# PINN network: maps (x, t) -> (Re(ψ), Im(ψ))
class SchrodingerPINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        # x, t are shape [N,1]
        X = torch.cat([x, t], dim=1)  # [N,2]
        u = X
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)  # [N,2]

# Potential function
def V_func(x):
    # Infinite well: V=0 inside, large outside (enforced by BCs)
    return torch.zeros_like(x)

# PDE residual
def schrodinger_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    psi = model(x, t)
    psi_r = psi[:, 0:1]
    psi_i = psi[:, 1:2]

    # time derivatives
    psi_r_t = autograd.grad(psi_r, t, torch.ones_like(psi_r), create_graph=True)[0]
    psi_i_t = autograd.grad(psi_i, t, torch.ones_like(psi_i), create_graph=True)[0]

    # spatial second derivatives
    psi_r_x = autograd.grad(psi_r, x, torch.ones_like(psi_r), create_graph=True)[0]
    psi_r_xx = autograd.grad(psi_r_x, x, torch.ones_like(psi_r_x), create_graph=True)[0]
    psi_i_x = autograd.grad(psi_i, x, torch.ones_like(psi_i), create_graph=True)[0]
    psi_i_xx = autograd.grad(psi_i_x, x, torch.ones_like(psi_i_x), create_graph=True)[0]

    V = V_func(x)
    # Real and imaginary parts of TDSE residual
    res_r =  hbar * psi_i_t + (hbar**2/(2*m)) * psi_r_xx - V * psi_r
    res_i = -hbar * psi_r_t + (hbar**2/(2*m)) * psi_i_xx - V * psi_i
    return res_r, res_i

# Sampling utilities
def sample_domain(N):
    x = torch.rand(N, 1) * L
    t = torch.rand(N, 1) * T
    return x, t

def sample_initial(N):
    x = torch.rand(N,1) * L
    t = torch.zeros_like(x)
    return x, t

def sample_boundary(N):
    # enforce ψ=0 at x=0 and x=L for all t
    t0 = torch.rand(N,1) * T
    x0 = torch.zeros(N,1)
    xL = torch.full((N,1), L)
    return x0, t0, xL, t0.clone()

# Initial condition: Gaussian wavepacket
def psi_initial(x):
    x0 = 0.5 * L
    sigma = 0.1
    return torch.exp(-((x-x0)**2)/(2*sigma**2))

# Build model
layers = [2, 64, 64, 64, 2]
model = SchrodingerPINN1D(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 3000
for epoch in range(1, epochs+1):
    # PDE loss
    x_r, t_r = sample_domain(2000)
    res_r, res_i = schrodinger_residual(model, x_r, t_r)
    loss_pde = (res_r**2 + res_i**2).mean()
    # Initial condition loss
    x_i, t_i = sample_initial(1000)
    psi0 = psi_initial(x_i)
    pred0 = model(x_i, t_i)
    loss_ic = ((pred0[:,0:1] - psi0)**2 + pred0[:,1:2]**2).mean()
    # Boundary condition loss
    x0, t0, xL, tL = sample_boundary(1000)
    psi0_l = model(x0, t0)
    psiL_l = model(xL, tL)
    loss_bc = (psi0_l**2 + psiL_l**2).mean()
    # Total loss
    loss = loss_pde + loss_ic*10 + loss_bc*10

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.3e}")

# Visualization at t=T/2
t_plot = 0.5 * T
n_plot = 200
x_plot = torch.linspace(0, L, n_plot).unsqueeze(1)
t_plot_t = torch.full_like(x_plot, t_plot)
with torch.no_grad():
    psi_pred = model(x_plot, t_plot_t).numpy()
psi_r = psi_pred[:,0]
psi_i = psi_pred[:,1]

plt.figure(figsize=(8,4))
plt.plot(x_plot, psi_r, label='Re ψ')
plt.plot(x_plot, psi_i, label='Im ψ')
plt.legend()
plt.title('PINN Solution at t=0.5T')
plt.show()
