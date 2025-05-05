import torch
import torch.nn as nn
import torch.autograd as autograd

torch.manual_seed(0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class SchrodingerPINN(nn.Module):
    def __init__(self, layers):
        """
        layers: list of layer widths, e.g. [4, 100, 100, 100, 2]
        input: (x,y,z,t), output: (Re(psi), Im(psi))
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        self.apply(init_weights)

    def forward(self, X):
        # X: [N, 4]
        u = X
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)

# PDE residual for time-dependent Schrödinger
# H = - (hbar^2 / 2m) \nabla^2 + V(x,y,z)

def schrodinger_residual(model, x, y, z, t, V_func, hbar=1.0, m=1.0):
    X = torch.stack([x, y, z, t], dim=1).requires_grad_(True)
    psi = model(X)               # [N,2]
    psi_r = psi[:,0:1]
    psi_i = psi[:,1:2]

    # time derivatives
    psi_r_t = autograd.grad(psi_r, t, torch.ones_like(psi_r), create_graph=True)[0]
    psi_i_t = autograd.grad(psi_i, t, torch.ones_like(psi_i), create_graph=True)[0]

    # spatial second derivatives
    grads = {}
    for name, comp in [('r', psi_r), ('i', psi_i)]:
        comp_x = autograd.grad(comp, x, torch.ones_like(comp), create_graph=True)[0]
        comp_xx = autograd.grad(comp_x, x, torch.ones_like(comp_x), create_graph=True)[0]
        comp_y = autograd.grad(comp, y, torch.ones_like(comp), create_graph=True)[0]
        comp_yy = autograd.grad(comp_y, y, torch.ones_like(comp_y), create_graph=True)[0]
        comp_z = autograd.grad(comp, z, torch.ones_like(comp), create_graph=True)[0]
        comp_zz = autograd.grad(comp_z, z, torch.ones_like(comp_z), create_graph=True)[0]
        grads[f'{name}_lap'] = comp_xx + comp_yy + comp_zz

    # potential term
    V = V_func(x, y, z)       # [N,1]

    # real and imag residuals
    res_r = hbar * psi_i_t + (hbar**2/(2*m)) * grads['r_lap'] - V * psi_r
    res_i = -hbar * psi_r_t + (hbar**2/(2*m)) * grads['i_lap'] - V * psi_i
    return res_r, res_i

# Sampling utilities
def sample_domain(N):
    # uniform sampling in [0,L]^3 x [0,T]
    x = torch.rand(N, 1)*L
    y = torch.rand(N, 1)*L
    z = torch.rand(N, 1)*L
    t = torch.rand(N, 1)*T
    return x, y, z, t

def sample_initial(N):
    # at t=0, sample for initial condition
    x = torch.rand(N,1)*L
    y = torch.rand(N,1)*L
    z = torch.rand(N,1)*L
    t = torch.zeros_like(x)
    return x, y, z, t

# Example potential and initial wavefunction (user can modify)
L, T = 1.0, 1.0

def V_func(x, y, z):
    # free particle
    return torch.zeros_like(x)

def psi_initial(x, y, z):
    # gaussian packet
    return torch.exp(-((x-0.5*L)**2 + (y-0.5*L)**2 + (z-0.5*L)**2)/(0.1**2))

# Training
model = SchrodingerPINN([4, 100, 100, 100, 2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 5001):
    # PDE residual loss
    x_r, y_r, z_r, t_r = sample_domain(2000)
    res_r, res_i = schrodinger_residual(model, x_r, y_r, z_r, t_r, V_func)
    loss_pde = (res_r**2 + res_i**2).mean()

    # Initial condition loss
    x_i, y_i, z_i, t_i = sample_initial(1000)
    psi0 = psi_initial(x_i, y_i, z_i)
    psi_pred = model(torch.stack([x_i, y_i, z_i, t_i], dim=1))
    psi_r0, psi_i0 = psi_pred[:,0:1], psi_pred[:,1:2]
    loss_ic = ((psi_r0 - psi0)**2 + psi_i0**2).mean()

    loss = loss_pde + loss_ic
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.3e}")

print("Training complete")
