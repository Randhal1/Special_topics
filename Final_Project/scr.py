# Import necessary libraries
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define constants of the equation in natural units
hbar = 1.0
m    = 1.0
L    = 1.0
T    = 1.0

# --- SIREN layer definition ---
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, is_first=False,
                 w0=30.0, c=6.0):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.w0 = w0
        self.is_first = is_first
        # weight init
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(c / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

# Define the Schrodinger equation which inherits from nn.Module
class SchrodingerPINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # layers is a list of dimensions, e.g. [2,128,128,2]
        self.net = nn.ModuleList()
        # first SIREN layer
        self.net.append(SirenLayer(layers[0], layers[1], is_first=True, w0=30.0))
        # hidden SIREN layers
        for i in range(1, len(layers)-2):
            self.net.append(SirenLayer(layers[i], layers[i+1], is_first=False, w0=1.0))
        # final linear head (no sine activation)
        self.net.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x, t):
        # concatenate inputs
        xt = torch.cat([x, t], dim=1)
        y = xt
        for layer in self.net:
            y = layer(y)
        return y

    def countpar(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Define the Schrodinger equation Potential term
def V_func(x):
    return torch.zeros_like(x)

# Define the loss function for the PDE
def schrodinger_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    psi = model(x, t)
    psi_r, psi_i = psi[:,0:1], psi[:,1:2]
    # time derivatives
    psi_r_t = autograd.grad(psi_r, t, torch.ones_like(psi_r), create_graph=True)[0]
    psi_i_t = autograd.grad(psi_i, t, torch.ones_like(psi_i), create_graph=True)[0]
    # spatial derivatives
    psi_r_x  = autograd.grad(psi_r, x, torch.ones_like(psi_r), create_graph=True)[0]
    psi_i_x  = autograd.grad(psi_i, x, torch.ones_like(psi_i), create_graph=True)[0]
    psi_r_xx = autograd.grad(psi_r_x, x, torch.ones_like(psi_r_x), create_graph=True)[0]
    psi_i_xx = autograd.grad(psi_i_x, x, torch.ones_like(psi_i_x), create_graph=True)[0]
    # real and imaginary residuals
    # i*hbar*psi_t + (hbar^2/2m)*psi_xx - V(x)*psi
    r_r = - hbar * psi_i_t + (hbar**2/(2*m)) * psi_r_xx - V_func(x)*psi_r
    r_i =   hbar * psi_r_t + (hbar**2/(2*m)) * psi_i_xx - V_func(x)*psi_i
    return r_r, r_i

# Define the function to sample the domain
def sample_domain(N):
    x = torch.rand(N,1) * L
    t = torch.rand(N,1) * T
    return x, t

# Define the function to sample the initial condition
def sample_initial(N):
    x = torch.rand(N,1) * L
    t = torch.zeros_like(x)
    return x, t

# Define the function to sample the boundary condition
def sample_boundary(N):
    t0 = torch.rand(N,1) * T
    x0 = torch.zeros(N,1)
    xL = torch.full((N,1), L)
    return x0, t0, xL, t0

# Create a function to train the model
def train(model, params, psi_0):

    # Import the parameters for the training
    epochs = params['epochs']
    max_iter_lb =  int((params['fine_tuning%']/100) * params['epochs'])
    print(f"Started with --> Adam epochs: {epochs}, LBFGS epochs: {max_iter_lb}")
    eta = params['eta']
    l2_lambda = params['l2_lambda']

    # Define the optimizer Adam for the general training
    # Adam optimizer with learning rate eta and weight decay l2_lambda
    # Note: weight decay is used for L2 regularization
    # Note: Adam optimizer uses a default L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=l2_lambda)

    # Prepare fixed collocation/IC/BC sets
    x_r, t_r       = sample_domain(int( params.get('collocation_pts',20000) ))
    x_i_full, t_i_full = sample_initial(int( params.get('ic_pts',5000) ))
    x0, t0, xL, tL = sample_boundary(int( params.get('bc_pts',5000) ))
    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    losses = {
        'loss': [],
        'loss_pde': [],
        'loss_ic': [],
        'loss_bc': []
    }

    # Adam training phase
    for epoch in range(1, epochs+1):
        # Sample mini-batches from fixed sets
        idx_r = torch.randperm(x_r.size(0))[:2000]
        xr, tr = x_r[idx_r], t_r[idx_r]
        idx_i = torch.randperm(x_i_full.size(0))[:1000]
        xi, ti = x_i_full[idx_i], t_i_full[idx_i]
        idx_b = torch.randperm(x0.size(0))[:1000]
        x0_b, t0_b = x0[idx_b], t0[idx_b]
        xL_b, tL_b = xL[idx_b], tL[idx_b]

        # Compute the residual of the PDE
        res_r, res_i = schrodinger_residual(model, xr, tr)
        loss_pde = (res_r**2 + res_i**2).mean()
        
        # Compute the initial condition loss
        pred0 = model(xi, ti)
        psi0_vals = psi_0(xi)
        loss_ic = ((pred0[:,0:1] - psi0_vals)**2 + pred0[:,1:2]**2).mean()
        
        # Compute the boundary condition loss
        p0 = model(x0_b, t0_b); pL = model(xL_b, tL_b)
        loss_bc = (p0**2 + pL**2).mean()

        
        # Compute the total loss
        # Lambda_ic and Lambda_bc are the weights for the initial and boundary conditions
        # After some tests, I found that 10 is a good value for both
        loss = loss_pde + 10*loss_ic + 10*loss_bc

        # Store the losses
        losses['loss'].append(loss.item())
        losses['loss_pde'].append(loss_pde.item())
        losses['loss_ic'].append(10*loss_ic.item())
        losses['loss_bc'].append(10*loss_bc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.2e}, PDE: {loss_pde.item():.2e}, IC: {loss_ic.item():.2e}, BC: {loss_bc.item():.2e}")

    # Fine-tuning with L-BFGS
    lbfgs_opt = optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter_lb,
                             tolerance_grad=1e-9, tolerance_change=1e-12,
                             line_search_fn='strong_wolfe')

    def closure():
        # Store the losses
        lbfgs_opt.zero_grad()
        res_r, res_i = schrodinger_residual(model, x_r, t_r)
        loss_pde = (res_r**2 + res_i**2).mean()
        pred0 = model(x_i_full, t_i_full)
        psi0_vals = psi_0(x_i_full)
        loss_ic  = ((pred0[:,0:1] - psi0_vals)**2 + pred0[:,1:2]**2).mean()
        p0, pL = model(x0, t0), model(xL, tL)
        loss_bc  = (p0**2 + pL**2).mean()
        loss = loss_pde + 10*loss_ic + 10*loss_bc
        loss.backward()
        return loss

    print("Starting L-BFGS fine-tuning…")
    final_loss = lbfgs_opt.step(closure)
    print(f"[L-BFGS] Final Loss: {final_loss.item():.2e}")

    return losses, epoch+max_iter_lb

# Define a function to compute the exact solution
def psi_exact(x_plot, t_plot, L = 1, n=1):
    # Define the exact solution of the Schrodinger equation
    E1 = (n**2 * np.pi**2 * hbar**2)/(2*m*L**2)
    return np.sqrt(2/L) * np.sin(n*np.pi * x_plot.numpy() / L) * np.exp(-1j*E1*t_plot)

# Function to plot the results (unchanged)
# Create a function to plot the results
def plot_results(t_plot, x_plot, model, losses, epochs, nl = 1):

    # 1000 points in the x direction
    n = 1000

    x_plot = torch.linspace(0, L, n).unsqueeze(1)
    t_plot_t = torch.full_like(x_plot, t_plot)

    # Compute the predicted solution
    with torch.no_grad():
        pred = model(x_plot, t_plot_t).numpy()

    psi_r_pred = pred[:,0]
    psi_i_pred = pred[:,1]

    # Exact infinite-well n=1
    veritas = psi_exact(x_plot, t_plot, n=nl)
    psi_r_ex = np.real(veritas).flatten()
    psi_i_ex = np.imag(veritas).flatten()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    
    fig.suptitle(f'Schrodinger Equation 1D PINN at t={t_plot}', fontsize=16)
    #fig.tight_layout()

    # The left subplot is for the real part
    ax1.plot(x_plot, psi_r_pred, label='Real Model')
    ax1.plot(x_plot, psi_r_ex, '--', label='Real Exact')
    ax1.set_title(f'Real part.')
    ax1.set_xlim(0, L)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('$\psi(x,t)_{Real}$')
    ax1.legend()
    
    # The center subplot is for the imaginary part
    ax2.plot(x_plot, -psi_i_pred, label='Imag Model')
    ax2.plot(x_plot, psi_i_ex, '--', label='Imag Exact')
    ax2.set_title(f'Imag part.')
    ax2.set_xlim(0, L)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('$\psi(x,t)_{Imag}$')
    ax2.legend()

    # The right subplot is the loss training
    ax3.set_title('Losses during training.')
    ax3.plot(losses['loss'], label='Total loss')
    ax3.plot(losses['loss_pde'], label='Loss PDE')
    ax3.plot(losses['loss_ic'], label='Loss IC')
    ax3.plot(losses['loss_bc'], label='Loss BC')
    ax3.legend()
    ax3.set_xlim(0, epochs)
    ax3.set_yscale('log')
    
    plt.show()

