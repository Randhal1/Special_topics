# Import necessary libraries
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define constants of the equation in natural units
hbar = 1.0
m = 1.0
L = 1.0
T = 1.0

# Define the Schrodinger equation which inherits from nn.Module
class SchrodingerPINN1D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
        for m in self.layers:
            if isinstance(m, nn.Linear):
                # Use Xavier initialization, which is good for the Tanh activation function
                # Xavier initialization is the best choice for the Schrodinger equation
                # because it helps to keep the variance of the activations constant
                nn.init.xavier_normal_(m.weight) 
                nn.init.zeros_(m.bias)

    # Do the forward pass
    # The input is a concatenation of x and t
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        u = X
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)
    
    # Returns the number of parameters in the model
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
    psi_r_t = autograd.grad(psi_r, t, torch.ones_like(psi_r), create_graph=True)[0]
    psi_i_t = autograd.grad(psi_i, t, torch.ones_like(psi_i), create_graph=True)[0]
    psi_r_x = autograd.grad(psi_r, x, torch.ones_like(psi_r), create_graph=True)[0]
    psi_r_xx = autograd.grad(psi_r_x, x, torch.ones_like(psi_r_x), create_graph=True)[0]
    psi_i_x = autograd.grad(psi_i, x, torch.ones_like(psi_i), create_graph=True)[0]
    psi_i_xx = autograd.grad(psi_i_x, x, torch.ones_like(psi_i_x), create_graph=True)[0]
    V = V_func(x)
    res_r =  hbar * psi_i_t + (hbar**2/(2*m)) * psi_r_xx - V * psi_r
    res_i = -hbar * psi_r_t + (hbar**2/(2*m)) * psi_i_xx - V * psi_i
    return res_r, res_i


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
# This model inbuits the optimizer and the loss function
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

    # Store the losses.
    losses = {
        'loss': [],
        'loss_pde': [],
        'loss_ic': [],
        'loss_bc': []
    }


    # Stage 1: Adam optimization
    print("Stage 1: Adam optimization")

    for epoch in range(1, epochs+1):

        #optimizer.zero_grad()
        # Sample the domain, initial conditions and boundary conditions
        x_r, t_r = sample_domain(2000)
        
        # Compute the residual of the PDE
        res_r, res_i = schrodinger_residual(model, x_r, t_r)

        # Compute the PDE loss
        loss_pde = (res_r**2 + res_i**2).mean()
        # Sample the initial condition
        x_i, t_i = sample_initial(1000)
        x0, t0, xL, tL = sample_boundary(1000)     
        
        # Compute the initial condition loss
        pred0 = model(x_i, t_i)
        psi0 = psi_0(x_i)
        loss_ic = ((pred0[:,0:1] - psi0)**2 + pred0[:,1:2]**2).mean()
        
        # Compute the boundary condition loss
        p0 = model(x0, t0); pL = model(xL, tL)
        loss_bc = (p0**2 + pL**2).mean()
        
        # Compute the total loss
        # Lambda_ic and Lambda_bc are the weights for the initial and boundary conditions
        # After some tests, I found that 10 is a good value for both
        loss = loss_pde + 10*loss_ic + 10*loss_bc
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Store the losses
        losses['loss'].append(loss.item())
        losses['loss_pde'].append(loss_pde.item())
        losses['loss_ic'].append(10*loss_ic.item())
        losses['loss_bc'].append(10*loss_bc.item())
        
        # Print the loss every 500 epochs
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.2e}, PDE Loss: {loss_pde.item():.2e}, IC Loss: {loss_ic.item():.2e}, BC Loss: {loss_bc.item():.2e}")

    print("Stage 1: Adam optimization finished. \n\nStage 2: LBFGS optimization.") 

    # Stage 2: LBFGS optimization
    # The strong Wolfe line search is used to find the optimal step size
    fine_opt = optim.LBFGS(model.parameters(), max_iter=max_iter_lb, history_size=10, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        fine_opt.zero_grad()
        # Sample the domain, initial conditions and boundary conditions
        x_r, t_r = sample_domain(2000)
        
        # Compute the residual of the PDE
        res_r, res_i = schrodinger_residual(model, x_r, t_r)

        # Compute the PDE loss
        loss_pde = (res_r**2 + res_i**2).mean()
        # Sample the initial condition
        x_i, t_i = sample_initial(1000)
        x0, t0, xL, tL = sample_boundary(1000)     
        
        # Compute the initial condition loss
        pred0 = model(x_i, t_i)
        psi0 = psi_0(x_i)
        loss_ic = ((pred0[:,0:1] - psi0)**2 + pred0[:,1:2]**2).mean()
        
        # Compute the boundary condition loss
        p0 = model(x0, t0); pL = model(xL, tL)
        loss_bc = (p0**2 + pL**2).mean()
        
        # Compute the total loss
        loss = loss_pde + 10*loss_ic + 10*loss_bc

        # Store the losses
        losses['loss'].append(loss.item())
        losses['loss_pde'].append(loss_pde.item())
        losses['loss_ic'].append(10*loss_ic.item())
        losses['loss_bc'].append(10*loss_bc.item())

        loss.backward()
        
        return loss

    # Run the LBFGS optimizer
    for i in range(max_iter_lb):
        final_loss = fine_opt.step(closure)
        if i % 500 == 0:
            print(f"Final loss: {final_loss:.2e}, PDE Loss: {loss_pde.item():.2e}, IC Loss: {loss_ic.item():.2e}, BC Loss: {loss_bc.item():.2e}")
        
    return losses, epoch+max_iter_lb


# Define a function to compute the exact solution
def psi_exact(x_plot, t_plot, L = 1, n=1):
    # Define the exact solution of the Schrodinger equation
    E1 = (n**2 * np.pi**2 * hbar**2)/(2*m*L**2)
    return np.sqrt(2/L) * np.sin(n*np.pi * x_plot.numpy() / L) * np.exp(-1j*E1*t_plot)


# Create a function to plot the results
def plot_results(t_plot, x_plot, model, losses, epochs, nl = 1, exact_sol = None):

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
    if exact_sol:
        veritas = exact_sol(x_plot, t_plot)
    else:
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
