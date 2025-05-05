import torch
from torch import nn
import torch.autograd as autograd
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, depth=4, activation='tanh'):
        super().__init__()
        activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU(), 
                       'leaky_relu': nn.LeakyReLU(), 'swish': nn.SiLU(), 'sin': torch.sin}
        act = activations.get(activation, nn.Tanh())
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act)
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def exact_solution(r, t, Ap=1.0, Bp=0.5, m=1.0, hbar=1.0, L=1.0):
    x = r[:, 0:1]
    y = r[:, 1:2]
    z = r[:, 2:3]
    omega1 = (np.pi / L)**2 * hbar / (2*m)
    omega2 = (2*np.pi / L)**2 * hbar / (2*m)
    sin1 = torch.sin(np.pi * x / L) * torch.sin(np.pi * y / L) * torch.sin(np.pi * z / L)
    sin2 = torch.sin(2*np.pi * x / L) * torch.sin(np.pi * y / L) * torch.sin(np.pi * z / L)
    real = Ap * sin1 * torch.cos(omega1 * t) + Bp * sin2 * torch.cos(omega2 * t)
    imag = -Ap * sin1 * torch.sin(omega1 * t) - Bp * sin2 * torch.sin(omega2 * t)
    return real, imag

def loss_fun(t_batch, model, BCs, ICs, Nf=2000, Nb=500):
    m, hbar, L = ICs['mass'], ICs['hbar'], ICs['L']
    lambda_ic, lambda_bc = ICs['lambda_ic'], BCs['lambda_bc']
    # interior points + time
    x = torch.rand(Nf,1, requires_grad=True) * L
    y = torch.rand(Nf,1, requires_grad=True) * L
    z = torch.rand(Nf,1, requires_grad=True) * L
    t = t_batch.detach().requires_grad_(True)
    if t.shape[0] != Nf:
        t = t.repeat(Nf,1)
    pts = torch.cat([x, y, z, t], dim=1)
    uv = model(pts)
    u, v = uv[:,0:1], uv[:,1:2]
    # derivatives
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]
    lap_u = u_xx + u_yy + u_zz
    lap_v = v_xx + v_yy + v_zz
    # PDE residual
    res_u = u_t + (hbar/(2*m))*lap_v
    res_v = v_t - (hbar/(2*m))*lap_u
    pde_loss = torch.mean(res_u**2 + res_v**2)
    # BCs ψ=0 on walls
    xb = torch.cat([torch.zeros(Nb,1), torch.ones(Nb,1)*L], dim=0)
    yb = torch.rand(2*Nb,1)*L
    zb = torch.rand(2*Nb,1)*L
    tb = torch.rand(2*Nb,1)*t_batch.mean().item()
    bpts = torch.cat([xb, yb, zb, tb], dim=1)
    bc_loss = torch.mean(model(bpts)**2)
    # IC at t=0
    t0 = torch.zeros(Nf,1)
    pts0 = torch.cat([x,y,z,t0], dim=1)
    u0p, v0p = model(pts0)[:,0:1], model(pts0)[:,1:2]
    u0e, v0e = exact_solution(pts0, t0)
    ic_loss = torch.mean((u0p-u0e)**2 + 1200*(v0p-v0e)**2)
    loss = pde_loss + lambda_bc*bc_loss + lambda_ic*ic_loss
    return 3000*loss, 4000*pde_loss, 3000*bc_loss, 2000*ic_loss

def train_model(model, params, device='cpu'):
    ICs = {'mass':params['mass'],'hbar':params['hbar'],'L':params['L'],'lambda_ic':params['lambda_ic']}
    BCs = {'lambda_bc':params['lambda_bc']}
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    for ep in range(params['epochs']):
        t_b = torch.rand((params['Nf'],1), device=device) * params['T_max']
        loss, pl, bl, il = loss_fun(t_b, model, BCs, ICs, Nf=params['Nf'], Nb=params['Nb'])
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%100==0:
            print(f"Epoch {ep+1}/{params['epochs']} | PDE {pl:.2e} | BC {bl:.2e} | IC {il:.2e}")
    return model
