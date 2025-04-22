#!../xptopics/bin/python
# Import the modules.
import numpy as np
import matplotlib.pyplot as plt
import timeit
from numba import njit
from tqdm import tqdm

#########################################
# Define the activation functions (Fx). #
#########################################

# The sigmoid function.
@njit
def sigmoid(x, W, b):
    # Evaluate the sigmoid function.
    x = x.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)
    arg = W@x+b
    return 1/(1+np.exp(-arg))

# The ReLu function.
@njit
def relu(x, W, b):
    # Evaluate the sigmoid function.
    x = x.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)
    arg = W@x+b
    return np.maximum(arg, 0)

# The Tanh function
@njit
def tanh(x, W, b):
    # Evaluate the sigmoid function.
    x = x.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)
    arg = W@x+b
    return np.tanh(arg)
    
# Define the derivatives for the backward pass.
@njit
def dsigmoid(ak, Ep):
    return ak*(1-ak)*Ep

@njit
def drelu(ak, Ep):
    b = np.where(ak > 0, 1, 0)
    return b*Ep

@njit
def dtanh(ak, Ep):
    return (1-ak**2)*Ep

####################################

# Define the cost function.
@njit
def cost(x1, x2, y, W2, W3, W4, b2, b3, b4, Fx2, Fx3, Fx4):

    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    y = y.astype(np.float64)
    W2 = W2.astype(np.float64)
    W3 = W3.astype(np.float64)
    W4 = W4.astype(np.float64)
    b2 = b2.astype(np.float64)
    b3 = b3.astype(np.float64)
    b4 = b4.astype(np.float64)

    costvec = np.zeros(10)
    
    for k in range(10):
        x = np.array([[x1[k]], [x2[k]]])
        a2 = Fx2(x, W2, b2)
        a3 = Fx3(a2, W3, b3)
        a4 = Fx4(a3, W4, b4)
        costvec[k] = np.linalg.norm(y[:, k:k+1] - a4, 2)

    return np.linalg.norm(costvec, 2)**2


# Create the core function netbp.
@njit
def netbp_core(x1, x2, y, savecost, W2, W3, W4, b2, b3, b4, eta, Fx2, Fx3, Fx4, dfx2, dfx3, dfx4, cost=cost):
    #np.random.seed(5000) # Set the random seed
    # Forward and back propagate
    # Pick a trining point at random
    k = np.random.randint(10)
    x = np.array([[x1[k]], [x2[k]]])

    # Forward pass
    a2 = Fx2(x, W2, b2)
    a3 = Fx3(a2, W3, b3)
    a4 = Fx4(a3, W4, b4)
    
    # Backward pass
    #delta4 = a4*(1-a4)*(a4-y[:, k:k+1])
    #delta3 = a3*(1-a3)*(W4.T@delta4)
    #delta2 = a2*(1-a2)*(W3.T@delta3)
    delta4 = dfx4(a4, a4-y[:, k:k+1])
    delta3 = dfx3(a3, W4.T@delta4)
    delta2 = dfx2(a2, W3.T@delta3)

    # Gradient step
    W2 -= eta*delta2@x.T
    W3 -= eta*delta3@a2.T
    W4 -= eta*delta4@a3.T
    b2 -= eta*delta2
    b3 -= eta*delta3
    b4 -= eta*delta4

    # Monitor progress
    newcost = cost(x1, x2, y, W2, W3, W4, b2, b3, b4, Fx2, Fx3, Fx4)
    
    return newcost, W2, W3, W4, b2, b3, b4


def gridfunc(N, W2, W3, W4, b2, b3, b4, Fx2, Fx3, Fx4):
    # For the center figure with the shaded region.
    xvals = np.linspace(0, 1, N+1)
    yvals = np.linspace(0, 1, N+1)
    Aval = np.zeros((N+1, N+1), dtype=np.float64)
    Bval = np.zeros((N+1, N+1), dtype=np.float64)

    for k1, xk in enumerate(xvals):
        for k2, yk in enumerate(yvals):
            xy = np.array([[xk], [yk]])
            a2 = Fx2(xy, W2, b2)
            a3 = Fx3(a2, W3, b3)
            a4 = Fx4(a3, W4, b4)

            Aval[k2, k1] = a4[0].item()
            Bval[k2, k1] = a4[1].item()

    #X, Y = np.meshgrid(xvals, yvals)
    X = np.empty((N+1, N+1), dtype=np.float64)
    Y = np.empty((N+1, N+1), dtype=np.float64)

    for i in range(N+1):
        X[i, :] = xvals  # Fill row i with xvals
        Y[:, i] = yvals  # Fill column i with yvals

    Mval = Aval > Bval
    Mval = Mval*1.0  # Convert boolean array to float

    return X, Y, Mval


# Contruct an auxiliar function to print outputs.
def plotter(cost, xdata1, ydata1, X, Y, Mval, title):
    print("Generating graphics.")
    # Create the figure with adjusted spacing
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(title)

    # The left figure
    ax1.set_title('Initial data')
    ax1.set_xlabel('$x$-coordinate')
    ax1.set_ylabel('$y$-coordinate')
    ax1.set_xlim([0,1])
    ax1.set_xticks([i/10 for i in range(11)])
    ax1.set_yticks([i/10 for i in range(11)])
    ax1.plot(xdata1[0:5], ydata1[0:5], 'or', label='Class 0')
    ax1.plot(xdata1[5:10], ydata1[5:10], 'Xb', label='Class 1')
    ax1.legend(loc='upper left').set_visible(True)

    # Plot decision boundary
    ax2.set_title('Decision Boundary')
    ax2.set_xlabel('$x$-coordinate')
    ax2.set_ylabel('$y$-coordinate')
    ax2.set_xlim([0,1])
    ax2.set_xticks([i/10 for i in range(11)])
    ax2.set_yticks([i/10 for i in range(11)])
    ax2.plot(xdata1[0:5], ydata1[0:5], 'or', label='Class 0')
    ax2.plot(xdata1[5:10], ydata1[5:10], 'Xb', label='Class 1')
    ax2.contourf(X, Y, Mval, levels=[-0.1, 0.5, 1.1], colors=["white", "lightgray"])
    ax2.legend(loc='upper left').set_visible(True)

    # Plot the cost function against the number of iterations

    xcost = np.linspace(0,1,len(cost))

    ax3.set_title('Cost vs Number of iterations')
    ax3.set_xlabel('Iteration number $\\times 10^{5}$')
    ax3.set_ylabel('Cost function')
    ax3.set_xlim([0,1])
    ax3.set_xticks([i/10 for i in range(11)])
    ax3.semilogy(xcost, cost, label='Cost function $C(x)$.')
    ax3.semilogy(1, cost[-1], 'or', label=f'C({len(cost)})={cost[-1]:.2e}.')

    plt.legend()
    plt.show(block=False)
    plt.savefig(title+'.png')
    #plt.close()


def netbp_full(eta, Fx2, Fx3, Fx4, dfx2, dfx3, dfx4, title, Niter=int(1e6)):
    # Visualize results
    # Data: xcoords, ycoords, targets
    x1 = np.array([0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7], dtype=np.float64)
    x2 = np.array([0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6], dtype=np.float64)
    y = np.array([[1]*5 + [0]*5, [0]*5 + [1]*5], dtype=np.float64)
    
    # Create the function netbp_full
    # Set up data for neural net test
    # Use backpropagation to train

    # Initialize weights and biases
    W2 = 0.5*np.random.randn(2,2)
    W3 = 0.5*np.random.randn(3,2)
    W4 = 0.5*np.random.randn(2,3)
    b2 = 0.5*np.random.randn(2,1)
    b3 = 0.5*np.random.randn(3,1)
    b4 = 0.5*np.random.randn(2,1)

    # Create the function to save the cost
    savecost = np.zeros(Niter)
    
    print("Initializing config: "+title)
    for i in tqdm(range(Niter), desc="Training neurons: ", colour="green"):
        savecost[i], W2, W3, W4, b2, b3, b4 = netbp_core(x1, x2, y, savecost, 
                                                        W2, W3, W4, b2, b3, b4, eta, 
                                                        Fx2, Fx3, Fx4, dfx2, dfx3, dfx4)

    X, Y, Mval = gridfunc(500, W2, W3, W4, b2, b3, b4, Fx2, Fx3, Fx4)

    # Plot all the figures. 
    plotter(savecost, x1, x2, X, Y, Mval, title)


if __name__ == '__main__':
    netbp_full(0.05, sigmoid, sigmoid, sigmoid, dsigmoid, dsigmoid, dsigmoid, "L2-L3-L4-Sigmoid")
    netbp_full(0.05, relu, relu, sigmoid, drelu, drelu, dsigmoid, "L2-L3-ReLU-L4-Sigmoid")
    netbp_full(0.05, tanh, tanh, sigmoid, dtanh, dtanh, dsigmoid, "L2-L3-tanh-L4-Sigmoid")
    netbp_full(0.5, relu, relu, sigmoid, drelu, drelu, dsigmoid, "L2-L3-ReLU-L4-Sigmoid-eta-0.5")
    netbp_full(0.5, tanh, tanh, sigmoid, dtanh, dtanh, dsigmoid, "L2-L3-tanh-L4-Sigmoid-eta-0.5")
    netbp_full(0.05, tanh, tanh, sigmoid, dtanh, dtanh, dsigmoid, "L2-L3-L4-tanh")
