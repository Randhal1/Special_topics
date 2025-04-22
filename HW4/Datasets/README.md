#Datafiles for PINN Exercise (Section 5.8) 

### Reference data for the one-dimensional steady non-linear advection-diffusion-reaction equation.

#### Regimes available:
* $P_e = 0.01, Da = 0.01$: Diffusion dominates (Diff)
* $P_e = 20, Da = 0.01$: Advection dominates (Adv)
* $P_e = 0.01, Da = 60$: Reaction dominates (React)
* $P_e = 15, Da = 40$: Advection and reaction dominate (AdvReact)

#### Loading the data:

**Method 1**: You can load all four reference solutions stored in ```RefData.pkl```. To do this, use the following python script

```
import pickle

with open('RefData.pkl', 'rb') as fp:
    RefDataDict = pickle.load(fp)
```
```RefDataDict``` will be a python dictionary with the string keys  ```'Diff'```, ```'Adv'```, ```'Reac'```, ```'AdvReact'```. You can use the command ```RefDataDict.keys()``` to see the keys.

The value of each key is a list containing three items: $P_e$, $Da$, and a two-columned array containing the $(x, u(x))$ pairs of the reference solution. For example, ```RefDataDict['Adv'][0]``` will be 20, ```RefDataDict['Diff'][1]``` will be 0.01, and ```RefDataDict['Adv'][2]``` will be an 2D array with shape $N \times 2$.

**Method 2**: The reference data for the four regimes have also been provided in the files ```Pe_0.01_Da_0.01.txt```, ```Pe_20_Da_0.01.txt```, ```Pe_0.01_Da_60.txt```, ```Pe_15_Da_40.txt``` where the file names point to the various regimes. You can load these files directly by
```
import numpy as np

ref1 = np.loadtxt('Pe_0.01_Da_0.01.txt')
ref2 = np.loadtxt('Pe_20_Da_0.01.txt')
ref3 = np.loadtxt('Pe_0.01_Da_60.txt')
ref4 = np.loadtxt('Pe_15_Da_40.txt')
```

**Additional regimes**: If you are interested in testing the PINN out with other regimes, we have also provided a MATLAB script ```ref_sol_gen.m``` to generate the new reference solutions. In this script, you should change ```fname``` to a new file name to save to, and change the values of ```Pe``` and ```Da``` in the function ```dydx```. Note, you may need to change the initial guess in the function ```guess``` to converge to a positive solution of the non-linear problem.
