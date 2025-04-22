clc
clear all
close all

fname = 'Pe_15_Da_40.txt'; % Change name as needed

xmesh = linspace(0,1,1000);
solinit = bvpinit(xmesh, @guess);

sol = bvp4c(@bvpfcn, @bcfcn, solinit);

x = sol.x;
u = sol.y(1,:);
plot(x, u, '-o')

dlmwrite(fname,[x',u'],'delimiter',' ');

function dydx = bvpfcn(x,y) % equation to solve
dydx = zeros(2,1);
Pe   = 15;  % Change as needed
Da   = 40;  % Change as needed
dydx = [y(2)
       Pe*y(2) - Da*y(1)*(1-y(1))];
end
%--------------------------------
function res = bcfcn(ya,yb) % boundary conditions
res = [ya(1)
       yb(1)-1];
end
%--------------------------------
function g = guess(x) % initial guess for y and y'
g = [0*x + 1
     0*x];
% g = [0*x
%      0*x];
end

