function [d] = lbfgs(g,s,y,H0)
% BFGS Search Direction
% p - dim
% k - samples
% s - previous search directions (p by k)
% y - previous step sizes (p by k)
% g - gradient (p by 1)
% H0 - value of initial Hessian diagonal elements (scalar)

[p,k] = size(s);

for i = 1:k
    ro(i,1) = 1/(y(:,i)'*s(:,i));
end

q = zeros(p,k+1);
r = zeros(p,k+1);
al =zeros(k,1);
be =zeros(k,1);

q(:,k+1) = g;

for i = k:-1:1
    al(i) = ro(i)*s(:,i)'*q(:,i+1);
    q(:,i) = q(:,i+1)-al(i)*y(:,i);
end

% Multiply by Initial Hessian
r(:,1) = H0*q(:,1);

for i = 1:k
    be(i) = ro(i)*y(:,i)'*r(:,i);
    r(:,i+1) = r(:,i) + s(:,i)*(al(i)-be(i));
end
d=r(:,k+1);
