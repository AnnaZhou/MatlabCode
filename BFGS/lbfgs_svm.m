%%this file implement the paper:
%%Aryan Mokhtari, Alejandro Ribeiro. A quasi-newton method for large scale support Vector Machine. ICASSP 2014.

function [w,iter,loss] = lbfgs_svm(x0,y0,fun)
 
 m = 30;
 k = 1;
 wDivision=1;
 eta0=0.1;
 eta=eta0;
 lr=0.01;
 gamma=0.1;
 delta=0.01;
 
 x=x0(:,1:m); y=y0(1,1:m);
 num =  size(x0,2); dim=size(x0,1);
 w=zeros(dim,1);
 B=eye(dim)*delta;
 
 for iter = 1:1000
     index=floor( rand(1,1)*(num-m) );
     index=min(index, (num-m));
     x=x0(:,(index+1):(index+m)); y=y0(1,(index+1):(index+m));
     
     f= w'*x;
     [v,g]=myloss(x,y,f,lr,w);

     H=inv(B)+gamma*eye(dim);
     wt1=w-eta0*H*g;
     
     f= wt1'*x;
     [v1,g1]=myloss(x,y,f,lr,wt1);
     
     vt=wt1-w;
     rt=g1-g-delta*vt;
     
     B1=B*vt*vt'*B/(vt'*B*vt);
     if (vt'*rt~=0)
         B2=rt*rt'/(vt'*rt);
     else 
         B2=0;
     end
     B=B+B2-B1+delta*eye(dim);
 
     w=wt1;
     loss=v1;
 end
 
end
