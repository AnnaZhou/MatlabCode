
function [w,iter,loss] = lbfgs_classic(x0,y0,fun)
 
 m = 30;
 k = 1;
 wDivision=1;
 eta0=0.1;
 eta=eta0;
 lr=0.01;
 gamma=0.1;
 delta=0.01;
 dw=1;
 batchSize=30;
 
 x=x0(:,1:m); y=y0(1,1:m);
 num =  size(x0,2); dim=size(x0,1);
 w0=zeros(dim,1); 
 w=w0+delta;
 g0=0;
 B=eye(dim)*delta;
 
 for iter = 1:1000
     index=floor( rand(1,1)*(num-batchSize) );
     index=min(index, (num-batchSize));
     x=x0(:,(index+1):(index+batchSize)); y=y0(1,(index+1):(index+batchSize));
     
     f= w'*x;
     [v,g]=myloss(x,y,f,lr,w);

     s=w-w0;
     yy=g-g0;
     [p] = lbfgs(g,s,yy,B);
     dw= norm(p);
%      H=inv(B)+gamma*eye(dim);
     wt1=w-eta0*p;
     
     f= wt1'*x;
     [v1,g1]=myloss(x,y,f,lr,wt1);
     
%      vt=wt1-w;
%      rt=g1-g-delta*vt;
%      
%      B1=B*vt*vt'*B/(vt'*B*vt);
%      if (vt'*rt~=0)
%          B2=rt*rt'/(vt'*rt);
%      else 
%          B2=0;
%      end
%      B=B+B2-B1+delta*eye(dim);
 
     g0=g;
     w0=w;
     w=wt1;
     g=g1;
     loss=v1;
     if dw<0.00001
         break;
     end
 end
 
end
