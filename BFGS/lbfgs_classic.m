
function [w,iter,loss] = lbfgs_classic(x0,y0,fun)
 
 m = 50;
 k = 1;
 eta0=0.1;
 lr=0.01;
 gamma=0.1;
 delta=0.01;
 dw=1;
 batchSize=30;
 
 num =  size(x0,2); dim=size(x0,1);
 w0=zeros(dim,1); 
 w=w0+delta;
 g0=0;
 B=eye(dim)*delta;
 
 x=x0(:,1); y=y0(1,1);
 f= w'*x;
 [v,g]=myloss(x,y,f,lr,w);
 mys=w-w0;
 myy=g-g0;
 
 for iter = 1:1000
     index=floor( rand(1,1)*(num-batchSize) );
     index=min(index, (num-batchSize));
     x=x0(:,(index+1):(index+batchSize)); y=y0(1,(index+1):(index+batchSize));
     
     f= w'*x;
     [v,g]=myloss(x,y,f,lr,w);

     [p] = lbfgs(g,mys,myy,B);
     dw= norm(p);
     wt1=w-eta0*p;
     
     f= wt1'*x;
     [v1,g1]=myloss(x,y,f,lr,wt1);
     
     sk1=wt1-w;
     yk1=g1-g;
     mys=[mys,sk1];
     myy=[myy,yk1];
     if size(mys,2)>m
         mys=mys(:,2:(m+1));
         myy=myy(:,2:(m+1));
     end
     
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
