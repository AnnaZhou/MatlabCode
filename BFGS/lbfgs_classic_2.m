%%%
%%%
function [x,w,iter,loss] = lbfgs_classic_2(x0,y0,fun,options)
 
 m = 150;
 eta0=0.1;
 lr=0.01;
 gamma=0.1;
 delta=0.01;
 dw=1;
 batchSize = 30; 
 num =  size(x0,2); dim=size(x0,1);
  
     index=floor( rand(1,1)*(num-m) );
     index=min(index, (num-m));
     x=x0(:,(index+1):(index+m)); y=y0(1,(index+1):(index+m));
     

 w0=zeros(m,1); 
 w=w0+delta;
 g0=0;
 B=eye(m)*delta;

       Kerl=calckernel(options,x', x');
       L=laplacian(options,x');
       
 f1= x0(:,1);  f2= y0(1,1);
 [v,g]=myloss2(x,y,f1,f2,lr,w,Kerl,L,options);
 mys=w-w0;
 myy=g-g0;
 
 for iter = 1:1000
     index=floor( rand(1,1)*(num-batchSize) );
     index=min(index, (num-batchSize));
      f1=x0(:,(index+1):(index+batchSize)); f2=y0(1,(index+1):(index+batchSize));
     
     [v,g]=myloss2(x,y,f1,f2,lr,w,Kerl,L,options);

     [p] = lbfgs(g,mys,myy,B);
     dw= norm(p);
     wt1=w-eta0*p;
     
     [v1,g1]=myloss2(x,y,f1,f2,lr,wt1,Kerl,L,options);
     
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
