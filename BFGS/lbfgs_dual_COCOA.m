%%%COCOA:solving using LBFGS
%%%
function [deltaalpha,deltaw] = lbfgs_dual_COCOA(m,x0,y0,w,ww,options)
 
 %m = size(x0,2)/2;
 eta0=0.1;
 lr=0.01;
 gamma=0.1;
 delta=0.01;
 dw=1;
       
lambda = 0.1;
len = size(x0,1);

num =  size(x0,2); dim=size(x0,1);
  
index=floor( rand(1,1)*(m) );
index=min(index, (m));
x=x0(:,(index+1):(index+m));    y=y0(1,(index+1):(index+m));
     
 w0=zeros(m,1); 
 w=w0+delta;
 g0=0;
 B=eye(m)*delta;

 Kerl=calckernel(options,x', x');
 L=laplacian(options,x');
       
 f1= x0(:,1:m);  f2= y0(1,1:m);
 
 g = ( f2 .* (f1' * ww)' - 1.0 ) * ( lambda * len)/norm(f1);
 g=g';
 mys=w-w0;
 myy=g-g0;
 
     index=floor( rand(1,1)*(num) );
     index=min(index, (num-m));
     f1=x0(:,(index+1):(index+m)); f2=y0(1,(index+1):(index+m));

     [p] = lbfgs(g,mys,myy,B);
     dw= norm(p);
     
     wt1=w-eta0*p;

g1 = ( f2 .* (f1' * ww)' - 1.0 ) * ( lambda * len)/norm(f1);
g1=g1';
  
update = f1 * ( f2' .* (wt1-w) ) /(lambda * len);    
   
deltaalpha = wt1-w;
deltaw = update;

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

end
