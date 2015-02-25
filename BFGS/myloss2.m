function [v,g] = myloss2(x,y,f1,f2,lr,w,K,L,options)
  %x:dim*n; y:1*n; s:dim*1; v:1*1; g:1*n;
  %w:1*n;
  dim=size(x,1); n=size(x,2);
  num=size(f1,2);
  g= lr*w;
  sumloss = zeros(1,n);   
  v=0;
  myk=calckernel(options,x',f1'); %1*n
  
  f=w'*myk'; 
 
  l1=options.gamma_A*K*w;
  l2=options.gamma_I*K*L*K*w;
   

   for i=1:num
       d(1,i)=1-f(1,i)*f2(1,i);
       v= v + d(1,i);
       if d(1,i)>=0
          sumloss(1,:)=sumloss(1,:) - f2(:,i)*myk(i,:);
      end
  end

  g = l1 + (sumloss)' + l2;
