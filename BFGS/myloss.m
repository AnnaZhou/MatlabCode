%% the loss=1-<w,x>*y
%% the gradient: g=lw+dl/dw
function [v,g] = myloss(x,y,s,lr,w)
  %x:p*1; y:1*1; s:1*1; v:1*1; g:p*1;
  p=size(x,1); n=size(x,2);
  g= lr*w;
  sumloss = zeros(p,1);   
  v=0;
  for i=1:n
      d(1,i)=1-s(1,i)*y(1,i);
      if d(1,i)>=0
          v= v + d(1,i);
         sumloss=sumloss-x(:,i)*y(1,i);
      end
  end

  g = g + (sumloss./n);
