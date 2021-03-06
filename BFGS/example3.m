 %%f(x)= w_i * x_i
 %%f(x)= (sum_i )(alpha_i,k(x_i,x) )
 %%cost=loss+gamma_A*||f||_H+gamma_I*||f||_I
 %%optimization by SGD/l_BFGS, primal-dual method
 
  clear all;
  % load data
  load('/Users/Anna/Documents/MatlabCode/lapsvmp_v02/gui/2circles.mat');
  fun = @myloss;
  x=x';
  for i=1:2
      x(i,:)=(x(i,:)-min(x(i,:)))/((max(x(i,:)))-(min(x(i,:))));
  end
  %shafle the data
  clear xpos;clear xneg;
  xposi=1;xnegi=1; n = size(y,1);
  for i = 1:size(y,1)
      if y(i,1) == 1
         xpos(:,xposi)=x(:,i);
         xposi = xposi+1;
      else
         xneg(:,xnegi)=x(:,i);
         xnegi = xnegi+1;
      end
  end
  for i =1:min(size(xpos,2),size(xneg,2))
      xx(:,(2*i-1))=xpos(:,i);
      xx(:,(2*i)) = xneg(:,i);
      yy(:,(2*i-1))=1;
      yy(:,(2*i)) = -1;
  end
  
  options=make_options('gamma_I',0.1,'gamma_A',0.1,'Kernel', 'linear','NN',15,'KernelParam', 0.35);
 % Call L-BFGS starting from the origin
 %  [w,iter,loss] = lbfgs_svm(xx,yy,fun);
 % [w,iter,loss] = lbfgs_classic(xx,yy,fun);

% [deltaAlpha,deltaw] = lbfgs_dual_COCOA(xx,yy,fun,options);

% [w,iter,loss] = mylbfgs(xx,yy,fun);
% [xb,w,iter,loss] = lbfgs_classic_2(xx,yy,fun,options);
% [xb,w,iter,loss] = lapSGD(xx,yy,fun,options);

 m = 150;
 num = size(xx,1);              len = size(xx,2);
 alpha = zeros(1,m) + 0.01;     w = zeros(num,1) + 0.01;
% 
 for t =1:1
%     
% [deltaAlpha,deltaw] = localSDCA(w,alpha,xx,yy);
  [deltaAlpha,deltaw] = lbfgs_dual_COCOA(m,xx,yy,alpha,w,options);

 alpha = alpha + deltaAlpha';
 w = w + deltaw;
% 
 end

%test input data
  Thld = 0.00; %0.5 for 2circles;
%  myk=calckernel(options,xb',x');
%   y1=w'*myk';
  y1=w'*x;
  hold on;
  for i = 1:size(x,2)
      if y1(1,i)>= Thld;
         plot(x(1,i),x(2,i),'ro');
      else
         plot(x(1,i),x(2,i),'bd');
      end
  end



  
