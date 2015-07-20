
function [deltaAlpha,deltaw] = localSDCA(w,alpha,x,y)

lambda = 0.1;
len = size(x,1);

grad = ( y .* (x' * w)' - 1.0 ) * ( lambda * len);
% for i=1:size(x,2)
%     wx(:,i) = w .* x(:,i);
%     g1(:,i) = ( y(:,i) * wx(:,i) - 1.0 ) * ( lambda * len);
% end


xnorm = norm(x).^ 2;
qii= xnorm;
newAlpha = 1.0;

if qii ~= 0.0
   newAlpha = alpha - grad./qii;
   newAlpha = max(0, newAlpha);
   newAlpha = min(1.0, newAlpha);
end

update = x * ( y .* (newAlpha-alpha) )'/(lambda * len);

deltaw = update;
deltaAlpha = newAlpha - alpha;
