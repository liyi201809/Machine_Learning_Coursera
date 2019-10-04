function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%               You should set J to the cost and grad to the gradient.
J = (sum((X*theta - y).^2) + lambda*sum(theta(2:end,:).^2))/(2*m);

[exap,feature]=size(X);

grad(1) = sum((X*theta - y).*X(:,1))/m;  % for j = 0
for j = 2:feature                      % for j >=1 
    grad(j) = sum((X*theta - y).*X(:,j))/m + lambda*theta(j)/m;
end

grad = grad(:); % one row to one column

end
