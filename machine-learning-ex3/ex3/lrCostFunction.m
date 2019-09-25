function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
htheta=sigmoid(X*theta);         % test: 5X4*4X1 = 5X1
                                 % multi-class: 5000X401*401X1 = 5000*1 
                                 
J = 1 / m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
% =============================================================

grad = zeros(size(theta));
temp = theta;
temp(1) = 0;       % because we don't add anything for j = 0 
% grad = 1 / m * (X' * (htheta - y) + lambda * temp);      % fast alternatives: dJ/d(theta)=(4X5)*(5*1)
for i=1:length(grad)
    grad(i)=1/m*(X(:,i)'*(htheta-y) + lambda * temp(i));  % dJ/d(theta)=(1X5)*(5*1) for 4 times
end
end
