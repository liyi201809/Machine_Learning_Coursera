function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%Returns the cost and gradient for the collaborative filtering problem.

% Unfold the U and W matrices from params because the inputs were X(:) and Theta(:)
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
J = 1/2 * sum(sum((R.* ((X*Theta') - Y)).^2));
X_grad = (R .* (X*Theta' - Y)) * Theta;
Theta_grad = (R .* (X*Theta' - Y))' * X;

% With regularization
J = J + lambda/2 * (sum(sum(Theta.^2)) + sum(sum(X.^2)));
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;
% =============================================================
grad = [X_grad(:); Theta_grad(:)];

end
