function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    htheta = X * theta;
    theta0 = theta(1) - alpha / m * sum((htheta - y) .* X(:,1));
    theta1 = theta(2) - alpha / m * sum((htheta - y) .* X(:,2)); 
    theta = [theta0; theta1];
    
%abother method~~~~~~~~~~~~~~~~~~~~~~    
% predictions=X*theta;         % predictions of hypothesis on all m examples  
% errors=(predictions-y);      %  errors  
% sums=X'*errors;              %  
% delta=1/m *sums;  
%   
% % theta=theta-alpha*delta;     %iteration 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
