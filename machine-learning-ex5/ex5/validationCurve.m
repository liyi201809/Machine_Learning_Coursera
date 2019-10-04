function [lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval,Xte, yte)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    theta = trainLinearReg(X, y, lambda);
    error_train(i) = (sum((X * theta - y).^2)) / (length(y)*2); 
    error_val(i) = (sum((Xval * theta - yval).^2)) / (length(yval)*2);
    % Optional tasks
    error_test(i) = (sum((Xte * theta - yte).^2)) / (length(yte)*2);
end
minm = min(error_val);
minindex = find(error_val==minm);
fprintf('The minimum val error is %f with lambda value of %f\n', minm,lambda_vec(minindex));
minm = min(error_test);
minindex = find(error_test==minm);
fprintf('The minimum test error is %f with lambda value of %f\n', minm,lambda_vec(minindex));


% =========================================================================

end
