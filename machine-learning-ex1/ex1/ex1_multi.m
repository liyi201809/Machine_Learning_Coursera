%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).

%% Initialization
clear ; close all; clc

%% ================ Part 1: Feature Normalization ================
% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X_norm, mu, sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X_norm];
%% ================ Part 2: Gradient Descent ================
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
% Hint: At prediction, make sure you do the same feature normalization.
%
% init the parameters
fprintf('Running gradient descent ...\n');
alpha = 0.01;
num_iters = 400;

% Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display computed gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% part2 application: Estimate the price of a 1650 sq-ft, 3 br house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = ([1, 1650, 3] - [0 mu]) ./ [1 sigma] * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);


%% ================ Part 3: Normal Equations ================
% Load Data
fprintf('Solving with normal equations...\n');

data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta1 = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta1);
fprintf('\n');

%% part3 application: Estimate the price of a 1650 sq-ft, 3 br house
price = [1 1650 3] * theta1;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

