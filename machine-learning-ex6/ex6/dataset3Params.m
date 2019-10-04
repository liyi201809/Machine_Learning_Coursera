function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
C = 1;
sigma = 0.3;
% ====================== YOUR CODE HERE ======================
%You can use svmPredict to predict the labels on the cross
%validation set. For example, predictions = svmPredict(model, Xval);
%will return the predictions on the cross validation set.
%Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
clist = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmalist = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
predictlist =[];
for i=1:length(clist)
    for j =1:length(sigmalist)
        model= svmTrain(X, y, clist(i), @(x1, x2) gaussianKernel(x1, x2, sigmalist(j)));
        predictions = svmPredict(model, Xval);
        predictlist(i,j) = mean(double(predictions ~= yval));
    end
end
%predictlist
M = min(predictlist(:));
[xindex,yindex] = find(predictlist==M);
% only select one index, even if there are several minimums
C = clist(xindex(1));   
sigma = sigmalist(yindex(1));

end
