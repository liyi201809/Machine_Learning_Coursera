function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. 
%   You should return a matrix centroids, where each row of centroids is 
%   the mean of the data points
%   assigned to it.
% Useful variables
[m n] = size(X);
% You need to return the following variables correctly.
centroids = zeros(K, n);     % K x 2
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
% Note: You can use a for-loop over the centroids to compute this.

% xy_temp = zeros(K,n+1);
% for i=1:m
%     xy_temp(idx(i),1) = xy_temp(idx(i),1) + X(i,1);
%     xy_temp(idx(i),2) = xy_temp(idx(i),2) + X(i,2);
%     xy_temp(idx(i),3) = xy_temp(idx(i),3) + 1;      % count how many examples are assigned to each centi.
% end
% xy_temp(:,1:2) = xy_temp(:,1:2)./xy_temp(:,3);
% centroids = xy_temp(:,1:2); 

for k=1:K % for-loop over the centroids 
   centroids(k, :) = mean(X(idx==k, :));
end
