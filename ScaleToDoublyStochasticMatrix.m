function [B, C, err]= ScaleToDoublyStochasticMatrix(A, tol)
% Scale to doubly stochastic matrix with the alternative normalizing method
% in [1]
% This code is a part of [2], 
%
% [1] S. R, A relationship between arbitrary positive matrices and doubly
% stochastic matrices, The annals of mathematical statistics, vol. 35, 
% no. 2, pp. 876–879, 1964.
% [2] L. He, H. Zhu, T. Zhang, Y. Guan and H. Zhang, K-means via Doubly  
% Stochastic Scaling, submitted to TSP
%
% Input:
%           A:          non-negtive matrix, symmetric and zero diagonal
% Output:
%           B:          doubly stochasitc matrix, B = CAC
%           C:          diagnal matrix, B = CAC
%           err:        sum(B-1).^2 + sum(B'-1).^2
%
% heli@gdut.edu.cn

B = A;
if nargin==1
    tol = 1e-4;
end
maxiter = 5000;
counter = 0;
err = tol*100;
errList = [];
l = ones(size(B,1),1);

isPlot = false;
% isPlot = true;

while err>tol && counter<maxiter
    for i=1:size(B,1)
        k = sum(B(i,:));
        l(i) = l(i)/k;
        B(i,:) = B(i,:)/k;
        B(:,i) = B(:,i)/k;
    end
    for j=1:size(B,2)
        k = sum(B(:,j));
        l(j) = l(j)/k;
        B(:,j) = B(:,j)/k;
        B(j,:) = B(j,:)/k;
    end
    err = sum( (sum(B)-1).^2 + (sum(B')-1).^2 )^.5;
    errList = [errList err];
    counter = counter+1;
end
if isPlot
    figure(11);plot(1:length(errList),errList,'-b*');
end
C = diag(l);