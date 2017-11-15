function demo
% Demo of Doubly Stochastic Scaling k-menas in [1].
% We suggest to replace the Euclidean distance matrix in k-means with its
% nearest doubly stochastic matrix, and show the theoretical foudation of
% using the DS matrix.
%
% [1] L. He, H. Zhu, T. Zhang, Y. Guan and H. Zhang, K-means via Doubly
% Stochastic Scaling, submitted to TSP
%
% heli@gdut.edu.cn

%% 1 Initialization
clc
clear
close all

% evalation method: ARI, RI, Accuracy, NMI
addpath('./Evaluation');

% dataset path
addpath('./Dataset');

% please select one dataset
% load Iris.mat; % data, labels
load Wine.mat;
% load SVMGuide4.mat;
% load LiverDisorders.mat;
% load Ionosphere.mat;
% load Vowel.mat;

% number of classes
numCentrs = numel(unique(labels));

% Euclidean Distance Matrix
A = pdist2(data,data).^2;

%% 2 Doubly Stochastic Scaling
[~, C, ~] = ScaleToDoublyStochasticMatrix(A);
[cen, ~, ~] = SolveCentroid(data,C);

% Inverse Stereographic Projections P
P = inverseStereographicProjection(data, cen(end), cen(1:end-1));

%% 3. Comparison on Clustering
% max. of iterations
maxIter = 10;

% results to save, 4 metrics and maxIter iterations
resA = zeros(maxIter, 4); % original k-means
resE = zeros(maxIter, 4); % ours

for iter=1:maxIter
    %% 3.1. Original k-means: with A 
    yA = kmeans(data,numCentrs,'EmptyAction','singleton');
    [ARIA,RIA,~,~,AccA,NMIA]=RandIndex(yA,labels);
    resA(iter,:) = [ARIA, RIA, AccA, NMIA];
    
    %% 3.2. Our k-means: with E
    yE = kmeans(P,numCentrs,'EmptyAction','singleton');
    [ARIE,RIE,~,~,AccE,NMIE]=RandIndex(yE,labels);
    resE(iter,:) = [ARIE, RIE, AccE, NMIE];
end

% scale Accuracy in [0,1]
resA(:,3) = resA(:,3)/100;
resE(:,3) = resE(:,3)/100;

mnA = mean(resA);
mnE = mean(resE);
bar([mnA' mnE']);
legend('k-means','Ours');
set(gca,'XTickLabels',{'ARI','RI','Accuracy','NMI'});