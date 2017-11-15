function spt = inverseStereographicProjection(pt, radius, centroid)
% Inverse stereographic project. Project points on (d-1) plane to d sphere
% by Lemma 5 of [1].
%
% Given a sphere S with centroid (x,y,r) and the radius r, and data pt on 
% x=0 plane. Let l be the line linking the north pole of S and pt, and the
% intersecting points of l and S is spt. Given S and pt, this function
% outputs spt.
%
% This code is a part of [2].
%
% [1]  J. C. R, M. R. D, and T. M. W, “On the diagonal scaling of euclidean
% distance matrices to doubly stochastic matrices,” Linear Algebra and its
% Applications, vol. 397, pp. 253–264, 2005.
% [2] L. He, H. Zhu, T. Zhang, Y. Guan and H. Zhang, K-means via Doubly   
% Stochastic Scaling, submitted to TSP
%
% Input:
%       pt:         points on (d-1) plane
%       radius:     radius of sphere, shpere is S(centroid, radius)
%       centroid:   centroid of shpere, shpere is S(centroid, radius)
% Output:
%       spt:        points on d sphere
%
% heli@gdut.edu.cn

[n,dim] = size(pt);
if nargin<2
    radius = (.5/n)^.5;
    centroid = zeros(1,dim);
elseif nargin<3
    centroid = zeros(1,dim);
end
% since Lemma 5 requires the sphere lies on [0,0,...,0,r], so move
% all coordinates and let the centroid being the original
cenOffset = -centroid;
cenOffset(dim+1) = 0;

% move data with offset, now, centroid of sphere is [0,...0,r]
A = pt+repmat(cenOffset(1:dim),n,1);

% Lemma 5
t  =1./( 4*radius^2+sum(A.^2,2) );  % denominator of Lemma 5

spt = 4*radius^2*A.*repmat(t,1,dim);    % dim: 1~d
spt(:,dim+1) = 2*radius*sum(A.^2,2).*t; % dim: d+1

spt = spt-repmat(cenOffset,n,1);