function [cen, err, k] = SolveCentroid(pt,C)
% Solve the centroid of the unknown sphere given diagonal matrix C and input
% data pt according to Theorem 6 of [1]
%
%   d_ii = 4r^2/( 4r^2+norm(p)^2 )              (1)
%
% Theorem 7 denotes there exists a sphere S(z,r) on which the EDM A'=k*B,
% where B is the DS matrix that A scaling to (B=CAC). Since we know the
% scaling matrix C, we want to solve the sphere centroid z=[x1,...,xd,r]
% and k ( r=k^.5*r_B, where r_B = (.5/n)^.5 is known but k unknown ). 
% In (1), d_ii is D(i,i), D = C*k^.5, where k is unknown, r is the radius
%
% Suppose the centroid of sphere is [x1,x2,...,x_d,r], where
% x = [x1,x2,...,x_d] and r are unknown. First move the coordinates and let
% the centroid being above the original, [x1,...x_d]->[0,...,0], and the pt
% will be [pt-x]; then using Theorem 6, notice that p_i here is pt_i - x_i,
% and d_ii is C(i,i)*k^.5 and r = k^.5*(.5/numData)^.5. To solve x, we have
% the multivariate quadratic equotations:
%
% ||(pt-x)||^2 = 4r^2/dii - 4r^2                (2)
%
% take r=rB*k^.5 and dii=cii*k^.5 into (2) we got
%
% ||(pt-x)||^2 = 4*rB^2*k^.5/cii - 4*rB^2*k     (3)
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
%           pt:             n*m, n data described in m variables
%           C:              scaling matrix that scale EDM of pt A into
%                           doubly stochastic matrix B, B=CAC
% Output:
%           cen:            centroid of the sphere of Theorem 7
%           err:            error of centroid
%           k:              E=tB, where E is the EDM of pt on sphere
% Used Variables:
%           sphereC:        sphere pt inverse stereographic projected by,
%                           centered at [x1,...,xd,rB], rB as radius
%           sphereD:        sphere pt inverse stereographic projected by,
%                           then expanded radially with r=rB*k^.5; centered 
%                           at [x1,...,xd,rB], r as radius
%           r:              radius of sphere of Theorem 7
%           rB:             radius of the sphere of DS matrix B, 
%                           rB=(.5/n)^.5, r=k.^5*rB
%           C:              diagonal matrix scaling A to B, B=CAC
%           D:              diagonal matrix scaling A to the sphere of Thr7
%           k:              scalar in Theorem 7, C=D/k^.5, r=k.^5*rB
%
% heli@gdut.edu.cn

[n,dim] = size(pt);

r = (.5/n)^.5;
% b = 4*r^2./diag(C) - 4*r^2;

A = pt;

p = zeros(1,dim);

% solve the equotations || X-A ||^2 = b
% method 1, using fsolve
opt=optimset('MaxFunEvals',5000,'MaxIter',1000,'TolFun',1e-8);
x = fsolve( @(x) MQEs(x,A,C), [p 0] , opt);
k = exp( x(end) );
cen = x;
cen(end) = r*k^.5;

npt = pt-repmat(x(1:dim),n,1);
err = sum( (4*r^2*k./(4*r^2*k+diag(npt*npt'))/k^.5-diag(C)).^2 );


function y = MQEs(p,A,C)
% Solutions to multivariate quadratic equotations:
%
% (x1-a11)^2 + (x2-a12)^2 + ... + (xd-a1d)^2 = b1       (1)
%                   ...
% (x1-an1)^2 + (x2-an2)^2 + ... + (xd-and)^2 = bn       (n)
%
% or,
%
%   sum( (1*x - A).^2, 2 ) = b                          (a)
%
% where 1 is ones(n,1), x is a row vector, [x1,x2,...,xd], b=[b1,...,bn]',
% bi = 4r^2/(dii*k) - 4r^2
% Input:
%           p:          1*d+1 vector, initializing of x, d is dim of
%                       sphere, p(end) is the k 
%           A:          n*d matrix, aij in A above
%           C:          n*n vector, dialgonal matrix that B = CAC, to 
                        % compute bi in (1)~(n) above, D = C*k
% Output:
%           q:          solutions to (a)
% Notice: we will use fsolve function of maltab to solve this function, as:
%   x = fsolve('MQEs',)

[numData,dim] = size(A);
x = p(1:dim);
k = exp( p(end) );
rB = (.5/numData)^.5;
b = 4*rB^2*k^.5./diag(C)-4*rB^2*k;

y = sum( (repmat(x,numData,1) - A ).^2 ,2 ) - b;
y = y'*y;