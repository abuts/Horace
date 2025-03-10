function theta = rotmat_to_rotvec (rotmat, varargin)
% Convert rotation matrices to rotation vectors
%
% The rotation matrix relates the components of a vector expressed in a
% coordinate frame S to those in a frame S' by v'(i) = R(i,j) v(j).
%
%   >> theta = rotmat_to_rotvec(rotmat)
%   >> theta = rotmat_to_rotvec(rotmat, algorithm)
%
% Input:
% ------
%   rotmat      Rotation matrices: 3x3 (single matrix) or 3 x 3 x m array
%               Relates the components of a vector v expressed in the two
%              coordinate frames by:
%                   v'(i) = R(i,j) v(j)
%Optional:
%   algorithm   Method for algorithm
%                 =0  Fast method due to T.G.Perring (default)
%                 =1  Generic method based on matrix exponentiation
%
% Output:
% -------
%   theta      Rotation vector: vector length 3 (single rotation vector)
%              or 3 x m array (m is the number of vectors).
%               A rotation vector defines the orientation of a coordinate frame
%              S' with respect to a frame S by rotation about a unit vector
%              (n(1),n(2),n(3)) in S by angle THETA (in a right-hand sense).
%               This defines a 3-vector:
%                 (THETA(1), THETA(2), THETA(3)) where THETA(i) = THETA*n(i).
%
%               In this function the units are degrees.
%
% Note:
%   rotmat_to_rotvec       This function   -- returns Rotation vector in degrees
%   rotmat_to_rotvec_rad   Sister function -- returns Rotation vector in radians

theta = rotmat_to_rotvec_rad (rotmat, varargin{:});
theta = rad2deg(theta);  % convert to degrees

