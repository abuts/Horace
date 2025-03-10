function [ub, umat] = ubmatrix (u, v, b)
% Calculate UB matrix that transforms components of a vector given in r.l.u.
% into the components in an orthonormal frame defined by the two vectors
% u and v (each given in r.l.u)
%
%   >> ub = ubmatrix (u, v, b)
%
%   >> [ub, mess, umat] = ubmatrix (u, v, b)    % full syntax
%
% Input:
% -------
%   u, v    Two vectors expressed in r.l.u.
%   b       B-matrix of Busing and Levy (as calulcated by function bmat)
%
% Output:
% -------
%   ub      UB matrix; empty if there is a problem
%   mess    Error message; empty if all OK
%   umat    U matrix
%
% The orthonormal frame defined by vectors u and v is:
%   e1  parallel to u
%   e2  in the plane of u and v, with a +ve component along v
%   e3  perpendicular to u and v
%
% Use the matrix ub to convert components of a vector as follows:
%
%   Vuv(i) = UB(i,j) Vrlu(j)
%
% Also:
%
%   Vuv(i) = U(i,j) Vcryst(j)   % NOTE: inv(U) == U'


% Original author: T.G.Perring
%
%
% Horace v0.1   J. van Duijn, T.G.Perring

try
    u = u(:);   % convert to column vector
    v = v(:);   % convert to column vector

    uc = b*u;   % Get u, v in crystal cartesian coordinates
    vc = b*v;   

    e1 = uc/norm(uc);
    e3 = cross(uc,vc)/norm(cross(uc,vc));
    e2 = cross(e3,e1);

    umat = [e1';e2';e3']/det([e1';e2';e3']);    % renormalise to ensure determinant = 1
    ub = umat*b;
catch
    error('HORACE:ubmatrix:invalid_argument', ...
        'Problem calculating UB matrix. Check u (%s) and v (%s)are not parallel',...
		mat2str(u'),mat2str(v'));
end

