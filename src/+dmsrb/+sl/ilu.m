function [Msolve, info] = ilu(Ashift, droptol)
%ILU Shifted-Laplacian preconditioner by incomplete LU (ILUTP, no MILU).
%   [Msolve, info] = dmsrb.sl.ilu(A + 1i*beta*M, droptol) returns the action
%   Msolve(r) = U\(L\r); info.factor_bytes is the storage of the factors.
    if nargin < 2, droptol = 1e-3; end
    [L, U] = ilu(Ashift, struct('type', 'ilutp', 'droptol', droptol, 'thresh', 1.0, 'milu', 'off'));
    Msolve = @(r) U \ (L \ r);
    stats = whos('L', 'U');
    info = struct('factor_bytes', sum([stats.bytes]));
end
