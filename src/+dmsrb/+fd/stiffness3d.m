function K = stiffness3d(n, h, pml)
%STIFFNESS3D Seven-point PML Laplacian on the n(1)-by-n(2)-by-n(3) interior grid.
%   pml from dmsrb.fd.pml_profile(n, ...); Dirichlet data on the outer boundary.
    Kx = directional(pml.s{1}, pml.s_half{1}, n(1), h(1));
    Ky = directional(pml.s{2}, pml.s_half{2}, n(2), h(2));
    Kz = directional(pml.s{3}, pml.s_half{3}, n(3), h(3));
    Ix = speye(n(1)); Iy = speye(n(2)); Iz = speye(n(3));
    K = kron(Iz, kron(Iy, Kx)) + kron(Iz, kron(Ky, Ix)) + kron(Kz, kron(Iy, Ix));
end

function K1 = directional(s_node, s_half, n, h)
    a = 1 ./ s_node(2:n + 1);
    a_p = 1 ./ s_half(2:n + 1);
    a_m = 1 ./ s_half(1:n);
    diagonal = -(a_p + a_m) .* a / (h * h);
    upper = a(1:end - 1) .* a_p(1:end - 1) / (h * h);
    lower = a(2:end) .* a_m(2:end) / (h * h);
    idx = (1:n).';
    K1 = sparse([idx; (1:n - 1).'; (2:n).'], [idx; (2:n).'; (1:n - 1).'], ...
        [diagonal; upper; lower], n, n);
end
