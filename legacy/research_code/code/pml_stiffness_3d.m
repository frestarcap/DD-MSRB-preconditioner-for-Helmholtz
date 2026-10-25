function K = pml_stiffness_3d(nx, ny, nz, hx, hy, hz, pml)
%PML_STIFFNESS_3D Vectorized seven-point Helmholtz/PML stiffness assembly.

    Kx = directional_operator(pml.s_x, pml.s_x_half, nx, hx);
    Ky = directional_operator(pml.s_y, pml.s_y_half, ny, hy);
    Kz = directional_operator(pml.s_z, pml.s_z_half, nz, hz);

    Ix = speye(nx);
    Iy = speye(ny);
    Iz = speye(nz);
    K = kron(Iz, kron(Iy, Kx)) ...
        + kron(Iz, kron(Ky, Ix)) ...
        + kron(Kz, kron(Iy, Ix));
end

function K1 = directional_operator(s_node, s_half, n, h)
    a = 1 ./ s_node(2:n + 1);
    a_p = 1 ./ s_half(2:n + 1);
    a_m = 1 ./ s_half(1:n);

    diagonal = -(a_p + a_m) .* a / (h * h);
    upper = a(1:end - 1) .* a_p(1:end - 1) / (h * h);
    lower = a(2:end) .* a_m(2:end) / (h * h);

    idx = (1:n).';
    K1 = sparse([idx; (1:n - 1).'; (2:n).'], ...
        [idx; (2:n).'; (1:n - 1).'], ...
        [diagonal; upper; lower], n, n);
end
