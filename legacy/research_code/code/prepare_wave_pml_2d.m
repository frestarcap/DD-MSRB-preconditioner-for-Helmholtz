function pml = prepare_wave_pml_2d(pml, nx, ny, dt)
%PREPARE_WAVE_PML_2D Cache static PML arrays used by every time step.

    pml.sx_full = repmat(pml.sigma_x, 1, ny + 2);
    pml.sy_full = repmat(pml.sigma_y.', nx + 2, 1);
    pml.sigma_x_vx = 0.5 * (pml.sx_full(1:end-1, :) + pml.sx_full(2:end, :));
    pml.sigma_y_vy = 0.5 * (pml.sy_full(:, 1:end-1) + pml.sy_full(:, 2:end));
    pml.sigu = pml.sx_full + pml.sy_full;
    pml.sigma_xy = pml.sx_full .* pml.sy_full;

    pml.vx_den = 1 + 0.5 * dt * pml.sigma_x_vx;
    pml.vx_decay = (1 - 0.5 * dt * pml.sigma_x_vx) ./ pml.vx_den;
    pml.vx_drive = dt ./ pml.vx_den;
    pml.vy_den = 1 + 0.5 * dt * pml.sigma_y_vy;
    pml.vy_decay = (1 - 0.5 * dt * pml.sigma_y_vy) ./ pml.vy_den;
    pml.vy_drive = dt ./ pml.vy_den;
    pml.u_den = 1 + 0.5 * dt * pml.sigu(2:end-1, 2:end-1);
    pml.u_decay = (1 - 0.5 * dt * pml.sigu(2:end-1, 2:end-1)) ./ pml.u_den;
    pml.u_drive = dt ./ pml.u_den;
end
