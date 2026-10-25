function result = test_pc_mask_vectorization()
%TEST_PC_MASK_VECTORIZATION Compare local-box and full-grid PC masks.

    cfg = struct('setup_only', true, 'pc_points_per_a', 12, ...
        'train_params', [0.28, 0.31], 'test_params', 0.295, ...
        'do_plots', false, 'save_plots', false, 'save_benchmark', false, ...
        'parallel_train', false, 'parallel_test', false);
    setup = photonic_crystal_waveguide_correction_rom(cfg);
    prob = setup.problem;
    cfg = setup.cfg;

    a = cfg.pc_lattice_a;
    radius = cfg.pc_radius;
    x_centers = cfg.pc_x_margin:a:(cfg.Lx - cfg.pc_x_margin);
    offsets = (1:cfg.pc_num_rows) * a;
    y_mid = 0.5 * cfg.Ly;
    y_centers = [y_mid - fliplr(offsets), y_mid + offsets];
    y_centers = y_centers(y_centers > radius & y_centers < cfg.Ly - radius);
    [X, Y] = ndgrid(prob.x_int, prob.y_int);
    legacy = false(prob.nx, prob.ny);
    for ix = 1:numel(x_centers)
        for iy = 1:numel(y_centers)
            if strcmp(cfg.pc_rod_shape, 'circle')
                cell_mask = (X - x_centers(ix)).^2 + (Y - y_centers(iy)).^2 <= radius^2;
            else
                cell_mask = abs(X - x_centers(ix)) <= cfg.pc_square_halfwidth ...
                    & abs(Y - y_centers(iy)) <= cfg.pc_square_halfwidth;
            end
            legacy = legacy | cell_mask;
        end
    end
    physical = X >= 0 & X <= cfg.Lx & Y >= 0 & Y <= cfg.Ly;
    legacy = legacy & physical;

    mismatch = nnz(xor(prob.chi_right_grid, legacy));
    assert(mismatch == 0, 'PC material masks differ at %d nodes.', mismatch);
    result = struct('mismatch_count', mismatch, 'material_nodes', nnz(legacy));
    fprintf('PC mask mismatch count %d over %d material nodes.\n', ...
        mismatch, result.material_nodes);
end
