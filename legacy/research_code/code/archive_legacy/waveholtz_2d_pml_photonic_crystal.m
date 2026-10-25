function results = waveholtz_2d_pml_photonic_crystal(cfg)
%WAVEHOLTZ_2D_PML_PHOTONIC_CRYSTAL
% Fixed photonic-crystal waveguide check: WaveHoltz vs direct Helmholtz.
%
% This is the first-step workflow for a new problem.  It reuses the
% variable-speed WaveHoltz/Helmholtz implementation, but replaces the
% half-space material mask by two rows of photonic-crystal cylinders above
% and below a central source channel.
%
% Main physical parameter:
%   cfg.c_pc  - wave speed inside the crystal rods.
%
% Example:
%   r = waveholtz_2d_pml_photonic_crystal(struct('pc_points_per_a', 10));

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end

    cfg = apply_pc_defaults(cfg);
    if isfield(cfg, 'c_right_values')
        cfg = rmfield(cfg, 'c_right_values');
    end
    cfg.c_right_values = cfg.c_pc;

    results = waveholtz_piecewise_scattering_standalone(cfg);
    results.problem_name = 'photonic_crystal_waveguide_fixed';
    results.c_pc = cfg.c_pc;
    results.c_pc_values = results.c_right_values;
end

function cfg = apply_pc_defaults(cfg)
    cfg = set_default(cfg, 'pc_lattice_a', 1.0);
    cfg = set_default(cfg, 'pc_num_cols', 21);
    cfg = set_default(cfg, 'pc_num_rows', 7);
    cfg = set_default(cfg, 'Lx', cfg.pc_num_cols * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'Ly', (2 * cfg.pc_num_rows + 1) * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'L', cfg.Ly);
    cfg = set_default(cfg, 'pc_points_per_a', 10);
    cfg = set_default(cfg, 'Nphys_x', round(cfg.Lx / cfg.pc_lattice_a * cfg.pc_points_per_a));
    cfg = set_default(cfg, 'Nphys_y', round(cfg.Ly / cfg.pc_lattice_a * cfg.pc_points_per_a));
    cfg = set_default(cfg, 'Nphys', cfg.Nphys_y);
    cfg = set_default(cfg, 'npml', 2 * cfg.pc_points_per_a);
    cfg = set_default(cfg, 'pc_radius', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_rod_shape', 'circle');
    cfg = set_default(cfg, 'pc_square_halfwidth', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_x_margin', 0.5 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'lambda_over_a', 2.0);
    cfg = set_default(cfg, 'omega', 2 * pi / (cfg.lambda_over_a * cfg.pc_lattice_a));
    cfg = set_default(cfg, 'c_pc', 1 / sqrt(11.4));
    cfg = set_default(cfg, 'source_type', 'point');
    cfg = set_default(cfg, 'point_source_profile', 'box');
    cfg = set_default(cfg, 'point_amplitude', 1.0);
    cfg = set_default(cfg, 'point_box_radius_x', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'point_box_radius_y', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'source_col_phys', round((cfg.Nphys_x + 1) / 2));
    cfg = set_default(cfg, 'source_row_phys', round((cfg.Nphys_y + 1) / 2));
    cfg = set_default(cfg, 'compute_incident_reference', false);
    cfg = set_default(cfg, 'material_model', 'photonic_crystal');
    cfg = set_default(cfg, 'wh_solver', 'custom_gmres');
    cfg = set_default(cfg, 'wh_initial_guess', 'zero');
    cfg = set_default(cfg, 'wh_tol', 1e-6);
    cfg = set_default(cfg, 'wh_gmres_restart', 60);
    cfg = set_default(cfg, 'wh_gmres_maxit', 10);
    cfg = set_default(cfg, 'wh_auto_refine_against_helm', true);
    cfg = set_default(cfg, 'wh_helm_align_tol', 1e-3);
    cfg = set_default(cfg, 'wh_auto_refine_max_rounds', 2);
    cfg = set_default(cfg, 'wh_auto_refine_maxit_factor', 2);
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
