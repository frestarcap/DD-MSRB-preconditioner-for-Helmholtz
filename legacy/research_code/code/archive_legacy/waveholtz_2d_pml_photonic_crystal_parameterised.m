function results = waveholtz_2d_pml_photonic_crystal_parameterised(cfg)
%WAVEHOLTZ_2D_PML_PHOTONIC_CRYSTAL_PARAMETERISED
% Parameter sweep and SVD diagnostic for the photonic-crystal waveguide.
%
% This is the second-step workflow for a new problem:
%   1) solve several material-speed samples c_pc,
%   2) compare WaveHoltz with direct Helmholtz for each sample,
%   3) inspect singular-value decay of the frequency-domain snapshots.
%
% Example:
%   r = waveholtz_2d_pml_photonic_crystal_parameterised(struct( ...
%       'Nphys', 80, 'c_pc_values', linspace(0.285, 0.305, 10)));

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end

    cfg = apply_pc_defaults(cfg);
    if isfield(cfg, 'c_pc_values') && ~isempty(cfg.c_pc_values)
        cfg.c_right_values = cfg.c_pc_values;
    elseif ~isfield(cfg, 'c_right_values') || isempty(cfg.c_right_values)
        cfg.c_right_values = cfg.c_pc_ref * linspace(0.96, 1.04, 12);
    end
    cfg.c_pc_values = cfg.c_right_values;

    do_svd_plots = cfg.do_svd_plots;
    cfg = rmfield(cfg, 'do_svd_plots');

    results = waveholtz_piecewise_scattering_standalone(cfg);
    results.problem_name = 'photonic_crystal_waveguide_parameterised';
    results.c_pc_values = results.c_right_values;
    results.parameter_svd = compute_snapshot_svd(results.u_wave, results.u_wave_phys);

    fprintf('\n=== Photonic-crystal parameter SVD ===\n');
    print_svd_summary(results.parameter_svd.full, 'full interior');
    print_svd_summary(results.parameter_svd.physical, 'physical crop');

    if do_svd_plots
        make_svd_plots(results.parameter_svd);
    end
end

function cfg = apply_pc_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'Nphys', 120);
    cfg = set_default(cfg, 'pc_num_rows', 3);
    cfg = set_default(cfg, 'pc_lattice_a', []);
    if isempty(cfg.pc_lattice_a)
        cfg.pc_lattice_a = cfg.L / (2 * cfg.pc_num_rows + 3);
    end
    cfg = set_default(cfg, 'pc_radius', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_x_margin', cfg.pc_lattice_a);
    cfg = set_default(cfg, 'omega', 2 * pi / (1.44 * cfg.pc_lattice_a));
    cfg = set_default(cfg, 'c_pc_ref', 1 / sqrt(11.4));
    cfg = set_default(cfg, 'source_type', 'point');
    cfg = set_default(cfg, 'point_source_profile', 'box');
    cfg = set_default(cfg, 'point_amplitude', 1.0);
    cfg = set_default(cfg, 'point_box_radius_x', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'point_box_radius_y', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'source_col_phys', round((cfg.Nphys + 1) / 2));
    cfg = set_default(cfg, 'source_row_phys', round((cfg.Nphys + 1) / 2));
    cfg = set_default(cfg, 'compute_incident_reference', false);
    cfg = set_default(cfg, 'material_model', 'photonic_crystal');
    cfg = set_default(cfg, 'do_svd_plots', true);
end

function parameter_svd = compute_snapshot_svd(u_full, u_phys)
    Sphys = reshape(u_phys, [], size(u_phys, 3));
    parameter_svd = struct();
    parameter_svd.full = svd_pack(u_full);
    parameter_svd.physical = svd_pack(Sphys);
end

function out = svd_pack(S)
    [~, Sigma, ~] = svd(S, 'econ');
    s = diag(Sigma);
    out = struct();
    out.singular_values = s(:);
    out.relative_singular_values = s(:) / max(s(1), 1e-30);
    out.rank_1e2 = nnz(out.relative_singular_values >= 1e-2);
    out.rank_1e4 = nnz(out.relative_singular_values >= 1e-4);
    out.rank_1e6 = nnz(out.relative_singular_values >= 1e-6);
    out.rank_1e8 = nnz(out.relative_singular_values >= 1e-8);
end

function print_svd_summary(svd_info, label)
    fprintf('%s singular values: %s\n', label, mat2str(svd_info.singular_values(:).', 4));
    fprintf('%s relative ranks r1e-2/r1e-4/r1e-6/r1e-8 = %d / %d / %d / %d\n', ...
        label, svd_info.rank_1e2, svd_info.rank_1e4, svd_info.rank_1e6, svd_info.rank_1e8);
end

function make_svd_plots(parameter_svd)
    figure('Name', 'Photonic-crystal Parameter Snapshot SVD');
    hold on;
    semilogy(parameter_svd.full.relative_singular_values, 'o-', 'LineWidth', 1.3, 'DisplayName', 'full interior');
    semilogy(parameter_svd.physical.relative_singular_values, 's-', 'LineWidth', 1.3, 'DisplayName', 'physical crop');
    hold off;
    grid on;
    xlabel('mode index');
    ylabel('relative singular value');
    title('Photonic-crystal waveguide parameter SVD');
    legend('Location', 'best');
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
