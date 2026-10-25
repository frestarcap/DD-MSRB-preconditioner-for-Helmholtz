function run_appA(stage, omega)
%RUN_APPA Appendix A: composition of the residual after the ROM steps.
%   run_appA('probe', omega) loads the Experiment 3 models at omega and, for the
%   three common queries, measures the residual r after the k = 3 ROM steps: its
%   transverse roughness and its contraction by the components of one shifted-
%   Laplacian MG V-cycle.  At mu = 0.317 it also applies 40 SL steps to r/||r||
%   and stores |r| and the remaining residual on the plane z = 1/2.
%   run_appA('export') writes results/paper/appendixA (schema of paper_data/appendixA).
    switch stage
        case 'probe', probe(exp3_config(omega));
        case 'export', export_paper();
        otherwise, error('Unknown stage %s.', stage);
    end
end

function probe(c)
    maxNumCompThreads(c.online_threads);
    P = dmsrb.waveguide.problem(c.omega);
    part = dmsrb.waveguide.partition(P);
    folder = fullfile(results_root(), 'exp3', sprintf('omega%g', c.omega));
    rows = table();
    for mu = c.test
        A = dmsrb.waveguide.operator(P, mu);
        [M, info] = dmsrb.sl.mg(A + 1i * c.beta * spdiags(dmsrb.waveguide.mass(P, mu), 0, P.N, P.N), P.n, c.mg);
        for method = {'global', 'dd'}
            s = load(fullfile(folder, [method{1} '.mat']), 'model');
            u = dmsrb.prefix(A, P.b, mu, s.model, 0);
            r = P.b - A * u;
            k = mg_response(info, r, A);
            rows = [rows; table(c.omega, mu, string(method{1}), norm(r) / norm(P.b), ...
                transverse_roughness(P, r), k.pre, k.coarse, k.combined, 'VariableNames', ...
                {'omega', 'mu', 'method', 'rho', 'residual_roughness_transverse_over_omega2', ...
                'mg_pre_optimal_contraction', 'mg_coarse_optimal_contraction', ...
                'mg_combined_optimal_contraction'})]; %#ok<AGROW>
            if mu == c.representative
                rn = r / norm(r);
                correction = dmsrb.sl.gmres(@(v) A * v, M, rn, [], waveguide_gmres_options(c, 1e-14, 1));
                after = rn - A * correction;
                [~, iz] = min(abs(P.coordinates{3} - 0.5));
                R0 = reshape(abs(r) / norm(r), P.n); R1 = reshape(abs(after) / norm(after), P.n);
                slice.([method{1} '_r0']) = R0(:, :, iz);
                slice.([method{1} '_rafter']) = R1(:, :, iz);
                labels = part.labels(:, :, iz);
            end
        end
    end
    writetable(rows, fullfile(folder, 'residual_components.csv'));
    slice.x = P.coordinates{1}; slice.y = P.coordinates{2}; slice.spacing = P.h;
    slice.subdomain_labels = double(labels); slice.physical_grid_size = P.Nphys; slice.pml_cells = P.npml;
    save(fullfile(folder, 'residual_slices.mat'), '-struct', 'slice', '-v7');
end

function export_paper()
    out = fullfile(results_root(), 'paper', 'appendixA');
    if ~isfolder(out), mkdir(out); end
    rows = table();
    for omega = [24, 36, 48, 60]
        file = fullfile(results_root(), 'exp3', sprintf('omega%g', omega), 'residual_components.csv');
        if isfile(file), rows = [rows; readtable(file, 'TextType', 'string')]; end %#ok<AGROW>
    end
    writetable(rows, fullfile(out, 'residual_components.csv'));
    copyfile(fullfile(results_root(), 'exp3', 'omega48', 'residual_slices.mat'), ...
        fullfile(out, 'residual_slices_omega48.mat'));
end

function root = results_root()
    root = getenv('DMSRB_RESULTS');
    if isempty(root)
        root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    end
end
