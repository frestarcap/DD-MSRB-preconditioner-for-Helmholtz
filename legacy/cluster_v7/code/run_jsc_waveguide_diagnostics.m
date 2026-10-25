function run_jsc_waveguide_diagnostics
%RUN_JSC_WAVEGUIDE_DIAGNOSTICS Actual online defects for archived 3D models.
    source=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
        'gaussian_waveguide_waveguide_dd_memory_saving_41pct_reproduction_training.mat');
    s=load(source,'models','resolved_problem_cfg','rhs');
    problem=helmholtz_3d_problem_interface(s.resolved_problem_cfg); rows=table();
    for method={'global','dd'}
        model=s.models.(method{1});
        spaces=cellfun(@(l)l.space,model.layers,'UniformOutput',false);
        for mu=[.317,.483]
            mass=problem.affine.mass_base+mu*problem.affine.mass_parameter_diagonal;
            A=problem.K_pml+spdiags(mass,0,problem.N,problem.N);
            reduced=cellfun(@(l)l.Ar0+mu*l.Ar1,model.layers,'UniformOutput',false);
            [~,d]=rom_true_action_diagnostics(A,s.rhs,spaces,reduced);
            d.method=repmat(string(method{1}),height(d),1); d.parameter=repmat(mu,height(d),1);
            rows=[rows;d]; %#ok<AGROW>
        end
    end
    writetable(rows,helm_output_path('outputs','revision_20260907','deployment', ...
        'waveguide_online_diagnostics.csv'));
    disp(rows);
end
