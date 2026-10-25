function checks = check_waveguide_timing_order()
%CHECK_WAVEGUIDE_TIMING_ORDER Small reversed-order control, in a fresh process.
% Does not replace or overwrite the primary three-repeat measurements.
    out=helm_output_path('outputs','waveguide_threshold_tradeoff_20260908');
    folder=fullfile(out,'order_check');
    if ~exist(folder,'dir'), mkdir(folder); end
    file=fullfile(folder,'reference_global.mat');
    copyfile(fullfile(out,'reference_global.mat'),file);
    g=run_jsc_3d_deployment(file,'lazy',[],3,false);
    b=run_jsc_3d_deployment(file,'baseline',[],1,false);
    old_g=readtable(fullfile(out,'reference_global_lazy_timings.csv'));
    old_b=readtable(fullfile(out,'reference_global_baseline_timings.csv'));
    checks=table();
    for mu=unique(g.parameter).'
        gp=mean(old_g.total_s(old_g.parameter==mu));
        bp=mean(old_b.total_s(old_b.parameter==mu));
        gc=mean(g.total_s(g.parameter==mu)); bc=mean(b.total_s(b.parameter==mu));
        row=table(mu,gp,gc,bp,bc,bp/gp,bc/gc, ...
            'VariableNames',{'parameter','primary_global_s','control_global_s', ...
            'primary_baseline_s','control_baseline_s','primary_speedup','control_speedup'});
        checks=[checks;row]; %#ok<AGROW>
    end
    writetable(checks,fullfile(folder,'comparison.csv')); disp(checks);
end
