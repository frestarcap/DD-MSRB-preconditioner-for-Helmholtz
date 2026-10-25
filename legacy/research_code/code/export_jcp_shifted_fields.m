function export_jcp_shifted_fields
% Export verified frequency-domain fields for the JCP publication plots.
    out=helm_output_path('outputs','jcp_frequency_revision_20260914');
    if ~exist(out,'dir'), mkdir(out); end
    s=load(helm_output_path('outputs','jcp_shifted_20260914','paper_results.mat'),'cases');
    g=load(helm_output_path('outputs','halfspace_dd_refinement', ...
        'review_repaired_20260905','global.mat'),'artifact');
    v=s.cases{1}{10,3};
    assert(strcmp(v.method,'global') && abs(v.parameter-.862209674352196)<1e-13);
    writematrix(g.artifact.x_int(:),fullfile(out,'halfspace_x.csv'));
    writematrix(g.artifact.y_int(:),fullfile(out,'halfspace_y.csv'));
    writematrix(real(reshape(v.u,numel(g.artifact.x_int),numel(g.artifact.y_int))).', ...
        fullfile(out,'halfspace_global_field.csv'));
    e=load(helm_output_path('outputs','ecc_two_layer_paper_20260911','paper_data.mat'),'dd_results');
    p=e.dd_results.problem; v=s.cases{2}{1,4};
    assert(strcmp(v.method,'dd') && abs(v.parameter-.907218592905774)<1e-13);
    writematrix(p.x_int(:),fullfile(out,'ecc_x.csv'));
    writematrix(p.y_int(:),fullfile(out,'ecc_y.csv'));
    writematrix(real(reshape(v.u,p.nx,p.ny)).',fullfile(out,'ecc_dd_field.csv'));
    fprintf('JCP_FIELD_EXPORT_PASS\n');
end
