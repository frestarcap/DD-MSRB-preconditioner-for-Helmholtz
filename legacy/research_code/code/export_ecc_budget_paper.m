function export_ecc_budget_paper
% Export frozen, verified uniform5 evidence; no retraining or new timing.
    out=helm_output_path('outputs','ecc_dd_budget_20260919');
    s=load(fullfile(out,'uniform5_results.mat'),'c','cases','names','rows');
    assert(abs(s.c.parameters(1)-.909176375080444)<1e-12);
    e=load(helm_output_path('outputs','ecc_two_layer_paper_20260911','paper_data.mat'),'dd_results');
    p=e.dd_results.problem;
    writematrix(p.x_int(:),fullfile(out,'paper_x.csv'));
    writematrix(p.y_int(:),fullfile(out,'paper_y.csv'));
    history=table();representative=table();
    for name={'shifted','uniform5_global','uniform5_dd'}
        mi=find(strcmp(s.names,name{1}));c=s.cases{1,mi};
        h=c.ROM_history(:);ch=c.completion_history(:);
        if ~isempty(ch),h=[h;ch(2:end)];end
        r=s.rows(s.rows.method==string(name{1}) & s.rows.query==1 & s.rows.repeat==1,:);
        h(end)=r.residual;
        history=[history;table(repmat(string(name{1}),numel(h),1),(0:numel(h)-1)',h, ...
            'VariableNames',{'method','step','residual'})]; %#ok<AGROW>
        representative=[representative;r]; %#ok<AGROW>
        if strcmp(name{1},'uniform5_dd')
            writematrix(real(reshape(c.u,p.nx,p.ny)).',fullfile(out,'paper_dd_field.csv'));
        end
    end
    writetable(history,fullfile(out,'paper_histories.csv'));
    writetable(representative,fullfile(out,'paper_representative.csv'));
    spectra=table();
    s=load(fullfile(out,'uniform5_global.mat'),'training','model');
    for j=1:3
        v=s.training{j}.s(:);tail=arrayfun(@(k)norm(v(k+1:end))/norm(v),(0:numel(v))');
        spectra=[spectra;table(repmat(j,numel(v)+1,1),(0:numel(v))', ...
            [NaN;abs(v).^2/sum(abs(v).^2)],tail,'VariableNames',{'layer','rank','energy','tail'})]; %#ok<AGROW>
    end
    writetable(spectra,fullfile(out,'paper_spectra.csv'));
    fprintf('ECC_BUDGET_PAPER_EXPORT_PASS\n');
end
