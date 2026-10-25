function metrics = plot_ecc_global_spectral_energy(source)
%PLOT_ECC_GLOBAL_SPECTRAL_ENERGY Archived three-layer Global snapshot budgets.
% Reads existing paper data only; no ECC training or online solve is repeated.
    if nargin<1
        source=helm_output_path('outputs','helm_trajectory_evidence', ...
            'review_repaired_20260905','ecc_residual_ablation_paper.mat');
    end
    s=load(source,'global_results'); R=s.global_results;
    out=helm_output_path('outputs','figures','ecc_correction_rom');
    if ~exist(out,'dir'), mkdir(out); end
    spectra=cell(3,1); ranks=zeros(3,1); eta=zeros(3,1);
    achieved=zeros(3,1); previous=zeros(3,1);
    curves=table();
    for j=1:3
        p=R.correction_rom.layers{j}.pod;
        spectra{j}=p.singular_values(:); ranks(j)=p.rank_used; eta(j)=p.rank_tail_tol;
        v=spectra{j}; achieved(j)=norm(v(ranks(j)+1:end))/norm(v);
        previous(j)=norm(v(ranks(j):end))/norm(v);
        assert(achieved(j)<=eta(j) && previous(j)>eta(j));
        for r=0:numel(v)
            tail=norm(v(r+1:end))/norm(v);
            curves=[curves;table(j,r,tail,tail^2,'VariableNames', ...
                {'layer','retained_rank','relative_tail','discarded_energy_fraction'})]; %#ok<AGROW>
        end
    end
    metrics=table((1:3)',eta,ranks,achieved,previous,1-achieved.^2, ...
        'VariableNames',{'layer','eta','rank','achieved_tail','previous_rank_tail','retained_energy_fraction'});
    writetable(metrics,fullfile(out,'ecc_global_spectral_energy_metrics.csv'));
    writetable(curves,fullfile(out,'ecc_global_spectral_energy_tails.csv'));
    save(fullfile(out,'ecc_global_spectral_energy.mat'),'source','spectra','metrics','curves');
    clear s R;
    colors=[0.10,0.36,0.67;0.83,0.32,0.08;0.12,0.53,0.34];
    fig=figure('Color','w','Position',[100,100,1240,490],'Visible','off');
    layout=tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');
    a=nexttile(layout); hold(a,'on');
    b=nexttile(layout); hold(b,'on');
    for j=1:3
        v=spectra{j};
        semilogy(a,1:numel(v),abs(v).^2/sum(abs(v).^2),'-o', ...
            'Color',colors(j,:),'LineWidth',2,'MarkerSize',5, ...
            'DisplayName',sprintf('Layer %d',j));
        t=curves(curves.layer==j & curves.relative_tail>0,:);
        semilogy(b,t.retained_rank,t.relative_tail,'-o','Color',colors(j,:), ...
            'LineWidth',2,'MarkerSize',5, ...
            'DisplayName',sprintf('Layer %d: \\eta_%d = 10^{%d}, r_%d = %d',j,j,round(log10(eta(j))),j,ranks(j)));
        yline(b,eta(j),'--','Color',colors(j,:),'LineWidth',1.2,'HandleVisibility','off');
        plot(b,ranks(j),achieved(j),'o','Color',colors(j,:), ...
            'MarkerFaceColor',colors(j,:),'MarkerSize',10,'HandleVisibility','off');
    end
    for ax=[a,b]
        set(ax,'YScale','log','FontName','Helvetica','FontSize',16,'Box','on','LineWidth',0.9);
        grid(ax,'on'); ax.GridAlpha=0.13;
    end
    xlabel(a,'Mode index k'); ylabel(a,'\sigma_{j,k}^2 / \Sigma_i \sigma_{j,i}^2');
    title(a,'Global snapshot energy distribution','FontWeight','normal');
    xlim(a,[1,12]); xticks(a,1:2:12); legend(a,'Location','northeast','FontSize',14);
    xlabel(b,'Retained rank r'); ylabel(b,'Relative Frobenius tail E_j(r)');
    title(b,'Tail budgets and selected ranks','FontWeight','normal');
    xlim(b,[0,11]); xticks(b,0:2:11); legend(b,'Location','southwest','FontSize',13);
    exportgraphics(fig,fullfile(out,'ecc_global_spectral_energy.png'),'Resolution',220);
    exportgraphics(fig,fullfile(out,'ecc_global_spectral_energy.pdf'),'ContentType','vector');
    for extension={'png','pdf'}
        name=['ecc_global_spectral_energy.' extension{1}];
        published=helm_output_path('LaTeX_DL_468198_240419','figures','ecc_correction_rom',name);
        if isfile(published), copyfile(fullfile(out,name),published); end
    end
    close(fig); disp(metrics);
end
