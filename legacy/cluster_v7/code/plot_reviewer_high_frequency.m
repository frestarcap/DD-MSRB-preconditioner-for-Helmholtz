function plot_reviewer_high_frequency
%PLOT_REVIEWER_HIGH_FREQUENCY Actual Helmholtz and native WH histories, separate.
    out=helm_output_path('outputs','reviewer_20260909','high_frequency_waveholtz');
    fig=figure('Visible','off','Color','w','Position',[100,100,1000,430]);
    layout=tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    ax1=nexttile(layout); hold(ax1,'on'); ax2=nexttile(layout); hold(ax2,'on');
    colors=[0.15,0.37,0.70;0.84,0.32,0.16];
    for m=1:2
        methods={'global','dd'}; method=methods{m};
        d=load(fullfile(out,[method '_scale4.mat']),'artifact');
        k=find(abs(d.artifact.test.params-0.8761)<1e-12,1); assert(~isempty(k));
        h=d.artifact.test.online_precorr_relres_hist{k};
        semilogy(ax1,0:numel(h)-1,h,'-o','Color',colors(m,:), ...
            'LineWidth',1.6,'DisplayName',upper(method));
        if m==1
            baseline=d.artifact.test.plain_fp_relres_hist{k};
            semilogy(ax2,0:numel(baseline)-1,baseline,'-','Color',[.25 .25 .25], ...
                'LineWidth',1.6,'DisplayName','WaveHoltz baseline');
        end
        clear d;
        p=load(fullfile(out,[method '_scale4_prefix_result.mat']),'result');
        h=p.result.rows.native_history;
        semilogy(ax2,0:numel(h)-1,h,'-','Color',colors(m,:), ...
            'LineWidth',1.6,'DisplayName',[upper(method) ' first layer + WaveHoltz']);
    end
    set([ax1,ax2],'YScale','log','FontSize',12,'Box','on');
    xticks(ax1,0:3);
    xlim(ax2,[0,620]); xticks(ax2,0:100:600);
    xlabel(ax1,'Reduced FGMRES steps'); ylabel(ax1,'True Helmholtz relative residual');
    xlabel(ax2,'Native WaveHoltz GMRES steps'); ylabel(ax2,'Native fixed-point relative residual');
    title(ax1,'Three-layer deployed models'); title(ax2,'One-layer completion ablation');
    yline(ax1,1e-8,'--','Helmholtz target','HandleVisibility','off');
    yline(ax2,1e-10,'--','Native tolerance','HandleVisibility','off');
    legend(ax1,'Location','northeast'); legend(ax2,'Location','northeast','FontSize',10);
    grid(ax1,'on');grid(ax2,'on');
    title(layout,'High frequency: \omega = 214, N = 570,025, c_{right} = 0.8761','FontSize',14);
    exportgraphics(fig,fullfile(out,'high_frequency_histories.pdf'),'ContentType','vector');
    exportgraphics(fig,fullfile(out,'high_frequency_histories.png'),'Resolution',200);close(fig);
    dest=helm_output_path('LaTeX_DL_468198_240419','figures','reviewer_20260909');
    if ~isfolder(dest),mkdir(dest);end
    copyfile(fullfile(out,'high_frequency_histories.pdf'),dest);
end
