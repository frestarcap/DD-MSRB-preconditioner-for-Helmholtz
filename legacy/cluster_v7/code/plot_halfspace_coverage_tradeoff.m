function plot_halfspace_coverage_tradeoff
% Plot only Global/DD rows from the unchanged nine-query timing dataset.
    t=readtable(helm_output_path('outputs','revision_20260907','halfspace','summary_timing.csv'));
    fig=figure('Visible','off','Color','w','Position',[80,80,980,720]);
    ax=axes(fig); hold(ax,'on'); names={'global','dd'};
    labels={'Global TA-ROM','DD-TA-ROM'}; colors={[0.85,.05,.05],[.05,.45,.25]};
    for j=1:2
        q=t(startsWith(string(t.candidate),[names{j} '_n24_']),:);
        q=sortrows(q,'payload_MiB');
        semilogy(ax,q.payload_MiB,q.online_mean_s,'-o','Color',colors{j}, ...
            'LineWidth',3,'MarkerSize',9,'DisplayName',labels{j});
        for k=1:height(q)
            e=extractAfter(string(q.candidate(k)),'eta');
            text(ax,q.payload_MiB(k),q.online_mean_s(k)*1.08,['10^{' char(e) '}'],'FontSize',21);
        end
    end
    set(ax,'YScale','log','FontSize',25,'Box','on'); grid(ax,'on');
    xlabel(ax,'ROM payload (MiB)'); ylabel(ax,'Mean total online time (s)');
    legend(ax,'Location','northeast','FontSize',23);
    exportgraphics(fig,helm_output_path('LaTeX_DL_468198_240419','figures', ...
        'helm_correction_rom','helm_tol_tradeoff_paper.png'),'Resolution',300); close(fig);
end
