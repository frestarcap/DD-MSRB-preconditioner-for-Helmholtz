function fig=plot_rom_stage_residuals(tests,indices,labels,baseline,index,title_text,kind)
%PLOT_ROM_STAGE_RESIDUALS Display unscaled histories with explicit equations.
% Legacy combined data are read only up to the end of the true ROM prefix.
    fig=figure('Color','w','Position',[100,100,1050,1000],'Visible','off');
    layout=tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
    colors={[0.85,0.05,0.05],[0.05,0.45,0.25],[0.15,0.35,0.75]};
    markers={'o','s','d'}; ax=nexttile(layout); hold(ax,'on'); stage_steps=0;
    for m=1:numel(tests)
        t=tests{m}; k=indices(m);
        if isfield(t,'online_precorr_relres_hist')
            h=t.online_precorr_relres_hist{k};
        else
            h=t.online_combined_relres_hist{k};
            h=h(1:t.online_combined_transition_idx(k));
        end
        stage_steps=max(stage_steps,numel(h)-1);
        draw(ax,h,colors{m},markers{m},labels{m});
    end
    draw(ax,baseline.helm_baseline_relres_hist{index},[0.4,0,0.5],'.','Plain Helmholtz GMRES');
    format(ax); xlim(ax,[0,max(stage_steps,1)]); xticks(ax,0:max(stage_steps,1));
    xlabel(ax,'Helmholtz iteration (reduced-stage range)');
    ylabel(ax,'||b - A u||_2 / ||b||_2'); title(ax,'Frequency-domain residual');
    legend(ax,'Location','northeast','FontSize',18);
    ax=nexttile(layout); hold(ax,'on');
    for m=1:numel(tests)
        draw(ax,tests{m}.online_fp_relres_hist{indices(m)},colors{m},markers{m},labels{m});
    end
    draw(ax,baseline.plain_fp_relres_hist{index},[0.15,0.15,0.15],'.','Plain WaveHoltz');
    format(ax); xlabel(ax,'WaveHoltz iteration (from stage start)');
    if strcmp(kind,'ecc')
        ylabel(ax,'||b - A u_{rec}||_2 / ||b||_2');
        title(ax,'Completion: recovered Helmholtz residual');
    else
        ylabel(ax,'||b_{WH} - (I - \Pi_0)x||_2 / ||b_{WH}||_2');
        title(ax,'Completion: native WaveHoltz residual');
    end
    legend(ax,'Location','northeast','FontSize',18);
    title(layout,title_text,'FontSize',25,'FontWeight','normal');
end

function draw(ax,h,color,marker,label)
    if isempty(h) || ~any(isfinite(h)), return; end
    h=h(:); h(~isfinite(h) | h<=0)=NaN;
    semilogy(ax,0:numel(h)-1,h,'Color',color,'Marker',marker,'LineWidth',2.3, ...
        'MarkerSize',6,'MarkerIndices',unique(round(linspace(1,numel(h),min(22,numel(h))))), ...
        'DisplayName',label);
end

function format(ax)
    set(ax,'YScale','log','FontSize',20,'LineWidth',1.2,'Box','on');
    grid(ax,'on'); ax.GridAlpha=0.14; ylim(ax,[1e-15,2]);
end
