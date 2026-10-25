function plot_halfspace_true_residual_figure(d)
%PLOT_HALFSPACE_TRUE_RESIDUAL_FIGURE Same equation and unscaled stage boundary.
    fig=figure('Color','w','Position',[100,100,980,720],'Visible','off');
    ax=axes(fig); hold(ax,'on');
    tests={d.global,d.dd,d.nr}; indices=[1,d.dd_index,1];
    colors={[0.85,0.05,0.05],[0.05,0.45,0.25],[0.15,0.35,0.75]};
    labels={'Global TA-ROM','DD-TA-ROM','NR-ROM'}; markers={'o','s','d'};
    file=helm_output_path('outputs','halfspace_dd_paper','fixed_pod_residual_interpolation.mat');
    if isfile(file)
        s=load(file,'fixed'); assert(abs(s.fixed.parameter-d.parameter)<1e-13);
        tests{end+1}=s.fixed.test; indices(end+1)=1;
        colors{end+1}=[0.85,0.45,0.05]; markers{end+1}='^';
        labels{end+1}='Fixed solution POD';
    end
    for j=1:numel(tests)
        t=tests{j}; k=indices(j);
        if isfield(t,'online_precorr_relres_hist')
            h=t.online_precorr_relres_hist{k};
        else
            h=t.online_combined_relres_hist{k};
            h=h(1:t.online_combined_transition_idx(k));
        end
        m=numel(h)-1;
        draw(ax,0:m,h,colors{j},markers{j},labels{j});
        if t.online_fp_iter(k)>0
            assert(isfield(t,'online_true_helm_relres_hist'),'Missing true completion history.');
            w=t.online_true_helm_relres_hist{k};
            assert(~isempty(w),'Missing true completion history.');
            % Keep BOTH transition measurements at the same step index.
            plot(ax,[m,m],[h(end),w(1)],':','Color',colors{j}, ...
                'LineWidth',1.5,'HandleVisibility','off');
            line=draw(ax,m+(0:numel(w)-1),w,colors{j},markers{j},labels{j});
            line.HandleVisibility='off';
            plot(ax,m,w(1),'p','Color',colors{j},'MarkerSize',13, ...
                'MarkerFaceColor','w','HandleVisibility','off');
        end
    end
    w=d.global.plain_true_helm_relres_hist{1};
    draw(ax,0:numel(w)-1,w,[0.15,0.15,0.15],'o','Plain WaveHoltz');
    w=d.global.helm_baseline_relres_hist{1};
    draw(ax,0:numel(w)-1,w,[0.4,0,0.5],'o','Plain Helmholtz GMRES');
    style_halfspace_residual_axes(ax,d.parameter);
    name='helm_trajectory_vs_nr_residual_decay_c_right_0_8622.png';
    output=helm_output_path('outputs','figures','helm_correction_rom',name);
    set(fig,'PaperPositionMode','auto');
    print(fig,output,'-dpng','-r300'); close(fig);
    copyfile(output,helm_output_path('LaTeX_DL_468198_240419','figures','helm_correction_rom',name));
end
function h=draw(ax,x,y,c,m,label)
    y=y(:); y(~isfinite(y)|y<=0)=NaN;
    h=semilogy(ax,x,y,'Color',c,'Marker',m,'LineWidth',3.2, ...
        'MarkerSize',9,'MarkerFaceColor','w','MarkerIndices',unique(round(linspace(1,numel(y),min(22,numel(y))))), ...
        'DisplayName',label);
end
