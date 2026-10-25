function style_halfspace_residual_axes(ax, parameter)
%STYLE_HALFSPACE_RESIDUAL_AXES Shared paper styling for Figures 5 and 6.
    set(ax,'FontSize',28,'LineWidth',1.8,'Box','on', ...
        'YScale','log','XGrid','on','YGrid','on', ...
        'GridAlpha',0.28,'MinorGridAlpha',0.18, ...
        'XMinorGrid','off','YMinorGrid','on');
    ylim(ax,[1e-15,2]);
    yticks(ax,10.^(-15:5:0));
    xlabel(ax,'online step index','FontSize',30);
    ylabel(ax,'residual','FontSize',30);
    title(ax,sprintf('Half-space, c_{right}=%.4f',parameter), ...
        'FontSize',30,'FontWeight','bold');
    lgd=legend(ax,'Location','northeast','FontSize',25);
    lgd.Box='on'; lgd.LineWidth=1.1;
    ax.Position=[0.14,0.15,0.82,0.74];
    axtoolbar(ax,{});
end
