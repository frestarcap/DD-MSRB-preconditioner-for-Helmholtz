function cfg = halfspace_paper_dd_config(cfg)
%HALFSPACE_PAPER_DD_CONFIG Accepted two-material/eight-PML-block representation.
    if nargin < 1, cfg = struct(); end
    cfg.corr_basis_style = 'block3';
    cfg.block3_partition_mode = 'halfspace_refined';
    cfg.corr_block3_global_tail_strategy = 'marginal_storage';
    cfg.halfspace_dd_physical_mode = 'slabs';
    cfg.halfspace_dd_sections = 1;
    cfg.halfspace_dd_transverse_sections = 1;
    cfg.halfspace_dd_pml_mode = 'material_symmetric_depth';
    cfg.halfspace_dd_depth_edges = 0.5;
    if isfield(cfg, 'corr_pod_singular_value_tols')
        cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
    end
end
