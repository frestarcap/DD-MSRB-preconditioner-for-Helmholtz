function results = test_dd_residual_interfaces(output_dir)
%TEST_DD_RESIDUAL_INTERFACES Controlled planar boundary and constant fields.
    if nargin<1
        output_dir=fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'outputs','dd_highfreq_mechanism_20260913','interface_probe', ...
            char(datetime('now','Format','yyyyMMdd_HHmmssSSS')));
    end
    assert(~isfolder(output_dir),'Synthetic test directory already exists.');
    mkdir(output_dir); vectors_dir=fullfile(output_dir,'vectors'); mkdir(vectors_dir);
    n=[8 7 6];np=[6 5 4];pml=[1 1 1];
    physical=false(n);physical(2:7,2:6,2:5)=true;
    labels=3*ones(n,'uint32');labels(2:4,2:6,2:5)=1;labels(5:7,2:6,2:5)=2;
    blocks=struct('name',{},'idx',{});
    for k=1:3
        blocks(k).name=sprintf('block%d',k);blocks(k).idx=find(labels(:)==k);
    end
    partition=struct('blocks',blocks);
    geometry=struct('grid_size',n,'physical_grid_size',np,'pml_cells',pml,'spacing',[.1 .2 .3]);
    resolved_problem_cfg=struct('omega',24);
    training_file=fullfile(output_dir,'synthetic_training.mat');
    save(training_file,'partition','geometry','resolved_problem_cfg');
    r0=zeros(n);r0(4:5,2:6,2:5)=1;r0=r0(:);
    e0=double(labels(:)==2);
    rafter=(1+2i)*ones(prod(n),1);
    mu=.317;label='PLANAR';
    save(fullfile(vectors_dir,'w24_mu0.317_PLANAR_vectors.mat'),'r0','e0','rafter','mu','label');
    out=fullfile(output_dir,'interface_audit.json');
    report=audit_dd_residual_interfaces(training_file,vectors_dir,out);
    g=report.geometry;r=report.records(1);
    assert(isequal(g.physical_edge_count_by_axis,[20 0 0;80 96 90]));
    assert(isequal(g.all_internal_edge_count_by_axis,[(8-1)*7*6,8*(7-1)*6,8*7*(6-1)]));
    assert(isequal(g.transition_edge_count_by_axis,[40 48 60]));
    assert(isequal(g.mask_node_count,[256 40 244]));
    assert(isequal(g.physical_mask_node_count,[108 40 96]));
    assert(r.r0.energy_total==40 && r.r0.energy_physical==40);
    assert(r.r0.physical_energy_fraction(2)==1);
    assert(abs(r.r0.physical_enrichment(2)-3)<1e-14);
    assert(isequal(r.e0.edge_mean_squared_difference,[1;0]));
    assert(isequal(r.e0.edge_squared_difference_sum_by_axis,[20 0 0;0 0 0]));
    assert(abs(r.e0.edge_mean_squared_gradient(1)-100)<1e-11);
    assert(max(abs(r.rafter.enrichment-1))<1e-13);
    assert(max(abs(r.rafter.physical_enrichment-1))<1e-13);
    assert(isequal(r.rafter.edge_mean_squared_difference,[0;0]));
    blocked=false;
    try
        audit_dd_residual_interfaces(training_file,vectors_dir,out);
    catch err
        blocked=contains(err.message,'Refusing to overwrite');
    end
    assert(blocked,'Output overwrite was not rejected.');
    results=struct('all_passed',true,'output_dir',output_dir,'matlab_version',version, ...
        'physical_interface_nodes',g.mask_node_count(2), ...
        'physical_interface_edges',sum(g.physical_edge_count_by_axis(1,:)), ...
        'controlled_interface_enrichment',r.r0.physical_enrichment(2), ...
        'overwrite_rejected',blocked);
    save(fullfile(output_dir,'validation.mat'),'results');
    fprintf('DD_INTERFACE_TEST_PASS: 20 planar edges, 40 nodes, enrichment 3; constants uniform.\n');
end
