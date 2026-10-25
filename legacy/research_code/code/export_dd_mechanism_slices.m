function report = export_dd_mechanism_slices(training_file,vectors_dir,outdir,reference_dir)
%EXPORT_DD_MECHANISM_SLICES Small central-plane data from saved native vectors.
% No operator/model assembly. Reads one vector in chunks through MATFILE,
% retaining only its central x-y plane. Original inputs are read-only.
% Optional reference_dir contains global/dd_refined_error.mat from the
% independent stricter-reference audit. It is omitted when not available.
    if nargin<4,reference_dir='';end
    assert(~isfolder(outdir) && ~isfile(outdir),'Refusing to overwrite a slice directory.');
    s=load(training_file,'geometry','resolved_problem_cfg','partition','cfg');
    g=s.geometry;pc=s.resolved_problem_cfg;
    n=double(g.grid_size(:).');N=prod(n);
    assert(numel(n)==3 && isfield(g,'coordinates'),'Saved coordinates are required.');
    axes_xyz=g.coordinates;
    for d=1:3
        axes_xyz{d}=double(axes_xyz{d}(:));
        assert(numel(axes_xyz{d})==n(d) && all(diff(axes_xyz{d})>0),'Invalid saved grid coordinates.');
    end
    center=double(pc.domain_lengths(3))/2;
    [~,iz]=min(abs(axes_xyz{3}-center));
    first=(iz-1)*n(1)*n(2)+1;last=iz*n(1)*n(2);
    block_slice=zeros(n(1),n(2),'uint32');
    block_names=cell(1,numel(s.partition.blocks));
    for b=1:numel(s.partition.blocks)
        ids=s.partition.blocks(b).idx;
        selected=ids(ids>=first & ids<=last)-first+1;
        assert(~any(block_slice(selected)),'Overlapping partition at the slice.');
        block_slice(selected)=uint32(b);block_names{b}=s.partition.blocks(b).name;
    end
    assert(all(block_slice(:)>0),'Partition does not cover the selected plane.');
    x=axes_xyz{1};y=axes_xyz{2};z=axes_xyz{3}(iz);
    profile=[];
    if isfield(pc,'material_model') && ismember(pc.material_model,{'gaussian_waveguide','waveguide'})
        assert(strcmp(pc.waveguide_axis,'x'),'Profile overlay currently supports x-directed waveguides.');
        wc=pc.waveguide_center;ww=pc.waveguide_width;
        if isscalar(ww),ww=[ww ww];end
        profile=repmat(exp(-.5*(((y(:).'-wc(2))/ww(1)).^2+((z-wc(3))/ww(2)).^2)),n(1),1);
    end
    geometry=struct('grid_size',n,'physical_grid_size',double(g.physical_grid_size), ...
        'pml_cells',double(g.pml_cells),'spacing',double(g.spacing), ...
        'domain_lengths',double(pc.domain_lengths),'x',x,'y',y,'z',z, ...
        'z_index_1based',iz,'central_plane_requested_z',center, ...
        'flattening','MATLAB column-major [nx,ny,nz]; matrix rows=x, columns=y', ...
        'block_labels',block_slice,'block_names',{block_names},'waveguide_profile',profile);
    clear s ids selected axes_xyz block_slice profile
    report=struct('definition_version','dd_central_slices_v1','omega',pc.omega, ...
        'source_training',training_file,'source_vectors_dir',vectors_dir, ...
        'matlab_version',version,'node',getenv('SLURMD_NODENAME'),'job_id',getenv('SLURM_JOB_ID'), ...
        'geometry',geometry,'records',struct([]));
    report.definitions=struct('norm','Original full-vector Euclidean norm, accumulated from bounded chunks', ...
        'normalized_abs_slice','abs(v(x,y,z_center))/norm(v) on the original discrete grid', ...
        'slice_mass_fraction','Sum of squared magnitudes on this one discrete plane divided by full-vector squared norm', ...
        'cross_frequency','Discrete normalization depends on grid density; optional continuum-normalized plotting multiplies by 1/sqrt(prod(spacing))', ...
        'e0','Difference from the saved shifted-MG reference; not an exact or certified solution error', ...
        'rafter','Residual of the normalized-RHS fixed-work completion; its raw norm is on a different scale from r0', ...
        'strict_reference','Optional new_error uses a stricter computed reference, still not a certified true solution');
    mkdir(outdir);
    for lc={'GGG','DDD'}
        variant=lc{1};files=dir(fullfile(vectors_dir,['*_' variant '_vectors.mat']));
        assert(numel(files)==1,'Expected exactly one native %s vector file.',variant);
        source=fullfile(files(1).folder,files(1).name);
        meta=load(source,'mu','label','trace');
        assert(strcmp(meta.label,variant),'Filename and saved variant disagree.');
        entry=struct('variant',variant,'mu',meta.mu,'source_file',source,'trace',meta.trace, ...
            'fields',struct());raw_slices=struct();
        for fc={'r0','e0','rafter'}
            name=fc{1};[field,raw]=read_field(source,name,N,first,last,n(1:2));
            entry.fields.(name)=field;raw_slices.(name)=raw;
        end
        if ~isempty(reference_dir)
            if strcmp(variant,'GGG'),method='global';else,method='dd';end
            refined=fullfile(reference_dir,[method '_refined_error.mat']);
            assert(isfile(refined),'Explicit reference_dir lacks %s.',refined);
            rm=load(refined,'mu','method');
            assert(abs(rm.mu-meta.mu)<1e-10 && strcmp(rm.method,method),'Reference/query mismatch.');
            [field,raw]=read_field(refined,'new_error',N,first,last,n(1:2));
            field.source_file=refined;entry.fields.strict_e0=field;raw_slices.strict_e0=raw;
        end
        if isempty(report.records),report.records=entry;else,report.records(end+1)=entry;end
        save(fullfile(outdir,[variant '_slices.mat']),'entry','raw_slices','geometry','-v7');
        fprintf('DD_SLICE_EXPORT omega=%g mu=%.3f %s nx=%d ny=%d z=%.8g\n', ...
            pc.omega,meta.mu,variant,n(1),n(2),z);
    end
    report.validated=true;
    f=fopen(fullfile(outdir,'slices.json'),'w');assert(f>=0,'Cannot write slice JSON.');
    cleanup=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(report));
end

function [field,raw]=read_field(source,name,N,first,last,plane_n)
    mf=matfile(source);info=whos(mf,name);
    assert(isscalar(info) && isequal(info.size,[N 1]),'Expected a column vector %s with matching grid.',name);
    length_chunk=2^18;total_norm=0;
    for start=1:length_chunk:N
        stop=min(N,start+length_chunk-1);
        values=mf.(name)(start:stop,1);
        assert(all(isfinite(values)),'Nonfinite saved vector %s.',name);
        total_norm=hypot(total_norm,norm(values));
    end
    raw=reshape(mf.(name)(first:last,1),plane_n);
    if total_norm>0,normalized=abs(raw)/total_norm;else,normalized=zeros(plane_n);end
    field=struct('variable',name,'original_norm_l2',total_norm, ...
        'slice_raw_norm_l2',norm(raw(:)),'slice_mass_fraction',sum(normalized(:).^2), ...
        'normalized_abs_slice',normalized,'read_chunk_length',length_chunk, ...
        'normalization_zero_vector',total_norm==0);
end
