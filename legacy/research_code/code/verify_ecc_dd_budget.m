function verify_ecc_dd_budget(mode)
% Check production replay, conditional setup, storage, and frozen operators.
    if nargin<1,mode='validate';end
    prefix='';if ~strcmp(mode,'validate'),prefix=[mode '_'];end
    out=helm_output_path('outputs','ecc_dd_budget_20260919');
    s=load(fullfile(out,[mode '_results.mat']),'c','models','names','cases','specs');
    rows=table(); storage=table(); algebra=table();
    for mi=3:numel(s.names)
        name=s.names{mi};source=fullfile(out,[name '.mat']);
        r=load(source,'cfg','row');cfg=r.cfg;depth=numel(s.models{mi}.layers);cfg.corr_num_layers=depth;
        model=struct('num_layers',depth,'layers',{cell(depth,1)});
        basis_entries=0; map_entries=0; ranks=zeros(1,depth);
        for j=1:depth
            sp=s.models{mi}.layers{j}.space;
            ranks(j)=rom_reduced_space_utils('dimension',sp);
            A=s.c.systems{1}.A;
            Ar=rom_reduced_space_utils('assemble_reduced_matrix',A,sp);
            ref=rom_reduced_space_utils('assemble_reduced_matrix',@(v)A*v,sp);
            delta=norm(Ar-ref,'fro')/max(1,norm(ref,'fro'));
            assert(delta<1e-12);
            algebra=[algebra;table(string(name),j,delta)]; %#ok<AGROW>
            if strcmp(sp.type,'dense'),basis_entries=basis_entries+numel(sp.basis);
            else
                basis_entries=basis_entries+numel(sp.Un)+numel(sp.Uf)+numel(sp.Up);
                map_entries=numel(sp.partition.idx_near)+numel(sp.partition.idx_far)+numel(sp.partition.idx_pml);
                sp=rmfield(sp,intersect(fieldnames(sp),{'block_meta'}));
            end
            model.layers{j}=struct('basis_space',sp);
        end
        deployment_file=fullfile(out,[name '_deployment.mat']);
        save(deployment_file,'model','cfg','source','-v7.3');
        w=whos('model');
        storage=[storage;table(string(name),string(mat2str(ranks)),16*basis_entries/2^20, ...
            8*map_entries/2^20,(16*basis_entries+8*map_entries)/2^20,w.bytes/2^20, ...
            r.row.offline_s,'VariableNames',{'method','ranks','basis_MiB','shared_map_MiB', ...
            'payload_MiB','MATLAB_object_MiB','offline_s'})]; %#ok<AGROW>
        cfg.deployment_file=deployment_file; cfg.deployment_audit=true;
        for qi=1:numel(s.c.parameters)
            cfg.ratio_test=s.c.parameters(qi);
            v=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
            expected=s.cases{qi,mi}; A=s.c.systems{qi}.A; b=s.c.systems{qi}.b;
            rho=norm(b-A*v.final_u)/norm(b);
            delta=norm(v.final_u-expected.u)/max(norm(expected.u),eps);
            assert(rho<=1e-12 && delta<1e-9);
            accepted=v.precorr.relres_final<=1e-12;
            assert((v.completion.preconditioner_setup_time==0)==accepted);
            if accepted
                assert(v.completion.preconditioner_factor_bytes==0 && v.completion.iter_count==0);
            end
            rows=[rows;table(string(name),qi,rho,delta,accepted, ...
                v.completion.preconditioner_setup_time,'VariableNames', ...
                {'method','query','residual','field_difference','ROM_accepted','setup_s'})]; %#ok<AGROW>
        end
    end
    writetable(rows,fullfile(out,[prefix 'production_verification.csv']));
    writetable(storage,fullfile(out,[prefix 'storage.csv']));
    writetable(algebra,fullfile(out,[prefix 'assembly_verification.csv']));

    disp(storage); fprintf('ECC_DD_BUDGET_PRODUCTION_PASS checks=%d\n',height(rows));
end
