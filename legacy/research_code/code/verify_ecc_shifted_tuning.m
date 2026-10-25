function verify_ecc_shifted_tuning
% Check production replay, conditional setup, storage, and frozen operators.
    out=helm_output_path('outputs','ecc_shifted_tuning_20260919');
    s=load(fullfile(out,'confirm_results.mat'),'c','models','names','cases','specs');
    rows=table(); storage=table(); algebra=table(); regions=table();
    partition=s.c.models{2}.layers{1}.space.partition;
    for m=1:2
        for qi=1:numel(s.c.parameters)
            sys=s.c.systems{qi}; residual=sys.b-sys.A*s.c.expected{m}(:,qi);
            near_fraction=norm(residual(partition.idx_near))^2/norm(residual)^2;
            regions=[regions;table(string(s.c.names{m}),qi,norm(residual)/norm(sys.b),near_fraction, ...
                'VariableNames',{'method','query','post_residual','near_residual_energy_fraction'})]; %#ok<AGROW>
        end
    end
    for mi=5:numel(s.names)
        name=s.names{mi}; isdd=startsWith(name,'dd'); method='global'; if isdd,method='dd';end
        si=ceil((mi-4)/2); spec=s.specs(si,:);
        source=fullfile(out,sprintf('%s_n%d_d%d_eta%.0e.mat',method,spec(1),spec(2),spec(3)));
        r=load(source,'cfg','row'); cfg=r.cfg;
        model=struct('num_layers',spec(4),'layers',{cell(spec(4),1)});
        basis_entries=0; map_entries=0; ranks=zeros(1,spec(4));
        for j=1:spec(4)
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
            assert(rho<=1e-10 && delta<1e-9);
            accepted=v.precorr.relres_final<=1e-10;
            assert((v.completion.preconditioner_setup_time==0)==accepted);
            if accepted
                assert(v.completion.preconditioner_factor_bytes==0 && v.completion.iter_count==0);
            end
            rows=[rows;table(string(name),qi,rho,delta,accepted, ...
                v.completion.preconditioner_setup_time,'VariableNames', ...
                {'method','query','residual','field_difference','ROM_accepted','setup_s'})]; %#ok<AGROW>
        end
    end
    writetable(rows,fullfile(out,'production_verification.csv'));
    writetable(storage,fullfile(out,'storage.csv'));
    writetable(algebra,fullfile(out,'assembly_verification.csv'));
    writetable(regions,fullfile(out,'residual_regions.csv'));
    disp(storage); fprintf('ECC_SHIFTED_PRODUCTION_PASS checks=%d\n',height(rows));
end
