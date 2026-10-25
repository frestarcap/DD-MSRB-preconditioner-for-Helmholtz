function export_waveguide_rank_inputs(source_list,output_file)
% Read frozen 3D models and export only scalar rank-allocation inputs.
    lines=splitlines(string(fileread(source_list)));records={};
    for ii=1:numel(lines)
        if strlength(strtrim(lines(ii)))==0,continue;end
        parts=split(lines(ii),sprintf('\t'));omega=str2double(parts(1));source=char(parts(2));
        s=load(source,'models');model=s.models.dd;
        for j=1:numel(model.layers)
            blocks=model.layers{j}.space.blocks;
            rec=struct('tag',sprintf('waveguide_w%d',omega),'group','waveguide_frequency', ...
                'source',source,'layer',j,'n',[blocks.count],'saved_ranks',[blocks.k], ...
                'names',{{blocks.name}},'eta',blocks(1).meta.rank_global_tail_tol, ...
                'svals',{{}},'svals_hex',{{}},'spectrum_origin','saved local singular values');
            rec.eta_hex=num2hex(rec.eta);
            for b=1:numel(blocks)
                m=blocks(b).meta;
                assert(isempty(m.rank_tail_tol)&&isinf(m.rank_cap));
                assert(startsWith(m.rank_rule,'marginal_storage'));
                rec.svals{b}=m.singular_values(:).';
                rec.svals_hex{b}=cellstr(num2hex(m.singular_values(:)));
            end
            records{end+1}=rec; %#ok<AGROW>
            fprintf('EXPORTED omega=%g layer=%d blocks=%d\n',omega,j,numel(blocks));
        end
        clear s model
        payload=struct('matlab',version,'job_id',getenv('SLURM_JOB_ID'),'records',{records});
        fid=fopen(output_file,'w');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(payload));fclose(fid);
    end
    fprintf('WAVEGUIDE_RANK_EXPORT_PASS layers=%d\n',numel(records));
end
