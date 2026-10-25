function directory=build_helm_3d_hpc_mex()
%BUILD_HELM_3D_HPC_MEX Compile only on the machine that will run the solver.
    directory=helm_output_path('outputs','scnet_3d_20260909','mex',computer('arch'));
    if ~isfolder(directory), mkdir(directory); end
    source=fullfile(fileparts(mfilename('fullpath')),'helm_gmres_cgs2_mex.c');
    binary=fullfile(directory,['helm_gmres_cgs2_mex.' mexext]);
    built=false;
    if ~isfile(binary) || dir(source).datenum>dir(binary).datenum
        mex('-R2018a','-O','-outdir',directory,source,'-lmwblas');
        built=true;
    end
    addpath(directory);
    assert(exist('helm_gmres_cgs2_mex','file')==3);
    fprintf('CGS2 BLAS kernel: %s\n',binary);
    marker=[binary '.validated'];
    if built || ~isfile(marker)
        test_helm_3d_hpc;
        fid=fopen(marker,'w'); fprintf(fid,'Validated %s\n',datestr(now,30)); fclose(fid);
    end
end
