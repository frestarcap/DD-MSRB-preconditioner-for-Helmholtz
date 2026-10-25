function pool = helm_3d_parallel_pool(workers, threads)
%HELM_3D_PARALLEL_POOL One process pool, with bounded BLAS threads per worker.
    allocation = str2double(getenv('SLURM_CPUS_PER_TASK'));
    assert(workers >= 1 && threads >= 1 && workers == fix(workers) && threads == fix(threads));
    if isfinite(allocation)
        assert(workers * threads + 1 <= allocation, ...
            'Reserve one CPU for the MATLAB client during worker computation.');
    end
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workers
        delete(pool); pool = [];
    end
    if isempty(pool)
        cluster = parcluster('Processes');
        cluster.NumThreads = threads;
        jobid = getenv('SLURM_JOB_ID');
        if ~isempty(jobid)
            location = fullfile(tempdir, ['helm_pool_' jobid]);
            if ~isfolder(location), mkdir(location); end
            cluster.JobStorageLocation = location;
        end
        pool = parpool(cluster, workers);
    end
    f = parfevalOnAll(pool, @maxNumCompThreads, 0, threads);
    fetchOutputs(f);
    limits=fetchOutputs(parfevalOnAll(pool,@maxNumCompThreads,1));
    if iscell(limits), limits=cell2mat(limits); end
    assert(all(limits(:)==threads),'Worker computational thread limits were not applied.');
    fprintf('Process pool: %d workers x %d threads\n', workers, threads);
end
