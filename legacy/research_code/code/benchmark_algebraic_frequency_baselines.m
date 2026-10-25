function rows=benchmark_algebraic_frequency_baselines
%BENCHMARK_ALGEBRAIC_FREQUENCY_BASELINES Refresh baselines for the consolidated table.
root=helm_output_path('outputs','reviewer_20260909');
out=helm_output_path('outputs','dd_online_audit_20260910');rows=table();
for scale=[1 1.5 2 4]
 p=load(fullfile(root,sprintf('frequency_%g',scale),'problem.mat'),'ops','b','c');
 for qi=1:size(p.c.queries,1)
  mu=p.c.queries(qi,:);alpha=mu(2)^(-2)-1;A=p.ops{1}+alpha*p.ops{2};
  w2=max(full(diag(p.ops{2})));mass=w2*ones(numel(p.b),1)+alpha*full(diag(p.ops{2}));
  for rep=0:5
   order=1:2;if mod(rep+qi,2),order=2:-1:1;end
   for method=order
    tt=tic;setup=0;iters=0;
    if method==1,variant="direct";x=A\p.b;
    else
     variant="shifted";
     [M,mi]=build_shifted_laplacian_preconditioner(A+1i*.2*spdiags(mass,0,numel(p.b),numel(p.b)),struct('solver','ilu','ilu_droptol',1e-3));setup=mi.setup_time;
     [x,si]=shifted_laplacian_gmres(@(v)A*v,M,p.b,zeros(size(p.b)),struct('tol',p.c.tol,'restart',40,'maxit',20,'reorthogonalize',true));iters=si.iter_count;clear M;
    end
    rho=norm(p.b-A*x)/norm(p.b);elapsed=toc(tt);assert(rho<=p.c.tol);
    if rep>0
     rows=[rows;table(scale,mu(2),variant,rep,elapsed,setup,iters,rho, 'VariableNames', ...
      {'scale','c_right','variant','repetition','total_s','completion_setup_s','completion_iterations','true_residual'})]; %#ok<AGROW>
    end
   end
  end
  fprintf('scale=%g query=%.4f baseline repetitions verified\n',scale,mu(2));
  writetable(rows,fullfile(out,'baseline_timings.csv'));
 end
end
fprintf('PASS: %d refreshed baseline timings\n',height(rows));
end
