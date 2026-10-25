function rows=benchmark_dd_online_optimization
%BENCHMARK_DD_ONLINE_OPTIMIZATION Paired, serial, frozen-model verification.
% The before functions are archived in the audit output directory. Each query
% runs one warm-up and five measured repetitions in rotating method order.
root=helm_output_path('outputs','reviewer_20260909');
out=helm_output_path('outputs','dd_online_audit_20260910');
addpath(out);cleanup=onCleanup(@()rmpath(out)); %#ok<NASGU>
rows=table();checks=table();names={'global','dd'};
for scale=[1 1.5 2 4]
 folder=fullfile(root,sprintf('frequency_%g',scale));
 p=load(fullfile(folder,'problem.mat'),'ops','b','c');models=cell(1,2);
 for v=1:2,d=load(fullfile(folder,[names{v} '.mat']),'model');models{v}=d.model;end
 for qi=1:size(p.c.queries,1)
  mu=p.c.queries(qi,:);alpha=mu(2)^(-2)-1;A=p.ops{1}+alpha*p.ops{2};
  w2=max(full(diag(p.ops{2})));mass=w2*ones(numel(p.b),1)+alpha*full(diag(p.ops{2}));
  for v=1:2
   [x0,Q0,H0,Z0]=rom_replay_inverse_action_before(A,p.b,mu,models{v});
   [x,Q,H,Z]=rom_replay_inverse_action(A,p.b,mu,models{v});
   exact=isequal(x,x0)&&isequal(Q,Q0)&&isequal(H,H0)&&isequal(Z,Z0);
   assert(exact,'The allocation/reconstruction optimization must preserve every returned value.');
   rho=norm(p.b-A*x)/norm(p.b);
   checks=[checks;table(scale,mu(2),string(names{v}),exact,rho,'VariableNames', ...
      {'scale','c_right','variant','bitwise_equal','post_ROM_residual'})]; %#ok<AGROW>
  end
  clear x0 Q0 H0 Z0 x Q H Z;
  for rep=0:5
   order=circshift(1:4,[0,mod(rep+qi,4)]);
   for id=order
    v=1+mod(id-1,2);optimized=id>2;tt=tic;setup=0;iters=0;
    if optimized,x=rom_replay_inverse_action(A,p.b,mu,models{v});
    else,x=rom_replay_inverse_action_before(A,p.b,mu,models{v});end
    pre=norm(p.b-A*x)/norm(p.b);prefix=toc(tt);
    if pre>p.c.tol
     [M,mi]=build_shifted_laplacian_preconditioner(A+1i*.2*spdiags(mass,0,numel(p.b),numel(p.b)),struct('solver','ilu','ilu_droptol',1e-3));
     setup=mi.setup_time;
     [x,si]=shifted_laplacian_gmres(@(v)A*v,M,p.b,x,struct('tol',p.c.tol,'restart',40,'maxit',20,'reorthogonalize',true));
     iters=si.iter_count;clear M;
    end
    rho=norm(p.b-A*x)/norm(p.b);elapsed=toc(tt);assert(rho<=p.c.tol);
    if rep>0
     rows=[rows;table(scale,mu(2),string(names{v}),optimized,rep,elapsed,prefix,setup,iters,pre,rho,'VariableNames', ...
      {'scale','c_right','variant','optimized','repetition','total_s','prefix_s','completion_setup_s','completion_iterations','post_ROM_residual','true_residual'})]; %#ok<AGROW>
    end
   end
   fprintf('scale=%g query=%.4f repetition=%d verified\n',scale,mu(2),rep);
   writetable(rows,fullfile(out,'paired_timings.csv'));
  end
  writetable(checks,fullfile(out,'equivalence.csv'));
 end
end
fprintf('PASS: %d paired timings, %d exact trajectory comparisons\n',height(rows),height(checks));
end
