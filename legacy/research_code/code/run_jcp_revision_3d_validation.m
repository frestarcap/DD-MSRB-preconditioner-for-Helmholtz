function result = run_jcp_revision_3d_validation(source, report_file, outdir, query_index, mode)
% Frozen omega48 validation: new queries, rotated repeats and depth controls.
% Full-order assembly/model loading are excluded from online timings. Each
% recorded solve builds its own shifted-MG when completion is required.
% Production data and all original models are read-only inputs.
if nargin < 5, mode='production'; end
assert(ismember(mode,{'verify','production'}));
if ~isfolder(outdir), mkdir(outdir); end
assert(~isfile(fullfile(outdir,'complete.json')),'Completed run exists.');
run_clock=tic;
train=linspace(.2,.6,8);
fractions=[.31 .67 .43 .59 .37 .71 .53];
added=train(1:end-1)+diff(train).*fractions;
queries=[.237 .317 .483 added];
validateattributes(query_index,{'numeric'},{'scalar','integer','>=',1,'<=',10});
mu=queries(query_index);
assert(all(abs(mu-train)>1e-10));
old_query=query_index<=3;
if old_query, repetitions=3; else, repetitions=1; end
saved=load(source,'models','cfg','resolved_problem_cfg','rhs');
cfg=saved.cfg; problem_cfg=saved.resolved_problem_cfg; b=saved.rhs;
assert(problem_cfg.omega==48 && cfg.num_layers==3);
assert(max(abs(problem_cfg.train_params(:)-train(:)))<1e-12);
assert(max(abs(cfg.layer_tail_tolerances(:)-[.003;.008;.02]))<1e-12);
assert(cfg.shift_beta==2 && cfg.gmres_restart==40 && cfg.gmres_maxit==40);
assert(cfg.gmres_tol==1e-6 && cfg.mg_coarse_max_dofs==12000);
models=saved.models; clear saved
problem=helmholtz_3d_problem_interface(problem_cfg); N=problem.N;
assert(N==5543003 && norm(b)>0);
mass=problem.affine.mass_base+mu*problem.affine.mass_parameter_diagonal;
A=problem.K_pml+spdiags(mass,0,N,N);
Afun=@(v) A*v;
opts=struct('solver','mg','grid_size',problem.n, ...
 'mg_max_levels',cfg.mg_max_levels,'mg_coarse_max_dofs',cfg.mg_coarse_max_dofs, ...
 'mg_pre_sweeps',cfg.mg_pre_sweeps,'mg_post_sweeps',cfg.mg_post_sweeps, ...
 'mg_jacobi_weight',cfg.mg_jacobi_weight,'mg_cycles_per_apply',cfg.mg_cycles_per_apply);
gmopts=struct('restart',cfg.gmres_restart,'maxit',cfg.gmres_maxit, ...
 'tol',cfg.gmres_tol,'reorthogonalize',cfg.gmres_reorthogonalize, ...
 'orthogonalization','cgs2_mex','store_preconditioned_basis',false, ...
 'fixed_linear_preconditioner',true,'verbose',false);
assert(exist('helm_gmres_cgs2_mex','file')==3,'Validated MEX kernel required.');
report=jsondecode(fileread(report_file)); assert(report.all_converged && report.omega==48);
if old_query, assert(abs(report.tests(query_index).mu-mu)<1e-12); end
assert(abs(models.global.persistent_bytes/2^20-report.global_persistent_MiB)<1e-8);
assert(abs(models.dd.persistent_bytes/2^20-report.dd_persistent_MiB)<1e-8);
assert(abs(payload(models.global,3,N,false)-models.global.persistent_bytes)<1);
assert(abs(payload(models.dd,3,N,true)-models.dd.persistent_bytes)<1);
names={'global3','dd3','global2'};
model_names={'global','dd','global'}; depths=[3 3 2];
validation=struct([]);
for k=1:3
 model=models.(model_names{k});
 [x,trace]=replay(Afun,b,model,mu,depths(k),cfg.gmres_reorthogonalize);
 vr=struct('method',names{k},'depth',depths(k),'post_rom_residual',trace(end), ...
  'history',trace,'basis_MiB',sum(cellfun(@(l) l.basis_bytes,model.layers(1:depths(k))))/2^20, ...
  'persistent_MiB',payload(model,depths(k),N,strcmp(model_names{k},'dd'))/2^20);
 if old_query && depths(k)==3
  rr=report.tests(query_index).(model_names{k});
  expected=rr.rom_residual;
  assert(abs(trace(end)-expected)<max(1e-11,abs(expected)*1e-5),'Frozen trajectory mismatch.');
  vr.archived_post_rom_residual=expected;
 else, vr.archived_post_rom_residual=NaN;
 end
 validation=append(validation,vr); clear x
end
metadata=struct('omega',48,'N',N,'mu',mu,'query_index',query_index, ...
 'query_kind',ternary(old_query,'original_repeated','new_stratified'), ...
 'all_queries',queries,'train_parameters',train,'stratified_fractions',fractions, ...
 'repetitions',repetitions,'full_warmup_each_rom',old_query, ...
 'baseline_repetitions',1,'source',source,'source_report',report_file, ...
 'node',getenv('SLURMD_NODENAME'),'job_id',getenv('SLURM_JOB_ID'), ...
 'array_job_id',getenv('SLURM_ARRAY_JOB_ID'),'matlab',version, ...
 'thread_cap',maxNumCompThreads,'mg_options',opts,'gmres_options',gmopts, ...
 'shift_beta',cfg.shift_beta,'assembly_and_load_excluded',true, ...
 'per_solve_mg_setup_included',true,'frozen_replay_checks',validation);
writejson(fullfile(outdir,'protocol.json'),metadata);
fprintf('FROZEN_QUERY_VALIDATED index=%d mu=%.12g N=%d mode=%s\n',query_index,mu,N,mode);
if strcmp(mode,'verify')
 result=metadata; result.validated=true; result.elapsed_s=toc(run_clock);
 writejson(fullfile(outdir,'complete.json'),result);
 fprintf('JCP_REVISION_3D_VERIFY_PASS\n'); return;
end
rows=struct([]); histories=struct([]);
% Warm-ups are full solves at the three original parameters only. Added
% parameters are a separate single-run validation cohort, not repeat samples.
first_rep=1; if old_query, first_rep=0; end
for rep=first_rep:repetitions
 order=circshift(1:3,[0,mod(query_index+rep-1,3)]);
 for pos=1:3
  k=order(pos); method=names{k}; model=models.(model_names{k});
  fprintf('SOLVE_START q=%d mu=%.12g method=%s rep=%d\n',query_index,mu,method,rep);
  timer=tic; tr=tic;
  [x,trace]=replay(Afun,b,model,mu,depths(k),cfg.gmres_reorthogonalize);
  rom_s=toc(tr); pre=trace(end); setup_s=0; completion_s=0; nit=0; mg_MiB=0;
  history=pre; flag=0;
  if pre>cfg.gmres_tol
   ts=tic;
   [M,mg]=build_shifted_laplacian_preconditioner(A+1i*cfg.shift_beta*spdiags(mass,0,N,N),opts);
   setup_s=toc(ts); mg_MiB=mg.factor_bytes/2^20;
   [x,info]=shifted_laplacian_gmres(Afun,M,b,x,gmopts);
   completion_s=info.total_time; nit=info.iter_count; history=info.relres_hist; flag=info.flag;
  end
  rho=norm(b-A*x)/norm(b); total_s=toc(timer); converged=isfinite(rho)&&rho<=cfg.gmres_tol;
  row=struct('query_index',query_index,'parameter',mu,'method',method,'repetition',rep, ...
   'position',pos,'rom_s',rom_s,'setup_s',setup_s,'completion_s',completion_s, ...
   'total_s',total_s,'completion_iterations',nit,'precompletion_residual',pre, ...
   'true_residual',rho,'converged',converged,'flag',flag,'mg_MiB',mg_MiB, ...
   'persistent_MiB',validation(k).persistent_MiB,'node',metadata.node);
  rows=append(rows,row);
  histories=append(histories,struct('method',method,'repetition',rep,'rom_history',trace,'completion_history',history));
  writetable(struct2table(rows),fullfile(outdir,'timings.csv'));
  writejson(fullfile(outdir,'histories.json'),histories);
  fprintf('SOLVE_DONE q=%d method=%s rep=%d total=%.6f it=%d rho=%.9e accepted=%d\n',query_index,method,rep,total_s,nit,rho,converged);
  clear M mg info x model
 end
end
% One independently constructed full-order baseline per parameter. Its time
% is explicitly single-run; paired ROM conclusions use the cohorts above.
fprintf('BASELINE_START q=%d mu=%.12g\n',query_index,mu);
timer=tic;ts=tic;
[M,mg]=build_shifted_laplacian_preconditioner(A+1i*cfg.shift_beta*spdiags(mass,0,N,N),opts);
setup_s=toc(ts);
[x,info]=shifted_laplacian_gmres(Afun,M,b,[],gmopts);
rho=norm(b-A*x)/norm(b);total_s=toc(timer);converged=isfinite(rho)&&rho<=cfg.gmres_tol;
row=struct('query_index',query_index,'parameter',mu,'method','shifted_mg','repetition',1, ...
 'position',4,'rom_s',0,'setup_s',setup_s,'completion_s',info.total_time,'total_s',total_s, ...
 'completion_iterations',info.iter_count,'precompletion_residual',1,'true_residual',rho, ...
 'converged',converged,'flag',info.flag,'mg_MiB',mg.factor_bytes/2^20,'persistent_MiB',0,'node',metadata.node);
rows=append(rows,row);
histories=append(histories,struct('method','shifted_mg','repetition',1,'rom_history',1,'completion_history',info.relres_hist));
writetable(struct2table(rows),fullfile(outdir,'timings.csv'));
writejson(fullfile(outdir,'histories.json'),histories);
result=metadata; result.all_converged=all([rows.converged]);result.elapsed_s=toc(run_clock);
result.recorded_solve_count=sum([rows.repetition]>0);result.warmup_solve_count=sum([rows.repetition]==0);
writejson(fullfile(outdir,'complete.json'),result);
assert(result.all_converged,'One or more solves failed; raw records retained.');
fprintf('JCP_REVISION_3D_PASS q=%d mu=%.12g recorded=%d elapsed=%.3f\n',query_index,mu,result.recorded_solve_count,result.elapsed_s);
end

function [x,trace]=replay(A,b,model,mu,steps,reorth)
n=numel(b);nb=norm(b);Q=complex(zeros(n,steps+1));Z=complex(zeros(n,steps));H=complex(zeros(steps+1,steps));
Q(:,1)=b/nb;trace=ones(steps+1,1);x=complex(zeros(n,1));
for j=1:steps
 l=model.layers{j};
 if l.rank==0,trace=trace(1:j);break;end
 z=rom_reduced_space_utils('lift',l.space,(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,Q(:,j)));
 Z(:,j)=z;v=A(z);
 for i=1:j,H(i,j)=Q(:,i)'*v;v=v-H(i,j)*Q(:,i);end
 if reorth
  for i=1:j,h=Q(:,i)'*v;H(i,j)=H(i,j)+h;v=v-h*Q(:,i);end
 end
 H(j+1,j)=norm(v);if H(j+1,j)>1e-13,Q(:,j+1)=v/H(j+1,j);end
 g=complex(zeros(j+1,1));g(1)=nb;x=Z(:,1:j)*(H(1:j+1,1:j)\g);
 trace(j+1)=norm(b-A(x))/nb;
 if H(j+1,j)<=1e-13,trace=trace(1:j+1);break;end
end
end

function bytes=payload(model,depth,N,isdd)
bytes=sum(cellfun(@(l) l.basis_bytes+32*l.rank^2,model.layers(1:depth)))+8*N*isdd;
end
function values=append(values,row)
if isempty(values),values=row;else,values(end+1)=row;end
end
function s=ternary(t,a,b)
if t,s=a;else,s=b;end
end
function writejson(path,value)
f=fopen([path '.tmp'],'w');assert(f>=0);fprintf(f,'%s\n',jsonencode(value));fclose(f);
movefile([path '.tmp'],path,'f');
end
