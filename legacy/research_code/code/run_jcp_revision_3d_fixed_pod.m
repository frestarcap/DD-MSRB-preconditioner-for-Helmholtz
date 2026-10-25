function result=run_jcp_revision_3d_fixed_pod(source, report_file, outdir, mode)
% Complete solution-snapshot POD and local POD from existing omega48 caches.
% PREPARE validates and compresses cached training solutions, without solves.
% PRODUCTION uses pure Galerkin initial fields, then the validated MG solver.
if nargin<4, mode='prepare'; end
assert(ismember(mode,{'prepare','production'}));
if ~isfolder(outdir),mkdir(outdir);end
timer=tic; saved=load(source,'cfg','resolved_problem_cfg','rhs','partition');
cfg=saved.cfg; pcfg=saved.resolved_problem_cfg; b=saved.rhs; partition=saved.partition; clear saved
train=linspace(.2,.6,8); queries=[.237 .317 .483];
assert(pcfg.omega==48 && cfg.num_layers==3 && numel(partition.blocks)==37);
assert(max(abs(pcfg.train_params(:)-train(:)))<1e-12);
assert(cfg.snapshot_tol==1e-7 && cfg.gmres_tol==1e-6);
assert(cfg.shift_beta==2 && cfg.gmres_restart==40 && cfg.gmres_maxit==40 && cfg.mg_coarse_max_dofs==12000);
problem=helmholtz_3d_problem_interface(pcfg); N=problem.N;
assert(N==5543003 && norm(b)>0 && exist('helm_gmres_cgs2_mex','file')==3);
report=jsondecode(fileread(report_file)); assert(report.all_converged && report.omega==48);
assert(max(abs([report.tests.mu]-queries))<1e-12);
names={'global_pod8','dd_pod8','dd3'};
files=cellfun(@(s)fullfile(outdir,[s '.mat']),names,'UniformOutput',false);
if strcmp(mode,'prepare')
 assert(~isfile(fullfile(outdir,'prepare_complete.json')),'Prepared models already exist.');
 raw=load(source,'models'); originals=raw.models; clear raw
 assert(abs(originals.dd.persistent_bytes/2^20-report.dd_persistent_MiB)<1e-8);
 S=complex(zeros(N,8)); cache_rows=struct([]); cache_dir=fullfile(fileparts(source),'snapshot_cache','gaussian_waveguide','global','layer_01');
 for k=1:8
  file=fullfile(cache_dir,sprintf('sample_%03d.mat',k)); t=tic;
  item=load(file,'payload'); cache=item.payload; clear item
  load_s=toc(t); current=pcfg; current.test_params=cache.problem_cfg.test_params;
  assert(isequaln(current,cache.problem_cfg) && cache.mu==train(k));
  assert(cache.policy.snapshot_tol==cfg.snapshot_tol && cache.policy.num_layers==3);
  assert(cache.policy.shift_beta==cfg.shift_beta && cache.policy.snapshot_restart==40);
  rhs_difference=norm(cache.rhs-b/norm(b))/norm(cache.rhs);
  assert(rhs_difference<1e-14 && numel(cache.snapshot)==N);
  t=tic; residual=cache.rhs-apply_A(problem,train(k),cache.snapshot);
  rho=norm(residual)/norm(cache.rhs); verify_s=toc(t); clear residual
  assert(isfinite(rho)&&rho<=cfg.snapshot_tol*(1+1e-6),'Cached solution failed original operator check.');
  S(:,k)=cache.snapshot;
  row=struct('sample',k,'parameter',train(k),'cache_file',file,'cached_true_residual',cache.info.relative_residual, ...
   'recomputed_true_residual',rho,'rhs_difference',rhs_difference,'read_s',load_s,'verify_s',verify_s, ...
   'historical_solve_total_s',cache.info.total_time,'historical_setup_s',cache.info.setup_time, ...
   'historical_iterations',cache.info.iteration_count,'cache_job_id',cache.job_id);
  cache_rows=append(cache_rows,row);
  fprintf('FIXED48_CACHE sample=%d rho=%.9e passed\n',k,rho); clear cache
 end
 writetable(struct2table(cache_rows),fullfile(outdir,'cache_validation.csv'));
 training_cost=struct('historical_shared_first_wall_s',originals.global.shared_snapshot_time, ...
  'historical_global_stage1_snapshot_wall_s',originals.global.stage_wall_seconds(1,1), ...
  'historical_per_sample_solver_seconds',[cache_rows.historical_solve_total_s], ...
  'historical_sum_worker_wall_s',sum([cache_rows.historical_solve_total_s]), ...
  'historical_training_metadata',report.metadata, ...
  'note','Cached work is reused without new inverse solves. Old 8-worker/4-thread solve cost and new 8-thread construction cost are separate; cache read time is not full offline cost.');
 t=tic; [U,D,~]=svd(S,'econ'); sg=diag(D); r=sum(sg>sg(1)*1e-13);
 assert(r==8,'Expected all eight global directions to be numerically independent.');
 sp=rom_reduced_space_utils('make_dense_space',U(:,1:r)); global_svd_s=toc(t); clear U
 t=tic; [Ar0,Ar1]=assemble_3d_rom_affine_sparse(problem,sp); global_projection_s=toc(t);
 model=lean_model(sp,Ar0,Ar1); model.fixed_pod=true;
 metadata=inspect_model(model,problem,S,'global_pod8',global_svd_s,global_projection_s);
 metadata.singular_values=sg; all_meta=metadata;
 save(files{1},'model','metadata','-v7.3','-nocompression'); clear sp Ar0 Ar1 model
 t=tic; sp=rom_reduced_space_utils('build_blockn_space',S,partition,struct('rank_rtol',1e-13)); local_svd_s=toc(t);
 assert(all([sp.blocks.k]<=8) && all([sp.blocks.k]>0));
 t=tic; [Ar0,Ar1]=assemble_3d_rom_affine_sparse(problem,sp); local_projection_s=toc(t);
 model=lean_model(sp,Ar0,Ar1); model.fixed_pod=true;
 metadata=inspect_model(model,problem,S,'dd_pod8',local_svd_s,local_projection_s);
 metadata.singular_values=[]; all_meta=append(all_meta,metadata);
 save(files{2},'model','metadata','-v7.3','-nocompression'); clear sp Ar0 Ar1 model S
 model=build_jsc_3d_deployment_model(originals.dd); model.fixed_pod=false;
 metadata=inspect_model(model,problem,[],'dd3',NaN,NaN); metadata.singular_values=[];
 assert(abs(metadata.payload_MiB-originals.dd.persistent_bytes/2^20)<1e-8);
 all_meta=append(all_meta,metadata);
 prefix_checks=struct([]);
 for k=1:3
  A=@(v)apply_A(problem,queries(k),v);
  [x,h]=initial_field(A,b,model,queries(k),cfg.gmres_reorthogonalize);
  expected=report.tests(k).dd.rom_residual;
  assert(abs(h(end)-expected)<=max(1e-11,abs(expected)*1e-5));
  prefix_checks=append(prefix_checks,struct('parameter',queries(k),'observed',h(end),'archived',expected)); clear x
 end
 save(files{3},'model','metadata','-v7.3','-nocompression'); clear model originals
 result=protocol(source,report_file,cfg,pcfg,names,files);
 result.models=all_meta; result.cache_validation=cache_rows; result.training_cost=training_cost;
 result.dd3_prefix_checks=prefix_checks; result.elapsed_s=toc(timer); result.passed=true;
 writejson(fullfile(outdir,'prepare_complete.json'),result);
 fprintf('JCP_FIXED48_PREPARE_PASS elapsed=%.3fs\n',result.elapsed_s); return;
end
assert(isfile(fullfile(outdir,'prepare_complete.json')),'Run prepare first.');
prepared=jsondecode(fileread(fullfile(outdir,'prepare_complete.json'))); assert(prepared.passed);
assert(~isfile(fullfile(outdir,'production_complete.json')),'Completed production exists.');
models=cell(1,3); meta=cell(1,3);
for k=1:3, s=load(files{k},'model','metadata'); models{k}=s.model;meta{k}=s.metadata;end
opts=struct('solver','mg','grid_size',problem.n,'mg_max_levels',cfg.mg_max_levels, ...
 'mg_coarse_max_dofs',cfg.mg_coarse_max_dofs,'mg_pre_sweeps',cfg.mg_pre_sweeps, ...
 'mg_post_sweeps',cfg.mg_post_sweeps,'mg_jacobi_weight',cfg.mg_jacobi_weight,'mg_cycles_per_apply',cfg.mg_cycles_per_apply);
gmopts=struct('restart',cfg.gmres_restart,'maxit',cfg.gmres_maxit,'tol',cfg.gmres_tol, ...
 'reorthogonalize',cfg.gmres_reorthogonalize,'orthogonalization','cgs2_mex', ...
 'store_preconditioned_basis',false,'fixed_linear_preconditioner',true,'verbose',false);
result=protocol(source,report_file,cfg,pcfg,names,files); result.mg_options=opts;result.gmres_options=gmopts;
result.models=prepared.models;result.training_cost=prepared.training_cost;
writejson(fullfile(outdir,'protocol.json'),result);
rows=struct([]); histories=struct([]);
for q=0:3
 if q==0,mu=.317;phase='warmup';else,mu=queries(q);phase='measured';end
 mass=problem.affine.mass_base+mu*problem.affine.mass_parameter_diagonal;
 A=problem.K_pml+spdiags(mass,0,N,N); Afun=@(v)A*v;
 order=circshift(1:3,[0,mod(q,3)]);
 for pos=1:3
  k=order(pos);model=models{k};
  fprintf('FIXED48_SOLVE_START phase=%s mu=%.12g method=%s\n',phase,mu,names{k});
  total_clock=tic; tt=tic;
  [x,h]=initial_field(Afun,b,model,mu,cfg.gmres_reorthogonalize);rom_s=toc(tt);
  pre=h(end);setup_s=0;completion_s=0;nit=0;flag=0;hist=pre;mg_MiB=0;
  if ~isfinite(pre),error('Nonfinite Galerkin initial residual.');end
  if pre>cfg.gmres_tol
   tt=tic;[M,mg]=build_shifted_laplacian_preconditioner(A+1i*cfg.shift_beta*spdiags(mass,0,N,N),opts);
   setup_s=toc(tt);mg_MiB=mg.factor_bytes/2^20;
   tt=tic;[x,info]=shifted_laplacian_gmres(Afun,M,b,x,gmopts);completion_s=toc(tt);
   nit=info.iter_count;flag=info.flag;hist=info.relres_hist;
  end
  tt=tic;rho=norm(b-A*x)/norm(b);verify_s=toc(tt);total_s=toc(total_clock);
  passed=isfinite(rho)&&rho<=cfg.gmres_tol;
  row=struct('phase',phase,'query_index',q,'parameter',mu,'method',names{k},'position',pos, ...
   'rom_s',rom_s,'setup_s',setup_s,'completion_s',completion_s,'true_check_s',verify_s,'total_s',total_s, ...
   'completion_iterations',nit,'precompletion_residual',pre,'true_residual',rho,'converged',passed, ...
   'flag',flag,'mg_MiB',mg_MiB,'payload_MiB',meta{k}.payload_MiB,'model_object_MiB',meta{k}.model_object_MiB, ...
   'node',getenv('SLURMD_NODENAME'),'job_id',getenv('SLURM_JOB_ID'));
  rows=append(rows,row); histories=append(histories,struct('phase',phase,'parameter',mu,'method',names{k}, ...
   'ROM_history',h,'completion_history',hist));
  writetable(struct2table(rows),fullfile(outdir,'timings.csv'));writejson(fullfile(outdir,'histories.json'),histories);
  fprintf('FIXED48_SOLVE_DONE phase=%s mu=%.12g method=%s time=%.4f it=%d rho=%.9e pass=%d\n',phase,mu,names{k},total_s,nit,rho,passed);
  if ~passed,save(fullfile(outdir,sprintf('failed_q%d_%s.mat',q,names{k})),'x','row','-v7.3');end
  clear x M mg info model
 end
 clear A Afun
end
result.all_converged=all([rows.converged]);result.elapsed_s=toc(timer);
result.measured_solves=nnz(strcmp({rows.phase},'measured'));result.warmup_solves=nnz(strcmp({rows.phase},'warmup'));
writejson(fullfile(outdir,'production_complete.json'),result);
assert(result.all_converged,'One or more solves failed; raw results retained.');
assert(result.measured_solves==9&&result.warmup_solves==3);
fprintf('JCP_FIXED48_PRODUCTION_PASS elapsed=%.3fs\n',result.elapsed_s);
end

function model=lean_model(sp,Ar0,Ar1)
raw=struct('layers',{{struct('space',sp,'Ar0',Ar0,'Ar1',Ar1,'rank',sp.k)}});
model=build_jsc_3d_deployment_model(raw);
end

function meta=inspect_model(model,p,S,name,svd_s,projection_s)
basis=0;ops=0;ranklist=zeros(1,numel(model.layers));block_ranks=[];
for j=1:numel(model.layers)
 l=model.layers{j};ranklist(j)=l.rank;ops=ops+16*(numel(l.Ar0)+numel(l.Ar1));
 if strcmp(l.space.type,'dense'),basis=basis+16*numel(l.space.basis);
 else,basis=basis+16*sum(arrayfun(@(z)numel(z.basis),l.space.blocks));block_ranks=[block_ranks;[l.space.blocks.k]];end %#ok<AGROW>
end
indices=shared_indices(model);index_bytes=8*sum(cellfun(@numel,indices));w=whos('model');
compression=NaN;projection_error=0;
if ~isempty(S)
 l=model.layers{1}; residual2=0;
 % The shared project/lift helper accepts individual vectors. Validate all
 % snapshot columns explicitly, without invoking its vector-only path on S.
 for col=1:size(S,2)
  c=rom_reduced_space_utils('project',l.space,S(:,col),indices);
  reconstructed=rom_reduced_space_utils('lift',l.space,c,indices);
  residual2=residual2+norm(S(:,col)-reconstructed)^2;
 end
 compression=sqrt(residual2)/norm(S,'fro');clear reconstructed c
 assert(isfinite(compression)&&compression<1e-11,'Full numerical snapshot span was not captured.');
 for mu=[.237,.483]
  coeff=complex(sin((1:l.rank).'),cos((1:l.rank).'));
  u=rom_reduced_space_utils('lift',l.space,coeff,indices);
  projected=rom_reduced_space_utils('project',l.space,apply_A(p,mu,u),indices);
  expected=(l.Ar0+mu*l.Ar1)*coeff;
  error=norm(projected-expected)/max(norm(expected),eps);projection_error=max(projection_error,error);
  assert(isfinite(error)&&error<1e-10,'Reduced coupling check failed.');
 end
end
meta=struct('method',name,'layer_ranks',ranklist,'block_ranks',block_ranks, ...
 'basis_MiB',basis/2^20,'operator_MiB',ops/2^20,'index_MiB',index_bytes/2^20, ...
 'payload_MiB',(basis+ops+index_bytes)/2^20,'model_object_MiB',w.bytes/2^20, ...
 'snapshot_projection_error',compression,'operator_projection_error',projection_error, ...
 'construction_svd_s',svd_s,'construction_reduced_assembly_s',projection_s);
end

function [x,h]=initial_field(A,b,model,mu,reorth)
indices=shared_indices(model);n=numel(b);nb=norm(b);
if model.fixed_pod
 l=model.layers{1};c=(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,b,indices);
 x=rom_reduced_space_utils('lift',l.space,c,indices);h=[1;norm(b-A(x))/nb];return;
end
depth=model.num_layers;Q=complex(zeros(n,depth+1));Z=complex(zeros(n,depth));H=complex(zeros(depth+1,depth));
Q(:,1)=b/nb;x=complex(zeros(n,1));h=1;
for j=1:depth
 l=model.layers{j};assert(l.rank>0);
 z=rom_reduced_space_utils('lift',l.space,(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,Q(:,j),indices),indices);
 Z(:,j)=z;v=A(z);
 for i=1:j,H(i,j)=Q(:,i)'*v;v=v-H(i,j)*Q(:,i);end
 if reorth,for i=1:j,t=Q(:,i)'*v;H(i,j)=H(i,j)+t;v=v-t*Q(:,i);end,end
 H(j+1,j)=norm(v);if H(j+1,j)>1e-13,Q(:,j+1)=v/H(j+1,j);end
 g=complex(zeros(j+1,1));g(1)=nb;x=Z(:,1:j)*(H(1:j+1,1:j)\g);h(end+1,1)=norm(b-A(x))/nb; %#ok<AGROW>
 if H(j+1,j)<=1e-13,break;end
end
end

function y=apply_A(p,mu,x)
y=p.K_pml*x+(p.affine.mass_base+mu*p.affine.mass_parameter_diagonal).*x;
end
function idx=shared_indices(model)
idx={};if isfield(model,'shared_block_indices'),idx=model.shared_block_indices;end
end
function s=protocol(source,report_file,cfg,pcfg,names,files)
s=struct('omega',48,'N',5543003,'source',source,'source_report',report_file,'methods',{names}, ...
 'model_files',{files},'queries',[.237 .317 .483],'warmup_parameter',.317, ...
 'training_parameters',pcfg.train_params,'numerical_rank_rtol',1e-13,'true_tolerance',cfg.gmres_tol, ...
 'snapshot_tolerance',cfg.snapshot_tol,'node',getenv('SLURMD_NODENAME'),'job_id',getenv('SLURM_JOB_ID'), ...
 'matlab',version,'threads',maxNumCompThreads,'created',datestr(now,30), ...
 'timing_note','One full warmup per method at mu=.317, then one measured solve per method at each of three original queries. Diagnostic single-run paired comparison; setup/ROM/completion/true check included; model load and common assembly excluded.');
end
function a=append(a,b)
if isempty(a),a=b;else,a(end+1)=b;end
end
function writejson(file,value)
f=fopen([file '.tmp'],'w');assert(f>=0);fprintf(f,'%s\n',jsonencode(value));fclose(f);movefile([file '.tmp'],file,'f');
end
