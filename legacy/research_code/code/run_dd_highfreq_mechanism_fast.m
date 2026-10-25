function run_dd_highfreq_mechanism_fast(training_file,result_file,outdir,mode)
%RUN_DD_HIGHFREQ_MECHANISM Frozen-model diagnostic; all outputs go to a new dir.
% Normalized RHS probes are fixed-work comparisons, not new production timings.
if nargin<4, mode='full'; end
assert(~isfile(fullfile(outdir,'complete.json')),'Do not overwrite a completed audit.');
if ~isfolder(outdir), mkdir(outdir); end
clock0=tic; s=load(training_file,'models','resolved_problem_cfg','rhs','cfg','partition');
ss=load(result_file,'results'); candidate=ss.results.candidates{1}; clear ss
problem=helmholtz_3d_problem_interface(s.resolved_problem_cfg); b=s.rhs; nb=norm(b); cfg=s.cfg;
w=problem.cfg.omega; N=problem.N;
models=s.models; partition=s.partition; clear s
opts=struct('solver','mg','grid_size',problem.n,'mg_max_levels',cfg.mg_max_levels, ...
 'mg_coarse_max_dofs',cfg.mg_coarse_max_dofs,'mg_pre_sweeps',cfg.mg_pre_sweeps, ...
 'mg_post_sweeps',cfg.mg_post_sweeps,'mg_jacobi_weight',cfg.mg_jacobi_weight, ...
 'mg_cycles_per_apply',cfg.mg_cycles_per_apply);
gopts=struct('restart',40,'maxit',1,'tol',1e-14,'reorthogonalize',true, ...
 'orthogonalization','cgs2_mex','store_preconditioned_basis',false, ...
 'fixed_linear_preconditioner',true,'verbose',false);
if strcmp(mode,'smoke'),gopts.restart=4;end
queries=candidate.tests; rows=struct([]); cross=struct([]); normalized=struct([]);
for qi=1:numel(queries)
 q=queries(qi); mu=q.mu;
 if strcmp(mode,'smoke') && qi>1,break;end
 fprintf('QUERY_START omega=%g mu=%.3f\n',w,mu);
 mass=problem.affine.mass_base+mu*problem.affine.mass_parameter_diagonal;
 A=problem.K_pml+spdiags(mass,0,N,N); Afun=@(v) A*v;
 [M,probe,mginfo]=build_dd_mechanism_mg_probe(A+1i*cfg.shift_beta*spdiags(mass,0,N,N),problem.n,opts);
 ref=q.shifted.solution; reference_rho=norm(b-A*ref)/nb;
 assert(reference_rho<2e-6,'Saved reference failed residual verification.');
 ref_difference=max(norm(ref-q.global.solution),norm(ref-q.dd.solution));
 for names={'global','dd'}
  method=names{1}; model=models.(method);
  [states,~]=replay(Afun,b,model,mu);
  archived=q.(method).precorrection_solution;
  mismatch=norm(states{end}-archived)/max(norm(archived),realmin);
  assert(mismatch<1e-8,'Frozen ROM replay mismatch');
  for depth=1:numel(states)
   x=states{depth}; r=b-A*x; e=ref-x;
   row=struct('omega',w,'mu',mu,'method',method,'depth',depth, ...
    'rho',norm(r)/nb,'relative_field_error',norm(e)/norm(ref), ...
    'reference_residual',reference_rho,'reference_disagreement_over_error',ref_difference/max(norm(e),realmin), ...
    'replay_relative_mismatch',mismatch,'residual',dd_mechanism_component_metrics(r,problem,partition), ...
    'field_error',dd_mechanism_component_metrics(e,problem,partition));
   row.mg=probe(r,Afun);
   % Best Euclidean approximation in each third-layer space, evaluated on a common target.
   for other={'global','dd'}
    sp=models.(other{1}).layers{end}.space;
    pe=rom_reduced_space_utils('project',sp,e);
    row.(['error_capture_' other{1}])=min(1,norm(pe)^2/max(norm(e)^2,realmin));
   end
   rows=append(rows,row);
   fprintf('STATE omega=%g mu=%.3f %s L%d rho=%.5e\n',w,mu,method,depth,row.rho);
  end
  clear states x r e
 end
 % Principal query: isolate layer depth and exchange only the final space.
 if abs(mu-.317)<1e-8 || strcmp(mode,'smoke')
  labels={'GG','GGG','DD','DDD','GGD','DDG'};
  for vi=1:numel(labels)
   label=labels{vi}; layers=cell(1,numel(label));
   for j=1:numel(label)
    if label(j)=='G',layers{j}=models.global.layers{j};else,layers{j}=models.dd.layers{j};end
   end
   hybrid=struct('layers',{layers},'num_layers',numel(layers));
   [states,trace]=replay(Afun,b,hybrid,mu);xrom=states{end};clear states
   r0=b-A*xrom; e0=ref-xrom;
   cx=struct('omega',w,'mu',mu,'variant',label,'rho',norm(r0)/nb, ...
     'rom_history',trace,'field_error_relative',norm(e0)/norm(ref), ...
     'residual',dd_mechanism_component_metrics(r0,problem,partition),'mg',probe(r0,Afun));
   cross=append(cross,cx);
   rn=r0/norm(r0); fprintf('NORMALIZED_START %g %s\n',w,label);
   [correction,infog]=shifted_laplacian_gmres(Afun,M,rn,[],gopts);
   rafter=rn-A*correction;
   nr=struct('omega',w,'mu',mu,'variant',label,'iteration_count',infog.iter_count, ...
    'history',infog.relres_hist,'true_relative_residual',norm(rafter), ...
    'seconds',infog.total_time,'fixed_step_cap',gopts.restart, ...
    'after_residual',dd_mechanism_component_metrics(rafter,problem,partition));
   assert(abs(nr.history(end)-nr.true_relative_residual)<1e-8);
   nr.full_completion=struct();
   % Exact depth ablation on w24/w48, using common production tolerance.
   if ismember(w,[24 48]) && numel(label)==2 && ~strcmp(mode,'smoke')
    fo=gopts;fo.restart=40;fo.maxit=40;fo.tol=1e-6;
    [completed,ci]=shifted_laplacian_gmres(Afun,M,b,xrom,fo);
    rr=norm(b-A*completed)/nb; assert(rr<=1e-6);
    nr.full_completion=struct('iterations',ci.iter_count,'history',ci.relres_hist, ...
      'true_residual',rr,'seconds',ci.total_time);
   end
   normalized=append(normalized,nr);
   save(fullfile(outdir,sprintf('w%d_mu%.3f_%s_vectors.mat',w,mu,label)), ...
    'r0','e0','rafter','trace','mu','label','-v7.3','-nocompression');
   writejson(fullfile(outdir,'normalized.json'),normalized);
   writejson(fullfile(outdir,'cross_spaces.json'),cross);
   fprintf('NORMALIZED_DONE %g %s rho40=%.5e\n',w,label,nr.true_relative_residual);
  end
 end
 writejson(fullfile(outdir,'components.json'),rows);
 clear M probe mginfo A mass ref archived hybrid r0 e0 rn rafter correction completed
end
result=struct('omega',w,'N',N,'source_training',training_file,'source_results',result_file, ...
 'mode',mode,'matlab_version',version,'node',getenv('SLURMD_NODENAME'), ...
 'job_id',getenv('SLURM_JOB_ID'),'elapsed_seconds',toc(clock0),'validated',true, ...
 'gmres_probe_options',gopts,'mg_options',opts,'shift_beta',cfg.shift_beta);
writejson(fullfile(outdir,'complete.json'),result);
fprintf('DD_MECHANISM_PASS omega=%g\n',w);
end
function [states,trace]=replay(Afun,b,model,mu)
 n=numel(b);k=model.num_layers;nb=norm(b);Q=complex(zeros(n,k+1));Z=complex(zeros(n,k));H=complex(zeros(k+1,k));
 Q(:,1)=b/nb;states=cell(1,k);trace=ones(k+1,1);
 for j=1:k
  l=model.layers{j};z=rom_reduced_space_utils('lift',l.space,(l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,Q(:,j)));
  Z(:,j)=z;v=Afun(z);
  for pass=1:2
   for i=1:j,hh=Q(:,i)'*v;H(i,j)=H(i,j)+hh;v=v-Q(:,i)*hh;end
  end
  H(j+1,j)=norm(v);assert(H(j+1,j)>1e-13,'Unexpected ROM breakdown');Q(:,j+1)=v/H(j+1,j);
  g=complex(zeros(j+1,1));g(1)=nb;states{j}=Z(:,1:j)*(H(1:j+1,1:j)\g);
  trace(j+1)=norm(b-Afun(states{j}))/nb;
 end
end
function s=append(s,r)
 if isempty(s),s=r;else,s(end+1)=r;end
end
function writejson(path,value)
 f=fopen(path,'w');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(jsonsafe(value))); 
end

function x=jsonsafe(x)
 if isnumeric(x) && ~isreal(x),x=struct('real',real(x),'imag',imag(x));
 elseif isstruct(x)
  names=fieldnames(x);for k=1:numel(x),for j=1:numel(names),x(k).(names{j})=jsonsafe(x(k).(names{j}));end,end
 elseif iscell(x),x=cellfun(@jsonsafe,x,'UniformOutput',false);end
end
