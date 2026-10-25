function verify_dd_trajectory_span(outroot)
% Check the fixed-partition pure inverse-action span-closure property.
% Reads diagnostic snapshots and bases; does not alter any saved model.
if nargin<1,outroot=helm_output_path('outputs','dd_parameter_mechanism_20260915');end
dirs=dir(fullfile(outroot,'results','case_*'));rows=struct([]);maxNumCompThreads(2);
for di=1:numel(dirs)
 if ismember(dirs(di).name,{'case_00','case_101','case_102'}),continue;end
 files=dir(fullfile(dirs(di).folder,dirs(di).name,'models_*.mat'));
 for fi=1:numel(files)
  s=load(fullfile(files(fi).folder,files(fi).name),'models','c');models=s.models;
  for style={'dd','global'}
   target=[];
   for k=1:numel(models)
    if strcmp(models{k}.style,style{1})&&models{k}.fixed&&models{k}.tail==0,target=models{k}.layers{1}.space;break;end
   end
   if isempty(target),continue;end
   for k=1:numel(models)
    m=models{k};if m.fixed||~strcmp(m.style,style{1}),continue;end
    for j=1:3
     l=m.layers{j};S=l.S;projerr=0;maxcol=0;
     for col=1:size(S,2)
      c=rom_reduced_space_utils('project',target,S(:,col));
      e=S(:,col)-rom_reduced_space_utils('lift',target,c);projerr=projerr+norm(e)^2;
      if norm(S(:,col))>0,maxcol=max(maxcol,norm(e)/norm(S(:,col)));end
     end
     if strcmp(style{1},'global')
      W=l.space.basis;U=target.basis;errbasis=norm(W-U*(U'*W),'fro')/max(norm(W,'fro'),realmin);
     else
      num=0;den=0;
      for bi=1:numel(target.blocks)
       U=target.blocks(bi).basis;W=l.space.blocks(bi).basis;
       num=num+norm(W-U*(U'*W),'fro')^2;den=den+norm(W,'fro')^2;
      end
      errbasis=sqrt(num/max(den,realmin));
     end
     row=struct('case',dirs(di).name,'cohort',s.c.name,'method',m.name,'layer',j, ...
      'snapshot_relative_outside_full_solution_space',sqrt(projerr)/max(norm(S,'fro'),realmin), ...
      'max_individual_snapshot_relative_outside',maxcol, ...
      'basis_relative_outside_full_solution_space',errbasis);
     if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
    end
   end
  end
 end
end
assert(~isempty(rows));writetable(struct2table(rows),fullfile(outroot,'span_closure.csv'));
fprintf('SPAN_CLOSURE_CHECK_DONE rows=%d snapshot_max=%.6e basis_max=%.6e\n',numel(rows), ...
 max([rows.snapshot_relative_outside_full_solution_space]),max([rows.basis_relative_outside_full_solution_space]));
end
