function [x,rows]=rom_true_action_diagnostics(A,b,spaces,reduced)
%ROM_TRUE_ACTION_DIAGNOSTICS Actual online defects and FFOM trial bound.
% All residuals use norm(b). The spaces/reduced matrices belong to one frozen
% model and one query. This routine does not compute exact inverse actions.
    n=numel(b); k=numel(spaces); beta=norm(b);
    x=zeros(n,1,'like',b); rows=table();
    if beta==0, return; end
    Q=complex(zeros(n,k+1)); Z=complex(zeros(n,k)); H=zeros(k+1,k);
    Q(:,1)=b/beta; previous_ff=1;
    for j=1:k
        B=reduced{j}; sp=spaces{j};
        if isempty(B), break; end
        z=rom_reduced_space_utils('lift',sp, ...
            B\rom_reduced_space_utils('project',sp,Q(:,j)));
        Z(:,j)=z; w=A*z; defect=norm(w-Q(:,j));
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(w,Q(:,1:j));
        g=zeros(j+1,1); g(1)=beta;
        y=H(1:j+1,1:j)\g; x=Z(:,1:j)*y;
        rho=norm(b-A*x)/beta;
        sv=svd(B); sigma_min=min(sv); condition=max(sv)/sigma_min;
        bound=previous_ff*defect;
        hcond=rcond(H(1:j,1:j));
        if hcond>eps
            yff=H(1:j,1:j)\g(1:j);
            current_ff=norm(b-A*(Z(:,1:j)*yff))/beta;
        else
            current_ff=NaN;
        end
        rows=[rows;table(j,defect,previous_ff,bound,rho,current_ff, ...
            sigma_min,condition,hcond,'VariableNames',{'layer','action_defect', ...
            'previous_FFOM_relative','product_bound','FGMRES_relative', ...
            'FFOM_relative','sigma_min_reduced','condition_reduced','rcond_H'})]; %#ok<AGROW>
        % Roundoff in the full residual may exceed a tiny exact-arithmetic bound.
        previous_ff=current_ff;
        if stop, break; end
    end
end
