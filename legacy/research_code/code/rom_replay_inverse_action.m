function [x,Q,H,Z]=rom_replay_inverse_action(A,b,mu,model)
    n=numel(b); beta=norm(b); k=numel(model);
    Q=complex(zeros(n,k+1)); Q(:,1)=b/beta;
    Z=complex(zeros(n,k)); H=complex(zeros(k+1,k)); x=zeros(n,1,'like',Q);
    for j=1:k
        l=model{j}; B=assemble(l.operators,mu);
        z=rom_reduced_space_utils('lift',l.space, ...
            B\rom_reduced_space_utils('project',l.space,Q(:,j)));
        Z(:,j)=z;
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*z,Q(:,1:j));
        % Only the final iterate is returned; intermediate reconstructions
        % do not enter the Arnoldi recurrence or subsequent inverse actions.
        if j==k || stop
            g=zeros(j+1,1); g(1)=beta; x=Z(:,1:j)*(H(1:j+1,1:j)\g);
        end
        if stop, Q=Q(:,1:j+1); return; end
    end
end

function A=assemble(ops,mu)
 a=[1,mu(2)^(-2)-1];if numel(ops)==3,a(3)=mu(1)^(-2)-1;end
 A=a(1)*ops{1};for k=2:numel(ops),A=A+a(k)*ops{k};end
end
