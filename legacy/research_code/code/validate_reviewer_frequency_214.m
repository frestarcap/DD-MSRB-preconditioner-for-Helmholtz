function rows = validate_reviewer_frequency_214
%VALIDATE_REVIEWER_FREQUENCY_214 Audit the new algebraic models independently.
% Random-vector projection checks avoid expanding every DD column globally.
    out = helm_output_path('outputs', 'reviewer_20260909');
    folder = fullfile(out, 'frequency_4');
    s = load(fullfile(folder, 'problem.mat'), 'ops', 'b', 'c', 'p');
    assert(s.p.omega==214 && s.p.Ntot==570025);
    assert(isequal(s.c.training, [ones(24,1),linspace(0.8,0.95,24)']));
    assert(isequal(s.c.queries(:,2), [0.8053;0.8297;0.8761;0.9229;0.9431]));
    assert(isequal(s.c.eta, 1e-5*[1,1,1]) && s.c.tol==1e-10);
    old = load(fullfile(out, 'high_frequency_waveholtz', 'global_scale4_prefix_model.mat'), ...
        'A_base', 'A_right', 'rhs');
    assert(isequaln(s.ops{1},old.A_base) && isequaln(s.ops{2},old.A_right));
    assert(isequaln(s.b,old.rhs)); clear old;
    summary = readtable(fullfile(out, 'summary_train_frequency_4.csv'));
    rows = table(); rng(214);
    for method = {'global', 'dd'}
        d = load(fullfile(folder, [method{1} '.mat']), 'model', 'audit');
        basis_bytes = 0; operator_bytes = 0; index_bytes = 0;
        for j = 1:3
            sp = d.model{j}.space;
            if strcmp(sp.type, 'dense')
                basis_bytes = basis_bytes + 16*numel(sp.basis);
            else
                basis_bytes = basis_bytes + 16*sum(arrayfun(@(b)numel(b.basis),sp.blocks));
                index_bytes = max(index_bytes, 8*sum([sp.blocks.count]));
                assert(isequal(sort(vertcat(sp.blocks.idx)), (1:numel(s.b))'));
            end
            operator_bytes = operator_bytes + 16*sum(cellfun(@numel,d.model{j}.operators));
            assert(d.audit(j).tail<=1.001e-5 && d.audit(j).snapshot_residual<1e-9);
        end
        payload = (basis_bytes + operator_bytes + index_bytes)/2^20;
        expected = summary.payload_MiB(strcmp(summary.variant,method{1}));
        assert(abs(payload-expected)<1e-9, 'Persistent payload accounting differs.');
        for k = [2,12,23]
            alpha = s.c.training(k,2)^(-2)-1;
            A = s.ops{1}+alpha*s.ops{2};
            Q = complex(zeros(numel(s.b),4)); Q(:,1) = s.b/norm(s.b);
            for j = 1:3
                layer = d.model{j}; sp = layer.space;
                B = layer.operators{1}+alpha*layer.operators{2};
                v = randn(sp.k,1)+1i*randn(sp.k,1); v = v/norm(v);
                lifted = rom_reduced_space_utils('lift',sp,v);
                orth = norm(rom_reduced_space_utils('project',sp,lifted)-v);
                projected = rom_reduced_space_utils('project',sp,A*lifted);
                affine = norm(B*v-projected)/max(norm(B*v),eps);
                snapshot = norm(A*d.audit(j).E(:,k)-Q(:,j));
                assert(orth<1e-10 && affine<1e-10 && snapshot<1e-9);
                rows = [rows; table(string(method{1}),k,j,orth,affine,snapshot,payload, ...
                    'VariableNames',{'method','training_index','layer','orthogonality_defect', ...
                    'affine_projection_defect','snapshot_residual','payload_MiB'})]; %#ok<AGROW>
                z = rom_reduced_space_utils('lift',sp, ...
                    B\rom_reduced_space_utils('project',sp,Q(:,j)));
                [Q(:,j+1),~,~,stop] = rom_arnoldi_step(A*z,Q(:,1:j));
                assert(~stop, 'Unexpected early Arnoldi termination in audit.');
            end
        end
        clear d;
    end
    writetable(rows,fullfile(folder,'model_validation.csv'));
    fprintf('FREQUENCY_4_MODEL_VALIDATION_PASS: %d layer/query checks\n',height(rows));
end
