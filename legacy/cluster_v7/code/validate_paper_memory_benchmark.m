function validate_paper_memory_benchmark
% Validate completion and true residuals of every measured fresh process.
    out=helm_output_path('outputs','paper_memory_benchmark');
    names={'halfspace','ecc','waveguide'}; limits=[1e-12,1e-8,1e-6];
    for p=1:numel(names)
        for method={'global','dd'}
            previous=[];
            for rep=1:3
                s=load(fullfile(out,sprintf('%s_%s_run%d.mat',names{p},method{1},rep)));
                if istable(s.result)
                    rho=s.result.true_residual;
                    iterations=s.result.completion_iterations;
                else
                    rho=s.result.true_residual(:);
                    iterations=s.result.completion_iterations(:);
                end
                assert(all(isfinite(rho)) && all(rho<=limits(p)));
                if rep>1, assert(isequal(previous,iterations)); end
                previous=iterations;
                fprintf('%s %s %d: residual %.6g, updates %d, object %.3f MiB\n', ...
                    names{p},method{1},rep,max(rho),max(iterations),s.model_bytes/2^20);
            end
        end
    end
    disp('PASS: all 18 fresh-process solves satisfy residual checks; repeat counts agree.');
end
