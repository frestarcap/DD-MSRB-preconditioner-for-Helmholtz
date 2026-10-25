function tests = test_smoke
%TEST_SMOKE Every experiment driver runs end to end on a small problem.
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    tc.TestData.root = tempname;
    setenv('DMSRB_RESULTS', tc.TestData.root);
end

function teardownOnce(tc)
    setenv('DMSRB_RESULTS', '');
    if isfolder(tc.TestData.root), rmdir(tc.TestData.root, 's'); end
end

function test_experiment1(~)
    run_exp1('smoke');
end

function test_experiment2(~)
    run_exp2('smoke');
end

function test_experiment3(~)
    run_exp3('verify');
end
