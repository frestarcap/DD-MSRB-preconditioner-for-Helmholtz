function path_out = helm_output_path(varargin)
%HELM_OUTPUT_PATH Build an absolute path under the project root.
    path_out = fullfile(helm_project_root(), varargin{:});
end
