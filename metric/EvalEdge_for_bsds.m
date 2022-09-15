model_name_lst = {'tagfeeabs_redo-7'}

for i = 1:size(model_name_lst,2)
    tic;
    resDir = fullfile('./NMS_RESULTS_FOLDER/',model_name_lst{i});
    %resDir = fullfile('E:/PycharmProjects/one_shot/pr_evaluate/eval/test');
    n_num = length(dir(fullfile(resDir, '*.png')));
    fprintf('%s\n',resDir);
    
    if (n_num == 200)
        gtDir = 'E:/PycharmProjects/one_shot/BSDS500/data/groundTruth/test';
        edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075, 'thrs',99); % maxDist 0.011 0.0075
    end
    if (n_num == 654)
        gtDir = 'K:/gg/one_shot/semi(revised)/semi(revised)/NYUD/test_gt';
        edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.011, 'thrs',99); % maxDist 0.011 0.0075
    end
    
    figure; edgesEvalPlot(resDir,'HED');
    toc
end