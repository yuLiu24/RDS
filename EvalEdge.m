
clc;
clear;
addpath(genpath('../edges-master/'));


%% Simple version
tic;
resDir = './RDS_SE_NMS/';
%resDir = '../new-hed-master/examples/hed/Results/Reproduce5/hed_BSDS_Pretrained_NMS/';
%resDir = '~/Datasets/BSDS500/BSDS500/data/groundTruth/test_3annotators_400PNG_dilate/';
fprintf('%s\n',resDir);
gtDir = '~/Datasets/BSDS500/BSDS500/data/groundTruth/test';
edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);

figure; edgesEvalPlot(resDir,'RDS');
toc
