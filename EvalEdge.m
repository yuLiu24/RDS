
clc;
clear;
addpath(genpath('../edges-master/'));


%% Simple version
tic;
resDir = './RDS_SE_NMS/';
fprintf('%s\n',resDir);
gtDir = '~/Datasets/BSDS500/BSDS500/data/groundTruth/test';
edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);

figure; edgesEvalPlot(resDir,'RDS');
toc
