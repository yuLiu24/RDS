
clc;
clear;

addpath(genpath('../edges-master/'));
%addpath(NMS);

% %% set opts for training (see edgesTrain.m)
% opts=edgesTrain();                % default options (good settings)
% opts.modelDir='models/';          % model will be in models/forest
% opts.modelFnm='modelBsds';        % model name
% opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
% opts.useParfor=0;                 % parallelize if sufficient memory
% 
% %% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
% tic, model=edgesTrain(opts); toc; % will load model if already trained
% %% set detection parameters (can set after training)
% model.opts.multiscale=0;          % for top accuracy set multiscale=1
% model.opts.sharpen=2;             % for top speed set sharpen=0
% model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
% model.opts.nThreads=4;            % max number threads for evaluation
% model.opts.nms=1;                 % set to true to enable nms
% opts = model.opts;

% HED edge maps
f=dir('./RDS_gPb/');
%f=dir('../Data/hed_BSDS_Reproduce20/annotators_SE_ignorelabel_dilate/hed_BSDS_Test_Reproduce20/');
m=size(f);

%% old NMS 
% for i=3:m
%    imgname =strcat('edgemaps/HED_BSDS_Test_TXT/',f(i).name);
%    fuse = load(imgname); 
%    fuse = single(fuse);
%    E=convTri(fuse,1);
%    [Ox,Oy]=gradient2(convTri(E,4));
%    [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
%    O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
%    %E=edgesNmsMex(E,O,1,5,1.01,opts.nThreads);
%    E=edgesNmsMex(E,O,1,5,1.01,opts.nThreads); 
%    imwrite(uint8(E*255),['edgemaps/HED_BSDS_Test_NMS/',f(i).name(1:end-8),'.png']);
%     
%    disp(i-2); 
% end

%% HED NMS
for i = 3:m
    imgname =strcat('./RDS_gPb/',f(i).name);
    %imgname =strcat('../Data/hed_BSDS_Reproduce20/annotators_SE_ignorelabel_dilate/hed_BSDS_Test_Reproduce20/',f(i).name);
    x = load(imgname); 
    E=convTri(single(x),1);
    [Ox,Oy]=gradient2(convTri(E,4));
    [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
    O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    E=edgesNmsMex(E,O,1,5,1.01,4);
    %imwrite(uint8(E*255),['./hed_baseline_scale1_NMS/',f(i).name(1:end-8),'.png']);
    imwrite(uint8(E*255),['./RDS_gPb_NMS/',f(i).name(1:end-8),'.png']);
    
    disp(i-2);
end