%  Author: Relja Arandjelovic (relja@relja.info)

% This file contains a few examples of how to train and test CNNs for place recognition, refer to README.md for setup instructions, and to our project page for all relevant information (e.g. our paper): http://www.di.ens.fr/willow/research/netvlad/


%  The code samples use the GPU by default, if you want to use the CPU instead (very slow especially for training!), add `'useGPU', false` to the affected function calls (trainWeakly, addPCA, serialAllFeats, computeRepresentation)

% For a tiny example of running the training on a small dataset, which takes only a few minutes to run, refer to the end of this file.



% Set the MATLAB paths
setup;


% ---------- Use/test our networks

% Load our network
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
paths= localPaths();
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);

%  Compute the image representation by simply running the forward pass using
% the network `net` on the appropriately normalized image
% (see `computeRepresentation.m`).

im = imread('/Users/cameronfiore/C++/image_localization_project/data/chess/seq-03/frame-000000.color.png');
im = single(im); % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
feats= computeRepresentation(net, im, 'useGPU', false); % add `'useGPU', false` if you want to use the CPU

x = [];
