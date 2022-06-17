function descriptors=my_serial(folder, imFns)

    load('/Users/cameronfiore/C++/netvlad/models/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat', 'net');
    net = relja_simplenn_tidy(net);

    opts= struct(...
        'useGPU', false, ...
        'numThreads', 12, ...
        'batchSize', 10 ...
        );

    simpleNnOpts= {'conserveMemory', true, 'mode', 'test'};
    
    relja_display('serialAllFeats: Start');
    
    net= netPrepareForTest(net);

    if opts.useGPU
        net= relja_simplenn_move(net, 'gpu');
    else
        net= relja_simplenn_move(net, 'cpu');
    end
    
    nImages= length(imFns);
    nBatches= ceil( nImages / opts.batchSize );

    prog= tic;

    for iBatch= 1:nBatches
        relja_progress(iBatch, nBatches, 'serialAllFeats', prog);
        
        iStart= (iBatch-1)*opts.batchSize +1;
        iEnd= min(iStart + opts.batchSize-1, nImages);
        
        thisNumIms= iEnd-iStart+1;
        thisImageFns = strcat(folder, imFns(iStart:iEnd));
        ims_ = {};
        for i=1:length(thisImageFns)
            ims_{i} = single(imread(thisImageFns(i)));
        end
        
        % fix non-colour images
        for iIm= 1:thisNumIms
            if size(ims_{iIm},3)==1
                ims_{iIm}= cat(3,ims_{iIm},ims_{iIm},ims_{iIm});
            end
        end
        ims= cat(4, ims_{:});
        
        ims(:,:,1,:)= ims(:,:,1,:) - net.meta.normalization.averageImage(1,1,1);
        ims(:,:,2,:)= ims(:,:,2,:) - net.meta.normalization.averageImage(1,1,2);
        ims(:,:,3,:)= ims(:,:,3,:) - net.meta.normalization.averageImage(1,1,3);
        
        if opts.useGPU
            ims= gpuArray(ims);
        end
        
        % ---------- extract features
        res= vl_simplenn(net, ims, [], [], simpleNnOpts{:});
        clear ims;
        descriptors = reshape( gather(res(end).x), [], thisNumIms );
        clear res;  
        for i=1:length(thisImageFns)
            fn = append(thisImageFns(i), '.mat');
            d = descriptors(:,i);
            save(fn, 'd');
        end
    end
end