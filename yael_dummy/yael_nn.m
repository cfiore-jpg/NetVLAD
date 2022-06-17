function [idx, dis] = yael_nn (X, Q, k, distype)

    if ~exist('k', 'var'), k = 1; end
    if ~exist('distype', 'var'), distype = 2; end 
    assert (size (X, 1) == size (Q, 1));

    switch distype

    case {2,'L2'}
        % Compute half square norm
        X_nr = sum (X.^2) / 2;
        Q_nr = sum (Q.^2) / 2;

        sim = bsxfun (@plus, Q_nr', bsxfun (@minus, X_nr, Q'*X));
        %  sim = bsxfun (@minus, X_nr, Q'*X)
        %  sim = bsxfun (@plus, Q_nr', sim);

        if k == 1
            [dis, idx] = min (sim, [], 2);
        else  
            [dis, idx] = sort (sim, 2);
            dis = dis (:, 1:k);
            idx = idx (:, 1:k);
        end
  
        dis = dis' * 2;
        idx = idx';

    case {16,'COS'}
        sim = Q' * X;
                
        if k == 1
            [dis, idx] = min (sim, [], 2);
            dis = dis';
            idx = idx';
        else  
            [dis, idx] = sort (sim, 2);
            dis = dis (:, 1:k)';
            idx = idx (:, 1:k)';
        end
                 
    otherwise
        error ('Unknown distance type');
    end
end














% function [ids, dis]= yael_nn(v, q, k, distype)
%     assert( nargin<4 || distype==2 );
%     if nargin<3, k= 1; end
%     assert(k<=size(v,2));
%     
%     ids= zeros(k, size(q,2), 'int32');
%     dis= zeros(k, size(q,2), 'single');
%     
%     for iVec= 1:size(q,2)
%         ds= sum( bsxfun(@minus, v, q(:,iVec)).^2, 1 );
%         [ds, inds]= sort(ds);
%         dis(:,iVec)= ds(1:k);
%         ids(:,iVec)= inds(1:k);
%     end
% end
