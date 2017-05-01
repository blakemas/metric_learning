function [Khat] = triplets(Ktrue, X, Y, method)
p = size(Ktrue,1);

% cvx_solver sedumi;
cvx_begin
    % Variable representing the distance matrix
    variable Khat(p,p) symmetric semidefinite;
    % Variable representing the Likelihood function
    F = 0;
    Ghat = X'*Khat*X;
    for t=1:length(Y)
        i=Y(t,1); j=Y(t,2); k=Y(t,3);
        delta = Ghat(k,k)-2*Ghat(i,k)-(Ghat(j,j)-2*Ghat(i,j));
        F = F+log(1+exp(-delta));         
    end
    minimize(F);
    subject to
    if strcmp(method, 'nuclear')
        norm_nuc(Khat) <= norm_nuc(Ktrue);
    elseif strcmp(method, 'L12')
        L12_norm = 0;
        L12_true = 0;
        for i=1:p
            L12_norm = L12_norm + norm(Khat(i,:));
            L12_true = L12_true + norm(Ktrue(i,:));
        end
        L12_norm <= L12_true;
    end
cvx_end

