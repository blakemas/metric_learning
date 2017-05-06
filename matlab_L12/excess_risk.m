function err = excess_risk(Ktrue, Khat, X)
%     err = 0;
%     log_loss = 0;
%     Gtrue = X'*Ktrue*X;
%     Ghat = X'*Khat*X;
%     for t=1:num
%         q = randsample(length(M), 3);
%         i = q(1); j = q(2); k = q(3);
%         true = Gtrue(k,k)-2*Gtrue(i,k)-(Gtrue(j,j)-2*Gtrue(i,j));
%         hat = Ghat(k,k)-2*Ghat(i,k)-(Ghat(j,j)-2*Ghat(i,j));
%         score = hat*(sign(true));
%         log_loss = log_loss + log(1 + exp(-score));
%         if sign(true) ~= sign(hat)
%             err= err+1;
%         end
%     end
%     err = err/num;
% end
    n = size(X,2);
    err = 0;
    log_loss = 0;
    Gtrue = X'*Ktrue*X;
    Ghat = X'*Khat*X;
    num = 0;
    for i=1:n
        for j=1:i
            for k=1:n
                if i~=j && i~=k && j~=k
                    num = num + 1;
                    true = Gtrue(k,k)-2*Gtrue(i,k)-(Gtrue(j,j)-2*Gtrue(i,j));
                    hat = Ghat(k,k)-2*Ghat(i,k)-(Ghat(j,j)-2*Ghat(i,j));
                    score = hat*sign(true);
                    log_loss = log_loss + log(1 + exp(-score));
                    if sign(true) ~= sign(hat)
                        err= err+1;
                    end
                end
            end
        end
    end
    err = err/num;
end