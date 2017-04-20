clear
close all

n=1000;
dmax=100;

for d=1:dmax
    X = randn(d,n)/sqrt(d);
    for t=1:10000
        i=randsample((1:n),3);
        d1 = norm(X(:,i(1))-X(:,i(2)))^2;
        d2 = norm(X(:,i(1))-X(:,i(3)))^2;
        diff(d,t) = abs(norm(X(:,i(1)))^2-norm(X(:,i(2)))^2);
    end
    dd(d) = mean(diff(d,:));
end

%2.5/sqrt(50)
%mean(diff(d,:))
%histogram(diff(d,:))

plot(dd)
xlabel('log dimension d')
title('log average |Dij-Dik| vs. d')

plot(log(1:dmax),log(dd))
xlabel('log dimension d')
title('log average |Dij-Dik| vs. d')
