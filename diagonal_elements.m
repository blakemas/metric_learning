n = 50; p = 40;
Cent = eye(n) - 1/n*ones(n);
D_vals = [];
Dif_vals = [];
LGL_diag = [];
LGL_norms = [];

for r = 2:p
    % generate some rank(p) points
    X=[];
    z = randn(p,1); z = z/norm(z);
    for i=1:n
        z = randn(p,1)/sqrt(r); % these are norm 1 in expectation
        X = [X z];
    end
    X = X*Cent;     % center X
    
    % project down
    [U,S,V] = svd(X, 'econ');
    for i=r+1:size(V,2)
        S(i,i) = 0;
    end
    X = U*S*V';     % rank r approximation to X
    
    % rescale
    mu = mean(sqrt(sum(abs(X).^2,1)));      % mean of column norms
    for i=1:n
        X(:,i) = X(:,i)/mu;
    end
    G = X'*X;       % compute the gram matrix
    
    D_vals = [D_vals, sum(diag(G))];
    Dif_vals = [Dif_vals, 2*sum(diag(G)) - n*max(diag(G))];
    
    % compute the real LGL:
    LGL = zeros(n,n);           % empirically computed expression
    for i=1:n
        for k=1:n
            for j=1:k
                if i~=j && i~=k && j~=k
                    I = [i;j;k];
                    % Simplified expression
                    sub = zeros(n,n);
                    sub(I,I) = [(G(j,j)-2*G(j,k)+G(k,k)), (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (-G(i,j)+G(i,k)+G(j,k)-G(k,k)); ...
                                (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (G(i,i)-2*G(i,j)+G(j,j)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k));...
                                (-G(i,j)+G(i,k)+G(j,k)-G(k,k)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k)), (G(i,i)-2*G(i,k)+G(k,k))];
                    LGL = LGL + sub;
                end
            end
        end
    end
    T = n*nchoosek(n-1,2);
    LGL = LGL/T;
    
    disp(2*norm(G)/n - norm(LGL));
    LGL_diag = [LGL_diag, max(diag(LGL))];
    LGL_norms = [LGL_norms, norm(LGL, 2)];
    
end

figure(1); hold on;
ax = 2:p;
plot(ax,D_vals);
xlabel('Rank of X');
legend('sum of diagonal');
title('Comparison of rank to diagonal elements');
hold off;

figure(2); hold on;
ax = 2:p;
plot(ax,Dif_vals);
xlabel('Rank of X');
legend('2D - n*max(G_ii)');
title('Comparison of rank to diagonal elements');
hold off;

figure(3); hold on;
ax = 2:p;
plot(ax,LGL_diag,ax,LGL_norms);
xlabel('Rank of X');
legend('max(diag(LGL))', 'norm(LGL)');
title('Comparison of rank to diagonal elements');
hold off;

figure(4); hold on;
ax = 2:p;
plot(ax,D_vals,ax,LGL_diag,ax,LGL_norms)
xlabel('Rank of X');
legend('D', 'max(diag(LGL))', 'norm(LGL)');
title('Comparison of rank to diagonal elements');
hold off;

