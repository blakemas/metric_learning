% triplet operators
n = 10;        % number of points
T=10000;        % number of triplets
p = 5;          % dimension of x_i's

X=[];
z = randn(p,1); z = z/norm(z);
for i=1:n
    z = randn(p,1)/sqrt(p); % these are norm 1 in expectation
    X = [X z];
end
G = X'*X;           % the Gram matrix

LGL = zeros(n,n);           % empirically computed expression
LGL_alt = zeros(n,n);       % simplified 3x3 expression
for t=1:T
    I = randsample(n,3);
    i = I(1);
    j = I(2);
    k = I(3);
    
    % empirically compute
    Lt = zeros(n,n);
    Lt(I,I) = [0 -1 1;-1 1 0;1 0 -1];
    LGL = LGL + Lt*G*Lt;
    
    % Simplified expression
    sub = zeros(n,n);
    sub(I,I) = [(G(j,j)-2*G(j,k)+G(k,k)), (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (-G(i,j)+G(i,k)+G(j,k)-G(k,k)); ...
                (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (G(i,i)-2*G(i,j)+G(j,j)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k));...
                (-G(i,j)+G(i,k)+G(j,k)-G(k,k)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k)), (G(i,i)-2*G(i,k)+G(k,k))];
    LGL_alt = LGL_alt + sub;
end
LGL = LGL/T;
LGL_alt = LGL_alt/T;
% norm(LGL - LGL_alt,'fro')       % compare learned matrices

% verify sum of terms expression
% make centered data:
V = eye(n) - 1/n*ones(n,n);
X = V*rand(n,p);
G = X*X';       % centered gram matrix

% % compute sum over all triplets with a centered Gram matrix
% 
% LGL = zeros(n,n);
% for i=1:n
%     for j=1:n
%         for k=j+1:n            
%             % empirically compute for triplet i,j,k
%             I = [i;j;k];
%             Lt = zeros(n,n);
%             Lt(I,I) = [0 -1 1;-1 1 0;1 0 -1];
%             LGL = LGL + Lt*G*Lt;
%         end
%     end
% end
% 
% % build expression for LGL_alt
% D = sum(diag(G));
% LGL_alt = zeros(n,n);
% for i=1:n
%     for j=1:n
%         if i==j
%             LGL_alt(i,j) = D*(3*n - 4);     % diagonal
% %             LGL_alt(i,j) = D - G(i,i);
%         else
%             LGL_alt(i,j) = D*(2-3*n/2);     % off diagonal
%         end
%     end
% end
% T = n*nchoosek(n-1,2);
% % compare
% norm(LGL/T - LGL_alt/T, 'fro')
% imagesc(LGL/T - LGL_alt/T)


% see if we can replicate the Mii index.
% LGL = zeros(n,n);
% i = 5;      % fix some random index
% for j=1:n
%     for k=j+1:n            
%         % empirically compute for triplet i,j,k
%         I = [i;j;k];
%         Lt = zeros(n,n);
%         Lt(I,I) = [0 -1 1;-1 1 0;1 0 -1];
%         LGL = LGL + Lt*G*Lt;
%     end
% end
% Mii = LGL(i,i)
% D = sum(diag(G));
% % Mii_alt = nchoosek(n-1,2)*2*(D-G(i,i)) - (G(i,i)-D/2)
% Mii_alt = (n-1)*(D-G(i,i)) - (G(i,i)-D/2)

LGL = zeros(n,n);           % empirically computed expression
LGL_alt = zeros(n,n);       % simplified 3x3 expression
for i=1:n
    for k=1:n
        for j=1:k
            if i~=j && i~=k && j~=k
                I = [i;j;k];
                % empirically compute
                Lt = zeros(n,n);
                Lt(I,I) = [0 -1 1;-1 1 0;1 0 -1];
                LGL = LGL + Lt*G*Lt;

                % Simplified expression
                sub = zeros(n,n);
                sub(I,I) = [(G(j,j)-2*G(j,k)+G(k,k)), (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (-G(i,j)+G(i,k)+G(j,k)-G(k,k)); ...
                            (G(i,j)-G(i,k)-G(j,j)+G(j,k)), (G(i,i)-2*G(i,j)+G(j,j)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k));...
                            (-G(i,j)+G(i,k)+G(j,k)-G(k,k)), (-G(i,i)+G(i,j)+G(i,k)-G(j,k)), (G(i,i)-2*G(i,k)+G(k,k))];
                LGL_alt = LGL_alt + sub;
            end
        end
    end
end
T = n*nchoosek(n-1,2);
LGL = LGL/T;
LGL_alt = LGL_alt/T;
norm(LGL - LGL_alt,'fro')       % compare learned matrices
