% triplet operators
n = 100;        % number of points
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

norm(LGL - LGL_alt,'fro')