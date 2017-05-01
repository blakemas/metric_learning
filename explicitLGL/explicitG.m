n = 15;
d = 8;
J = eye(n)-1/n*ones(n,n);
X = rand(n,d);
%X = eye(n);
% a = rand(1,d);
% X=[];
% for i=1:n-1
%     X = [X; a];
% end
% X = [X; rand(1,d)];
G = X*X';

Agg = zeros(n,n);
LGL = zeros(n,n);  
for i=1:n
    for k=1:n
        for j=1:k
            if i~=j && i~=k && j~=k
         % empirically compute
            Lt = zeros(n,n);
            Lt([i j k],[i j k]) = [0 -1 1;-1 1 0;1 0 -1];
            Agg = Agg+Lt*Lt;
            R = Lt*Lt;
            %R([i j k],[i j k])
            LGL = LGL + Lt*G*Lt;
            end
        end
    end
end

i = 3
s = 0
for k=1:n
    for j=1:k
        if i~=j && i~=k && j~=k
            s = s+ G(j,j)+G(k,k)-2*G(j,k);
        end
    end
end

ct = 0
for r=1:n
    for t=1:n
        if i~=t && i~=r && t~=r
            ct = ct+1;
            s = s+G(r,r)-2*G(r,i)+G(i,i);
        end
    end
end
i=3
D = sum(diag(G));
ut = sum(sum(triu(G,1)));
col = sum(G(:,i));
c = sum(X, 1);
as = (2*n-4)*(D)+(n^2-3*n)*G(i,i)-2*ut-2*(n-3)*col
ass = (2*n-3)*(D)+(n^2-3*n)*G(i,i)-c*c'-2*(n-3)*c*X(i,:)'
LGL(i,i)


i=3; j=4;
coli = sum(G(:,i));
colj = sum(G(:,j));
as = (n-4)*G(i,j)-(n-2)*G(j,j)-(n-2)*G(i,i)-D+coli+colj
LGL(i,j)

%(2*n-3)*sum(diag(G))*eye(n)+(n^2-3*n)*diag(diag(G))
%norm(X'*diag(diag(LGL))*X)
nxlglx = norm(X'*LGL*X)
%norm(X'*X)
M = [];
N = [];

for i=1:n
    
    M = [M; D G(i,i) sum(G(i,:))-G(i,i) sum(sum(triu(G,1)))];
    N = [N; LGL(i,i)];
end
M\N
% for i=1:n
%     for j=1:i
%         if i~=j
%             M = [M; D G(i,i) G(j,j) G(i,j)];
%             N = [N; LGL(i,j)];
%         end
%     end
% end
% 
% 
% 
% MD = [];
% ND = [];
% 
% for i=1:n
%     MD = [MD; D G(i,i)];
%     ND = [ND; LGL(i,i)];
% end
% 
% M\N;
% MD\ND;
% %Agg = 2*Agg/(n*(n-1)*(n-2))
% (2*n-3)*D+n*(n-3)*G(i,i);
% 3*n^2-6*n;
% % ij: -D -(n-2)(g_ii+g_jj)+(n-4)g_ij
% % ii: (2n-3)D+n(n-3)g_ii