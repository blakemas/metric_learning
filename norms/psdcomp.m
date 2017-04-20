n = 100;
d = 2;
X = rand(n,d);
M = X*X';
a = zeros(n,n);
for i=1:n
    for j=1:i
        if rand() < d^2/n
            a(i,j) = 1;
            a(j,i) = 1;
        end
    end
end

cvx_begin
variable Mhat(n,n) symmetric semidefinite ;
minimize(norm(Mhat,'fro'))
subject to
for i=1:n
    for j=1:i
        if a(i,j)==1
            Mhat(i,j) == M(i,j);
        end
    end
end
cvx_end

norm(Mhat-M,'fro')/norm(M,'fro')