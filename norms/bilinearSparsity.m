n = 100;
A = rand(n/2,n);

x = rand(n,1);
x(randsample(n,95)) = 0;

b = A*x;

cvx_begin
variable xhat(n);
F = 0;
for i=1:n
    for j=1:n
        F = F+abs(xhat(i)*xhat(j));
    end
end
minimize(xhat'*eye(n)*xhat)
subject to
%xhat >= 0;
A*xhat == b;
cvx_end