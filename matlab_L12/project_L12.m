function Mp = project_L12(M, tau)
n = size(M,1); m = size(M,2);
row_l2_norms = sqrt(sum(M.*M, 2));
w = project_L1(row_l2_norms, tau);
Mp = M./repmat(row_l2_norms,1,m).*repmat(w, 1, m);
end

function w = project_L1(v, b)
if (b < 0)
  error('Radius of L1 ball is negative: %2.3f\n', b);
end
if (norm(v, 1) < b)
  w = v;
  return;
end
u = sort(abs(v),'descend');
sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = max(0, (sv(rho) - b) / rho);
w = sign(v) .* max(abs(v) - theta, 0);
end