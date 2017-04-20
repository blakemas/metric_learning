function [X] = randl1(n,d)
X = [];
d = d+1;
for i=1:n
    a = rand(d-1,1);
    a = sort(a);
    b = zeros(d,1);
    b(1) = a(1);
    for i=2:d-1
        b(i) = a(i) - a(i-1);
    end
    b(d) = 1-a(d-1);
    b = b.*((rand(d,1) > 0.5)*2 - 1);
    X = [X; b(1:d-1)'];
end