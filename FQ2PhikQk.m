function [Phik, Qk] = FQ2PhikQk(F, Q, dt)

dim = length(F);

A = [
    -F          Q;
    zeros(dim)  F']*dt;
B       = expm(A);
Phik    = B(dim+1:end, dim+1:end)';
Qk      = Phik*B(1:dim, dim+1:end);

end