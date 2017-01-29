function [ U,V,obj ] = CCALin( X,Y,r_x,r_y,eta,T,k )
%CCALin implementation
%Input: Data matrices X in d_x\times N, Y in d_y\times N
%regularization parameters r_x, r_y
%step for solver eta
%number of iterations T
%Output: top-k left and right canonical directions U,V
%value of objective obj
[d_x,N] = size(X);
[d_y,~] = size(Y);
U = randn(d_x,k);
V = randn(d_y,k);
%Orthonormalize wrt to XX'/N and YY'/N
U = GS_gen(U,X/sqrt(N),r_x);
V = GS_gen(V,Y/sqrt(N),r_y);
%Initialize parameters for svrg
M = 2*k;
m = N;
eta_x = eta/max(sum(abs(X).^2,1));
eta_y = eta/max(sum(abs(Y).^2,1));
for t=1:T
    G_t = inv((U'*X)*(X'*U)/N + r_x*(U'*U) + (V'*Y)*(Y'*V)/N + r_y*(V'*V))\((U'*X)*(Y'*V)/N + (V'*Y)*(X'*U)/N); %this better be kxk
    U_tilde = SVRG_k(U*G_t,V,X,Y,r_x,M,m,eta_x);
    V_tilde = SVRG_k(V*G_t,U,Y,X,r_y,M,m,eta_y);
    U = GS_gen(U_tilde,X/sqrt(N),r_x);
    V = GS_gen(V_tilde,Y/sqrt(N),r_y);
end
obj = trace(abs((U'*X)*(Y'*V)/N));
end

