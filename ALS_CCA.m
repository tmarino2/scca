function [ u,v,sigma ] = ALS_CCA( X,Y,r_x,r_y,eta,T)
%Input: Data matrices X in d_x\times N, Y in d_y\times N
%regularization parameters r_x, r_y
%step for solver eta
%number of iterations T
%Output: top left and right canonical directions u,v
%top singular value sigma
[d_x,N] = size(X);
[d_y,~] = size(Y);
%Initialize u,v from a multivariate normal(0,1)
u = randn(d_x,1);
v = randn(d_y,1);
%Normalize u,v
u = u/sqrt(norm((1/N)*(u'*X))^2+r_x*norm(u)^2);
v = v/sqrt(norm((1/N)*(v'*Y))^2+r_y*norm(v)^2);
%Initialize parameters for svrg
M = 2;
m = N;
eta_x = eta/max(sum(abs(X).^2,1));
eta_y = eta/max(sum(abs(Y).^2,1));
%C_x = X*X'/N;
%C_y = Y*Y'/N;
%C_xy = X*Y'/N;
for t = 1:T
    u_tilde = SVRG(u,v,X,Y,r_x,M,m,eta_x);
    v_tilde = SVRG(v,u,Y,X,r_y,M,m,eta_y);
    %u_tilde2 = C_x\(C_xy*v);
    %v_tilde2 = C_y\(C_xy'*u);
    %disp('u diff')
    %disp(norm(u_tilde-u_tilde2))
    %disp(r_x*norm(u_tilde)^2+norm(u_tilde'*X/N - v'*Y/N)^2-r_x*norm(u_tilde2)^2+norm(u_tilde2'*X/N - v'*Y/N)^2)
    %disp('v diff')
    %disp(norm(v_tilde-v_tilde2))
    %disp(r_y*norm(v_tilde)^2+norm(v_tilde'*Y/N - u'*X/N)^2-r_y*norm(v_tilde2)^2+norm(v_tilde2'*Y/N - u'*X/N)^2)
    u = u_tilde/sqrt((u_tilde'*X)*(X'*u_tilde)/N+r_x*norm(u_tilde)^2);
    v = v_tilde/sqrt((v_tilde'*Y)*(Y'*v_tilde)/N+r_y*norm(v_tilde)^2);
end
sigma = abs((u'*X)*(Y'*v)/N);
end