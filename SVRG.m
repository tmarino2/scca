function [ u_j ] = SVRG( u_j,v,X,Y,r_x,M,m,eta )
%Minimizes f(u) = 1/N\sum (1/2|u'x_i-v'y_i|^2+r_x/2|u|^2)
%u_j is the initial value of u
%M is number of steps of outer loop
%m is number of steps of inner loop
%eta is step size
[~,N] = size(X);
for j=1:M
    w_0 = u_j;
    w_t = w_0;
    batch_grad = X*(X'*w_0-Y'*v)/N+r_x*w_0;
    rand_m = randi([1,m]);
    for t=1:rand_m
        i_t = randi([1,N]);
        x_i_t = X(:,i_t);
        w_t = w_t - eta*((x_i_t'*(w_t-w_0))*x_i_t + r_x*(w_t-w_0)+batch_grad);
        %w_t = w_t - eta*((x_i_t*x_i_t' + r_x*eye(d_x))*(w_t-w_0)+batch_grad);
        %norm((x_i_t'*(w_t-w_0))*x_i_t)
    end
    u_j = w_t;
    %for testing only
    %r_x*norm(u_j)^2+norm(u_j'*X/N - v'*Y/N)^2
end
end

