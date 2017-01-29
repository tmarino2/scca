function [ U_j ] = SVRG_k( U_j,V,X,Y,r_x,M,m,eta )
%SVRG to solve min_U_j tr(1/2U_j'XX'U_j/N - U_j'XY'V)
%For more details see SVRG
[~,N] = size(X);
for j=1:M
    W_0 = U_j;
    W_t = W_0;
    batch_grad = X*(X'*W_0-Y'*V)/N+r_x*W_0;
    rand_m = randi([1,m]);
    for t=1:rand_m
        i_t = randi([1,N]);
        x_i_t = X(:,i_t);
        W_t = W_t - eta*(x_i_t*(x_i_t'*(W_t-W_0)) + r_x*(W_t-W_0)+batch_grad);
    end
    U_j = W_t;
    %for testing only
    %disp(r_x*norm(U_j)^2+norm(U_j'*X/N - V'*Y/N)^2)
end
end

