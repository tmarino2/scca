function [ U,V ] = CCA( X,Y,r_x,r_y )
%X and Y must be centered
    Cxy = X*Y';
    Cx = X*X';
    Cy = Y*Y';
    Rx = inv(Cx + r_x*eye(size(Cx)));
    Ry = inv(Cy + r_y*eye(size(Cy)));
    [U,~,~] = eig(Rx*Cxy*(Ry*Cxy'));
    [V,~,~] = eig(Ry*Cxy'*(Rx*Cxy));


