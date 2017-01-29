function [ U ] = GS_gen( V,B, r )
%Generalized GS with inner product <,>_BB'
%Input: Matrix for inner product B
%Matrix whose columns are going to be orthogonalized V
%Regularization parameter r
%Output: Matrix with orthonormal columns U wrt BB'
%i.e. U'*B*B'*U = I
n = size(V,1);
k = size(V,2);
U = zeros(n,k);
U(:,1) = V(:,1)/sqrt((V(:,1)'*B)*(B'*V(:,1))+r*(V(:,1)'*V(:,1)));
for i = 2:k
    U(:,i) = V(:,i);
    for j = 1:i-1
       U(:,i) = U(:,i) - ( (U(:,i)'*B)*(B'*U(:,j)) + r*U(:,j)'*U(:,j))*U(:,j);
    end
    U(:,i) = U(:,i)/sqrt((U(:,i)'*B)*(B'*U(:,i)) + r*U(:,i)'*U(:,i));
end
end

