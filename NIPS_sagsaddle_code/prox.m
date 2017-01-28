
function a = prox(x,sigma, param)

switch param.type
    case 1, a = x; % nothing
    case 2, a = sign(x) .* max( abs(x) - param.weight * sigma, 0); % L1-norm
    case 5, % clustering norm
        [a,b] = sort(x,'descend');
        n = length(x);
        weights = x(b) - (n - 2*(1:n)' + 1 ) / n /n * sigma * param.weight;
        a = zeros(n,1);
        a(b) = -pav(-weights);
    case 6, % auc loss
        w = sigma^(-1) * ( x .* (param.ep/param.np+param.em/param.nm) - param.em * ( param.ep' * x) / param.np /param.nm - param.ep * ( param.em' * x) / param.np / param.nm );
        kappa = sigma^(-1) - param.gamma;
        alpha = kappa/param.np/param.nm * sqrt( param.np * param.nm  / ( 1 + kappa/param.np) / (1 + kappa/param.nm) );
        
        d = ( 1+kappa/param.np) * param.ep + (1 + kappa/param.nm) * param.em;
        temp = w./sqrt(d);
        tempm = (param.em'*temp);
        tempp = (param.ep'*temp);
        temp = temp + (1/(1-alpha^2)-1)*(param.ep*tempp/param.np+param.em*tempm/param.nm) + alpha/(1-alpha^2)/sqrt(param.np*param.nm)*(param.ep*tempm+param.em*tempp);
        a = temp ./ sqrt(d);
end