
function a = value(x, param)

switch param.type
    case 1, a = 0; % nothing
    case 2, a = param.weight * sum( abs(x) ); % L1 norm
    case 5, % clustering norm
        [a,b] = sort(x,'descend');
        n = length(x);
        a = param.weight / n / n * sum( (n - 2*(1:n)' + 1 ) .* a );
    case 6, % auc loss
        a = .5 * sum( x.^2 .* ( param.ep * param.np + param.em * param.nm - param.gamma ) ) - .5 * ( x' * param.ep)^2 - .5 * ( x' * param.em)^2;
end