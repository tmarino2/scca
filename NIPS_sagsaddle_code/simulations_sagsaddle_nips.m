function ret = simulations_sagsaddle_nips(ftype, gtype)

% simple simulations, for saddle point problem
% lambda/2 * ||x||^2 + f(x) - gamma/2 * ||y||^2 - g(y) + y' * K * x + a'*x - b'*y
% where f and g are one-strongly convex functions

%ARGUMENTS:
%ftype: Regularizer options for function f(x) 
%       2 for L1-Norm
%       5 for cluster norm

%gtype: Loss options for function g(y)
%       1 for Squared Hinge-loss
%       6 for AUC Loss

% sampling of full columms and rows, a single one per iteration
% requires f, proxf, g, proxg, coded by paramf, paramg
%


%clear all
seed = 1;
randn('state',seed);
rand('state',seed);
ret=-1;


total_number_of_passes = 500; % all methods will exactly run with this number of passes


% choice of function for f (on top of the strongly convex terms)

if(ftype~=2 & ftype~=5)
    fprintf('ftype argument must be:\n \t 2 (for L1-Norm regularizer) or 5 (for Cluster Norm regularizer) ! Please check..\n \t Taking default value 2.. \n');
    ftype=2;
end

if(gtype~= 1  & gtype~=6)
    fprintf('gtype argument must be:\n \t 1 (for squared hinge loss) or 6 (for AUC loss) ! Please check.. \n \t Taking default value 1..\n');
    gtype=1;
end


paramf.type=ftype;
if(paramf.type==2)
    paramf.weight = 1e-4;
else
 paramf.weight = 1e-3;
end


% choice of function for g (on top of the strongly convex terms)

paramg.type = gtype; 


%prepare synthetic data set
        n = 500;
        d = 300;
        X = randn(n,d);
        
        % planting eigenvalues
        [u,s,v] = svd(X,'econ');
        s =  1./(1:min(n,d));
        s = s / sqrt(sum(s.^2)) * sqrt(2*n); % planting ||K||_F^2 / lambda / gamma = n
        X = u * diag(s) * v';
        
        % preparing data
        xpred = randn(d,1);
        noise = randn(n,1);
        y =   sign(X * xpred + .1 * noise * norm(X * xpred,'fro') / norm( noise, 'fro'));
        
        ytrain = y(1:n/2);
        ytest = y(n/2+1:end);
        Xtrain = X(1:n/2,:);
        Xtest = X(n/2+1:end,:);
%end data set creation

K = Xtrain;
[n d ] = size(K);

factor = 1/1; % factor of 1 is the standard to try
lambda =  norm(K,'fro').^2 / n / n * factor;

if paramg.type == 1
    % for square loss
    gamma = n;
    b = ytrain;
    a = zeros(d,1);
    
    
elseif paramg.type == 6
    % for auc loss
    indp = find(ytrain==1); np = length(indp);
    indm = find(ytrain==-1); nm = length(indm);
    ytrain = ytrain([indp; indm]);
    K = K([indp; indm],:);
    gamma = np * nm / ( np + nm) % for auc loss
    paramg.np = np;
    paramg.nm = nm;
    ep = [ ones(np,1); zeros(nm,1)];
    em = [ zeros(np,1); ones(nm,1)];
    paramg.ep = ep;
    paramg.em = em;
    paramg.gamma = gamma;
    %
    b = zeros(n,1);
    a = K' * (ep / np - em / nm);
    
    % prepare test
    indp = find(ytest==1); np = length(indp);
    indm = find(ytest==-1); nm = length(indm);
    ep = [ ones(np,1); zeros(nm,1)];
    em = [ zeros(np,1); ones(nm,1)];
    
    ytest = ytest([indp; indm]);
    Xtest = Xtest([indp; indm],:);
    atest = (ep / np - em / nm);
    Atest =  diag( ep/np + em/nm) - (ep*em' + em*ep')/nm/np;
    
end

KT = K';


% constants
L = max(svds(K)) / sqrt( lambda * gamma );
Lbar = norm(K,'fro') / sqrt( lambda * gamma );


diag_K_KT = diag( K*KT );
diag_KT_K = diag( KT*K );


% get the optimal solutions by svrg-acc run for very long (10 times longer
% than usual)

% svrg - acc
fprintf('computing the optimal solution');
sigma = 1 / ( L^2 + 3 * Lbar^2 );

tau = max(Lbar * sqrt( (n+d) /n/d ) - 1,0);
sigmaadjusted = sigma * ( 1 + tau) / ( 1 + sigma * tau * ( tau + 1 ) );

s = round(1 + log( 1 + tau  )  / log (4/3)  /4   ); % note the /4 to change more often. could implement an online test when computing the full gradient.
length_epogh_svrg_acc = round(log(4) * ( 1 + 1/sigma/(1+tau)^2 ) );
maxepoch_svrgacc = 1+ceil(total_number_of_passes * 10 * n * d / ( n*d + (n+d) * length_epogh_svrg_acc ) );


xtilde = zeros(d,1);
ytilde = zeros(n,1);
xbar = zeros(d,1);
ybar = zeros(n,1);
x = xtilde;
y = ytilde;

p = diag_K_KT; p = full(p / sum(p));
q = diag_KT_K; q = full(q / sum(q));
switchvalue = 0;
for iepoch = 1:maxepoch_svrgacc
    
    
    % if we reach the maximal number of epoch in a single meta-epoch, exit
    if s==1, xbar = xtilde; ybar = ytilde;
    else
        if mod(iepoch,s)==0, xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic_early(iepoch-1); end % change xbar, ybar
    end
    
    yy = ( - a - K' * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_svrg_acc_heuristic_early(iepoch) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    
    if (iepoch > maxepoch_svrgacc/5) && (opt_svrg_acc_heuristic_early(iepoch) < opt_svrg_acc_heuristic_early(1) * 1e-12), break; end
    
    % for the first meta-epoch, stop when the decrease slows down
    if ( (switchvalue==0) && (iepoch>=3) )
        if ( opt_svrg_acc_heuristic_early(iepoch-1) - opt_svrg_acc_heuristic_early(iepoch) ) < ( opt_svrg_acc_heuristic_early(iepoch-2) - opt_svrg_acc_heuristic_early(iepoch-1) )
            xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic_early(iepoch-1);
            
        end
    end
    
    % if we do at least as well as the previous epoch, exit
    % this is a bit aggressive, we could do one more epoch before exiting
    if (opt_svrg_acc_heuristic_early(iepoch) < switchvalue),
        xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic_early(iepoch-1);
    end
    
    
    
    fprintf('.');
    
    
    Kxtilde = K * xtilde;
    KTytilde = KT * ytilde;
    x = xtilde;
    y = ytilde;
    
    is = discretesample(p,length_epogh_svrg_acc);
    js = discretesample(q,length_epogh_svrg_acc);
    
    
    
    for iter = 1:length_epogh_svrg_acc
        i = is(iter);
        j = js(iter);
        
        
        xnew = prox( ( x - ( sigma * ( 1+tau ) / lambda ) * ( KTytilde + KT(:,i) * ( y(i) - ytilde(i))/p(i) + a - tau * lambda * xbar) )/(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/lambda , paramf);
        ynew = prox( ( y + ( sigma * ( 1+tau ) / gamma ) * ( Kxtilde + K(:,j) * ( x(j) - xtilde(j))/q(j)  - b + tau * gamma * ybar) ) /(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/gamma , paramg);
        
        
        
        x = xnew;
        y = ynew;
    end
    xtilde = x;
    ytilde = y;
end
fprintf('\n')
fprintf('precision=%e\n',opt_svrg_acc_heuristic_early(end));

xast = x;
yast = y;


switch paramg.type
    case 1
        
        optimal_test_error = mean( (Xtest * x - ytest).^2)
    case 6
        pred = Xtest * x;
        optimal_test_error = .5  + atest' * pred + .5 * pred' * Atest * pred
        
end

fprintf('precision=%e\n',opt_svrg_acc_heuristic_early(end));

optimal_test_error = mean( (Xtest * x - ytest).^2)
fprintf('optimal test err=%e\n',optimal_test_error);


% forward backward
fprintf('FB');
sigma = 1/L^2 ;
x = zeros(d,1);
y = zeros(n,1);

maxiter_fb = round(total_number_of_passes/2);
for iter = 1:maxiter_fb
    
    if mod(iter,round(maxiter_fb/20))==1, fprintf('.'); end
    
    yy = ( - a - KT * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_fb(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    optxy_fb(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
    
    switch paramg.type
        case 1
            test_fb(iter) = mean( (Xtest * x - ytest).^2);
        case 6
            pred = Xtest * x;
            test_fb(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
    end
    
    xnew = prox( ( x - sigma / lambda * ( KT * y + a) )/(1+sigma) , sigma/(1+sigma)/lambda , paramf);
    ynew = prox( ( y + sigma / gamma * ( K * x - b ) ) /(1+sigma) , sigma/(1+sigma)/gamma , paramg);
    x = xnew;
    y = ynew;
end
fprintf('\n');


% fb_acc
fprintf('FB-acc');
sigma = 1 / ( 2 * L ) ;
eta = 1 / (1 + 2 * L );
theta = (1 - eta) / (1 + eta);

x = zeros(d,1);
y = zeros(n,1);
xo = zeros(d,1);
yo = zeros(n,1);


maxiter_fb_acc = round(total_number_of_passes/2);
for iter = 1:maxiter_fb_acc
    
    if mod(iter,round(maxiter_fb_acc/20))==1, fprintf('.'); end
    
    yy = ( - a - KT * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_fb_acc(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    optxy_fb_acc(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
    
    switch paramg.type
        case 1
            test_fb_acc(iter) = mean( (Xtest * x - ytest).^2);
        case 6
            pred = Xtest * x;
            test_fb_acc(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
    end
    
    
    
    
    xnew = prox(  ( x - sigma / lambda * ( KT * ( y + theta * ( y - yo) ) + a ) ) / (1 + sigma ),  sigma/(1+sigma)/lambda , paramf);
    ynew = prox(  ( y + sigma / gamma * ( K * ( x + theta * ( x - xo) ) - b ) ) / (1 + sigma )  ,  sigma/(1+sigma)/gamma , paramg);
    xo = x;
    yo = y;
    x = xnew;
    y = ynew;
end
fprintf('\n')


%plot the results 
subplot(1,3,1)
plot((0:maxiter_fb-1)*(2*n*d) / (n*d) ,log10(opt_fb) -log10(opt_fb(1)),'k-','linewidth',2); hold on
plot((0:maxiter_fb_acc-1)*(2*n*d) / (n*d) ,log10(opt_fb_acc) -log10(opt_fb_acc(1)),'k:','linewidth',2); hold on
legend('fb','fb-acc' );
title('primal-dual gap');

subplot(1,3,2)
plot((0:maxiter_fb-1)*(2*n*d) / (n*d) ,log10(optxy_fb) -log10(optxy_fb(1)),'k-','linewidth',2); hold on
plot((0:maxiter_fb_acc-1)*(2*n*d) / (n*d) ,log10(optxy_fb_acc) -log10(optxy_fb_acc(1)),'k:','linewidth',2); hold on
legend('fb','fb-acc' );
title('distance to optimizers');


subplot(1,3,3)
plot((0:maxiter_fb-1)*(2*n*d) / (n*d) ,  test_fb ,'k-','linewidth',2); hold on
plot((0:maxiter_fb_acc-1)*(2*n*d) / (n*d) , test_fb_acc ,'k:','linewidth',2); hold on
legend('fb','fb-acc' );
title('test error');



% forward backward - stochastic
fprintf('FB sto');

x = zeros(d,1);
y = zeros(n,1);
p = diag_K_KT; p = full(p / sum(p));
q = diag_KT_K; q = full(q / sum(q));


maxiter_fb_sto = round(total_number_of_passes/(n+d)*n*d);
is = discretesample(p,maxiter_fb_sto);
js = discretesample(q,maxiter_fb_sto);
opt_fb_sto = zeros(maxiter_fb_sto,1);
optxy_fb_sto = zeros(maxiter_fb_sto,1);
test_fb_sto = zeros(maxiter_fb_sto,1);

for iter = 1:maxiter_fb_sto
    
    sigma = 2 / ( iter + 1 + 8 * Lbar^2 );
    
    
    if mod(iter,round(maxiter_fb_sto/20))==1, fprintf('.'); end
    
    if mod(iter,round(maxiter_fb_sto/total_number_of_passes/10))==1, % compute train error ten times a pass
        
        yy = ( - a - KT * y);
        xx = K * x - b;
        xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
        ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
        
        opt_fb_sto(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
        optxy_fb_sto(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
        switch paramg.type
            case 1
                test_fb_sto(iter) = mean( (Xtest * x - ytest).^2);
            case 6
                pred = Xtest * x;
                test_fb_sto(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
        end
        
    else
        opt_fb_sto(iter) = opt_fb_sto(iter-1);
        optxy_fb_sto(iter) = optxy_fb_sto(iter-1);
        test_fb_sto(iter) = test_fb_sto(iter-1);
    end
    i = is(iter);
    j = js(iter);
    
    xnew = prox( ( x - sigma / lambda * ( KT(:,i) * ( y(i) / p(i) ) + a) )/(1+sigma) , sigma/(1+sigma)/lambda , paramf);
    ynew = prox( ( y + sigma / gamma * ( K(:,j) * ( x(j) / q(j) ) - b ) ) /(1+sigma) , sigma/(1+sigma)/gamma , paramg);
    
    
    x = xnew;
    y = ynew;
end
fprintf('\n');


%plot results
subplot(1,3,1)
hold on
plot((0:maxiter_fb_sto-1)*(n+d) / (n*d),log10(opt_fb_sto) -log10(opt_fb_sto(1)),'b-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto' );


subplot(1,3,2)
hold on
plot((0:maxiter_fb_sto-1)*(n+d) / (n*d),log10(optxy_fb_sto) -log10(optxy_fb_sto(1)),'b-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto' );


subplot(1,3,3)
hold on
plot((0:maxiter_fb_sto-1)*(n+d) / (n*d),test_fb_sto ,'b-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto' );




% saga - non-uniform
fprintf('saga - non-uniform');
sigma = 1 / ( max( L^2 + 3 * Lbar^2, 3*max(n,d)/2 - 1 ) );

xtilde = zeros(d,1);
ytilde = zeros(n,1);
x = zeros(d,1);
y = zeros(n,1);

Kxtilde = zeros(n,1);
KTytilde = zeros(d,1);


p = diag_K_KT + 1; p = full(p / sum(p));
q = diag_KT_K + 1; q = full(q / sum(q));

maxiter_saga = 1+ceil(total_number_of_passes / (n+d) * n * d );

is = discretesample(p,maxiter_saga);
js = discretesample(q,maxiter_saga);



for iter = 1:maxiter_saga
    
    
    if mod(iter,round(maxiter_saga/20))==1, fprintf('.'); end
    
    if mod(iter,round(maxiter_saga/total_number_of_passes/10))==1, % compute train error ten times a pass
        
        yy = ( - a - KT * y);
        xx = K * x - b;
        xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
        ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
        
        opt_saga_nonunif(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
        optxy_saga_nonunif(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
        switch paramg.type
            case 1
                test_saga_nonunif(iter) = mean( (Xtest * x - ytest).^2);
            case 6
                pred = Xtest * x;
                test_saga_nonunif(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
        end
        
    else
        opt_saga_nonunif(iter) = opt_saga_nonunif(iter-1);
        optxy_saga_nonunif(iter) = optxy_saga_nonunif(iter-1);
        test_saga_nonunif(iter) = test_saga_nonunif(iter-1);
    end
    
    
    
    i = is(iter);
    j = js(iter);
    
    tempx = KT(:,i) * ( y(i) - ytilde(i));
    tempy = K(:,j) * ( x(j) - xtilde(j));
    
    xnew = prox( ( x - sigma / lambda * ( KTytilde + tempx/p(i) + a) )/(1+sigma) , sigma/(1+sigma)/lambda , paramf);
    ynew = prox( ( y + sigma / gamma * ( Kxtilde + tempy/q(j) - b ) ) /(1+sigma) , sigma/(1+sigma)/gamma , paramg);
    
    KTytilde = KTytilde + tempx;
    Kxtilde = Kxtilde + tempy;
    xtilde(j) = x(j);
    ytilde(i) = y(i);
    
    x = xnew;
    y = ynew;
end
fprintf('\n')


% saga - uniform
fprintf('saga - uniform');
sigma = 1 / ( max( L^2 + 3 * max(n * max( sum(K.^2,2) ), d * max( sum(K.^2,1) ) ) / (lambda * gamma), 3 * max(n,d)/2 -1 ) );


xtilde = zeros(d,1);
ytilde = zeros(n,1);
x = zeros(d,1);
y = zeros(n,1);

Kxtilde = zeros(n,1);
KTytilde = zeros(d,1);


p = ones(n,1); p = full(p / sum(p));
q = ones(d,1); q = full(q / sum(q));

maxiter_saga = 1+ceil(total_number_of_passes / (n+d) * n * d );

is = discretesample(p,maxiter_saga);
js = discretesample(q,maxiter_saga);

for iter = 1:maxiter_saga
    
    
    if mod(iter,round(maxiter_saga/20))==1, fprintf('.'); end
    
    if mod(iter,round(maxiter_saga/total_number_of_passes/10))==1, % compute train error ten times a pass
        
        yy = ( - a - KT * y);
        xx = K * x - b;
        xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
        ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
        
        opt_saga_unif(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
        optxy_saga_unif(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
        switch paramg.type
            case 1
                test_saga_unif(iter) = mean( (Xtest * x - ytest).^2);
            case 6
                pred = Xtest * x;
                test_saga_unif(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
        end
        
    else
        opt_saga_unif(iter) = opt_saga_unif(iter-1);
        optxy_saga_unif(iter) = optxy_saga_unif(iter-1);
        test_saga_unif(iter) = test_saga_unif(iter-1);
    end
    
    
    
    i = is(iter);
    j = js(iter);
    
    tempx = KT(:,i) * ( y(i) - ytilde(i));
    tempy = K(:,j) * ( x(j) - xtilde(j));
    
    xnew = prox( ( x - sigma / lambda * ( KTytilde + tempx/p(i) + a) )/(1+sigma) , sigma/(1+sigma)/lambda , paramf);
    ynew = prox( ( y + sigma / gamma * ( Kxtilde + tempy/q(j) - b ) ) /(1+sigma) , sigma/(1+sigma)/gamma , paramg);
    
    
    
    KTytilde = KTytilde + tempx;
    Kxtilde = Kxtilde + tempy;
    xtilde(j) = x(j);
    ytilde(i) = y(i);
    
    x = xnew;
    y = ynew;
end
fprintf('\n')

%plot results
subplot(1,3,1)
hold on
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  log10(opt_saga_nonunif) -log10(opt_saga_nonunif(1)),'c-','linewidth',2);
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  log10(opt_saga_unif) -log10(opt_saga_unif(1)),'y-','linewidth',2);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform' );

subplot(1,3,2)
hold on
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  log10(optxy_saga_nonunif) -log10(optxy_saga_nonunif(1)),'c-','linewidth',2);
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  log10(optxy_saga_unif) -log10(optxy_saga_unif(1)),'y-','linewidth',2);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform' );


subplot(1,3,3)
hold on
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  test_saga_nonunif ,'c-','linewidth',2);
plot((0:maxiter_saga-1)*(n+d)/ (n*d),  test_saga_unif ,'y-','linewidth',2);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform' );


% svrg
fprintf('svrg')
sigma = 1 / ( L^2 + 3 * Lbar^2 );
xtilde = zeros(d,1);
ytilde = zeros(n,1);
x = xtilde;
y = ytilde;
p = diag_K_KT; p = full(p / sum(p));
q = diag_KT_K; q = full(q / sum(q));

length_epogh_svrg = round(log(4) * ( 1 + 1/sigma ) );

maxepoch_svrg = 1+ceil(total_number_of_passes * n * d / ( n*d + (n+d) * length_epogh_svrg ) );



for iepoch = 1:maxepoch_svrg
    fprintf('.');
    yy = ( - a - KT * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_svrg(iepoch) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    optxy_svrg(iepoch) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
    
    switch paramg.type
        case 1
            test_svrg(iepoch) = mean( (Xtest * x - ytest).^2);
        case 6
            pred = Xtest * x;
            test_svrg(iepoch) = .5  + atest' * pred + .5 * pred' * Atest * pred;
    end
    
    Kxtilde = K * xtilde;
    KTytilde = KT * ytilde;
    x = xtilde;
    y = ytilde;
    
    is = discretesample(p,length_epogh_svrg);
    js = discretesample(q,length_epogh_svrg);
    
    for iter = 1:length_epogh_svrg
        i = is(iter);
        j = js(iter);
        
        
        xnew = prox( ( x - sigma / lambda * ( KTytilde + KT(:,i) * ( y(i) - ytilde(i))/p(i) + a) )/(1+sigma) , sigma/(1+sigma)/lambda , paramf);
        ynew = prox( ( y + sigma / gamma * ( Kxtilde + K(:,j) * ( x(j) - xtilde(j))/q(j)  - b ) ) /(1+sigma) , sigma/(1+sigma)/gamma , paramg);
        
        
        x = xnew;
        y = ynew;
    end
    xtilde = x;
    ytilde = y;
end
fprintf('\n');

%plot results
subplot(1,3,1)
hold on
plot((0:maxepoch_svrg-1)*(n*d + (n+d) * length_epogh_svrg )/(n*d),log10(opt_svrg) -log10(opt_svrg(1)),'m-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg' );

subplot(1,3,2)
hold on
plot((0:maxepoch_svrg-1)*(n*d + (n+d) * length_epogh_svrg )/(n*d),log10(optxy_svrg) -log10(optxy_svrg(1)),'m-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg' );

subplot(1,3,3)
hold on
plot((0:maxepoch_svrg-1)*(n*d + (n+d) * length_epogh_svrg )/(n*d),test_svrg ,'m-','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg' );



% svrg - acc
fprintf('svrg-acc');
sigma = 1 / ( L^2 + 3 * Lbar^2 );

tau = max(Lbar * sqrt( (n+d) /n/d ) - 1,0);
sigmaadjusted = sigma * ( 1 + tau) / ( 1 + sigma * tau * ( tau + 1 ) );

s = round(1 + log( 1 + tau  )  / log (4/3)  /4   ); % note the /4 to change more often. could implement an online test when computing the full gradient.
length_epogh_svrg_acc = round(log(4) * ( 1 + 1/sigma/(1+tau)^2 ) );
maxepoch_svrgacc = 1+ceil(total_number_of_passes * n * d / ( n*d + (n+d) * length_epogh_svrg_acc ) );

xtilde = zeros(d,1);
ytilde = zeros(n,1);
xbar = zeros(d,1);
ybar = zeros(n,1);
x = xtilde;
y = ytilde;

p = diag_K_KT; p = full(p / sum(p));
q = diag_KT_K; q = full(q / sum(q));

for iepoch = 1:maxepoch_svrgacc
    
    if mod(iepoch,s)==0, xbar = xtilde; ybar = ytilde;  end % change xbar, ybar
    
    fprintf('.');
    yy = ( - a - KT * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_svrg_acc(iepoch) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    optxy_svrg_acc(iepoch) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
    
    switch paramg.type
        case 1
            test_svrg_acc(iepoch) = mean( (Xtest * x - ytest).^2);
        case 6
            pred = Xtest * x;
            test_svrg_acc(iepoch) = .5  + atest' * pred + .5 * pred' * Atest * pred;
    end
    
    Kxtilde = K * xtilde;
    KTytilde = KT * ytilde;
    x = xtilde;
    y = ytilde;
    
    is = discretesample(p,length_epogh_svrg_acc);
    js = discretesample(q,length_epogh_svrg_acc);
    
    
    
    for iter = 1:length_epogh_svrg_acc
        i = is(iter);
        j = js(iter);
        
        
        xnew = prox( ( x - sigma * ( 1+tau ) / lambda * ( KTytilde + KT(:,i) * ( y(i) - ytilde(i))/p(i) + a - tau * lambda * xbar) )/(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/lambda , paramf);
        ynew = prox( ( y + sigma * ( 1+tau ) / gamma * ( Kxtilde + K(:,j) * ( x(j) - xtilde(j))/q(j)  - b + tau * gamma * ybar) ) /(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/gamma , paramg);
        
        
        
        x = xnew;
        y = ynew;
    end
    xtilde = x;
    ytilde = y;
end
fprintf('\n')


% svrg - acc
fprintf('svrg-acc-heuristic');
sigma = 1 / ( L^2 + 3 * Lbar^2 );

tau = max(Lbar * sqrt( (n+d) /n/d ) - 1,0);
sigmaadjusted = sigma * ( 1 + tau) / ( 1 + sigma * tau * ( tau + 1 ) );

s = round(1 + log( 1 + tau  )  / log (4/3)  /4   ); % note the /4 to change more often. could implement an online test when computing the full gradient.
length_epogh_svrg_acc = round(log(4) * ( 1 + 1/sigma/(1+tau)^2 ) );
maxepoch_svrgacc = 1+ceil(total_number_of_passes * n * d / ( n*d + (n+d) * length_epogh_svrg_acc ) );

xtilde = zeros(d,1);
ytilde = zeros(n,1);
xbar = zeros(d,1);
ybar = zeros(n,1);
x = xtilde;
y = ytilde;

p = diag_K_KT; p = full(p / sum(p));
q = diag_KT_K; q = full(q / sum(q));
switchvalue = 0;
for iepoch = 1:maxepoch_svrgacc
    
    
    % if we reach the maximal number of epoch in a single meta-epoch, exit
    if s==1, xbar = xtilde; ybar = ytilde;
    else
        if mod(iepoch,s)==0, xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic_early(iepoch-1); end % change xbar, ybar
    end
    
    
    yy = ( - a - KT * y);
    xx = K * x - b;
    xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
    ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
    
    opt_svrg_acc_heuristic(iepoch) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
    optxy_svrg_acc_heuristic(iepoch) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
    
    
    switch paramg.type
        case 1
            test_svrg_acc_heuristic(iepoch) = mean( (Xtest * x - ytest).^2);
        case 6
            pred = Xtest * x;
            test_svrg_acc_heuristic(iepoch) = .5  + atest' * pred + .5 * pred' * Atest * pred;
    end
    
    % opt_svrg_acc_computable(iepoch) =  1/lambda * norm( KT*ytilde + a + lambda * xtilde   )^2 +  1/gamma * norm( K*xtilde - b - gamma * ytilde  )^2;
    
    % for the first meta-epoch, stop when the decrease slows down
    if ( (switchvalue==0) && (iepoch>=3) )
        if ( opt_svrg_acc_heuristic(iepoch-1) - opt_svrg_acc_heuristic(iepoch) ) < ( opt_svrg_acc_heuristic(iepoch-2) - opt_svrg_acc_heuristic(iepoch-1) )
            xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic(iepoch-1);
            
        end
    end
    
    % if we do at least as well as the previous epoch, exit
    % this is a bit aggressive, we could do one more epoch before exiting
    if (opt_svrg_acc_heuristic(iepoch) < switchvalue),
        xbar = xtilde; ybar = ytilde; switchvalue = opt_svrg_acc_heuristic(iepoch-1);
    end
    
    
    
    fprintf('.');
    
    
    Kxtilde = K * xtilde;
    KTytilde = KT * ytilde;
    x = xtilde;
    y = ytilde;
    
    is = discretesample(p,length_epogh_svrg_acc);
    js = discretesample(q,length_epogh_svrg_acc);
    
    
    
    for iter = 1:length_epogh_svrg_acc
        i = is(iter);
        j = js(iter);
        
        
        xnew = prox( ( x - sigma * ( 1+tau ) / lambda * ( KTytilde + KT(:,i) * ( y(i) - ytilde(i))/p(i) + a - tau * lambda * xbar) )/(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/lambda , paramf);
        ynew = prox( ( y + sigma * ( 1+tau ) / gamma * ( Kxtilde + K(:,j) * ( x(j) - xtilde(j))/q(j)  - b + tau * gamma * ybar) ) /(1+sigma*(1+tau)*tau) / (1+sigmaadjusted) , sigmaadjusted/(1+sigmaadjusted)/gamma , paramg);
        
        
        
        x = xnew;
        y = ynew;
    end
    xtilde = x;
    ytilde = y;
end
fprintf('\n')

%plot results
subplot(1,3,1)
hold on
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),log10(opt_svrg_acc) -log10(opt_svrg_acc(1)),'m:','linewidth',2);
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),log10(opt_svrg_acc_heuristic) -log10(opt_svrg_acc_heuristic(1)),'m-.','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic' );

subplot(1,3,2)
hold on
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),log10(optxy_svrg_acc) -log10(optxy_svrg_acc(1)),'m:','linewidth',2);
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),log10(optxy_svrg_acc_heuristic) -log10(optxy_svrg_acc_heuristic(1)),'m-.','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic' );

subplot(1,3,3)
hold on
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),test_svrg_acc ,'m:','linewidth',2);
plot((0:maxepoch_svrgacc-1)*(n*d + (n+d) * length_epogh_svrg_acc )/(n*d),test_svrg_acc_heuristic ,'m-.','linewidth',2); hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic' );



%% BATCH PRIMAL TECHNIQUES

% proximal accelerated

fprintf('fb - acc - primal');
LL =  lambda + max(svds(K)).^2 / gamma;
mumu = lambda;
theta =  ( 1 - sqrt(mumu/LL) )/ ( 1 + sqrt(mumu/LL) );


 x = zeros(d,1);
 xt = zeros(d,1);
 
  
maxiter_fb_acc_primal = 1+ceil(total_number_of_passes);

 

for iter = 1:maxiter_fb_acc_primal
    
    
     
         
        % candidate dual       
        y = prox((K * x - b)/gamma, 1/gamma, paramg);
        yy = ( - a - KT * y);
        xx = K * x - b;
        xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
        ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
        
        opt_fb_acc_primal(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
        optxy_fb_acc_primal(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
        switch paramg.type
            case 1
                test_fb_acc_primal(iter) = mean( (Xtest * x - ytest).^2);
            case 6
                pred = Xtest * x;
                test_fb_acc_primal(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
        end
        
     
    
    fullgrad = KT * prox((K * xt - b)/gamma, 1/gamma, paramg);
        
    xnew = prox( xt - (1/LL) * fullgrad, 1/LL, paramf );
      
    xt = xnew + theta * ( xnew - x);       
    x = xnew;
end
fprintf('\n')


% proximal 

fprintf('fb - primal');
LL =  lambda + max(svds(K)).^2 / gamma;
mumu = lambda;


 x = zeros(d,1);
 xt = zeros(d,1);
 
  
maxiter_fb_primal = 1+ceil(total_number_of_passes);

 

for iter = 1:maxiter_fb_primal
    
    
     
         
        % candidate dual       
        y = prox((K * x - b)/gamma, 1/gamma, paramg);
        yy = ( - a - KT * y);
        xx = K * x - b;
        xtemp = prox( yy/lambda , 1/lambda , paramf ); fasttemp = yy' * xtemp - value(xtemp, paramf) - lambda/2*sum(xtemp.^2);
        ytemp = prox( xx/gamma , 1/gamma , paramg ); gasttemp = xx' * ytemp - value(ytemp, paramg) - gamma/2*sum(ytemp.^2);
        
        opt_fb_primal(iter) =  value(x, paramf) + lambda/2 * sum(x.^2) +  gasttemp + a'*x + gamma/2 * sum(y.^2) + value(y, paramg) + fasttemp + b'*y ;
        optxy_fb_primal(iter) = lambda * norm(x-xast)^2 + gamma * norm(y-yast)^2;
        switch paramg.type
            case 1
                test_fb_primal(iter) = mean( (Xtest * x - ytest).^2);
            case 6
                pred = Xtest * x;
                test_fb_primal(iter) = .5  + atest' * pred + .5 * pred' * Atest * pred;
        end
        
     
    
    fullgrad = KT * prox((K * x  - b)/gamma, 1/gamma, paramg);
        
    xnew = prox( x  - (1/LL) * fullgrad, 1/LL, paramf );
      
    x = xnew;
end
fprintf('\n')


%plot results
subplot(1,3,1)
hold on
plot((0:maxiter_fb_acc_primal-1),  log10(opt_fb_acc_primal) -log10(opt_fb_acc_primal(1)),'linestyle','--','linewidth',2,'color',[1 0.6 0]);
hold on;
plot((0:maxiter_fb_primal-1),  log10(opt_fb_primal) -log10(opt_fb_primal(1)),'linestyle','-.','linewidth',2,'color',[1 0.6 0]);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic','fb-primal-acc', 'fb-primal' );



subplot(1,3,2)
hold on
plot((0:maxiter_fb_acc_primal-1),  log10(optxy_fb_acc_primal) -log10(optxy_fb_acc_primal(1)),'linestyle','--','linewidth',2,'color',[1 0.6 0]);
hold on;
plot((0:maxiter_fb_primal-1),  log10(optxy_fb_primal) -log10(optxy_fb_primal(1)),'linestyle','-.','linewidth',2,'color',[1 0.6 0]);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic','fb-primal-acc', 'fb-primal' );


subplot(1,3,3)
hold on
plot((0:maxiter_fb_acc_primal-1),  test_fb_acc_primal ,'linestyle','--','linewidth',2,'color',[1 0.6 0]);
hold on;
plot((0:maxiter_fb_primal-1),  test_fb_primal ,'linestyle','-.','linewidth',2,'color',[1 0.6 0]);
hold off
legend('fb','fb-acc','fb-sto','saga-non-uniform','saga-uniform','svrg','svrg-acc','svrg-acc-heuristic','fb-primal-acc', 'fb-primal' );


ret=1;
return ;

end %end function