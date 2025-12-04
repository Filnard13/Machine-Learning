%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%
%                 Project Work: Machine Learning for Economists           %
%
%  University of Bologna — LM(EC)² Programme
%  Last Updated: 21/10/2025
%
%  Group Members:
%    • Filippo Nardoni
%    • Federica Carrieri
%    • Luca Danelli
%    • Anita Scambia
%
%  Description:
%    Empirical analysis of European stock market predictability using
%    factor-augmented regression models (FARM Predict) and related
%    benchmark approaches (AR, VAR, PCR, AR–PCR, VAR–LASSO).
%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%





%% -------------------------------------------------------------------- %%
%               SECTION 1: Uploading Data and Cleaning data              %
%  --------------------------------------------------------------------  %


% 1.1] Setting the Environment
clc 
clear all
file_path = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data";
fl_pt = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/";

full
%  1.2] Uploading the data
data = readtable(fullfile(file_path, "/EU_stocks_data_daily_2_new.csv"));
col_names = data.Properties.VariableNames;
[T N] = size(data);
data(:, 1) = [];
time_data = readtable(fullfile(file_path, "/dates_daily_new.csv" ));
col_names = col_names(2:N);
data = table2array(data);
N = N-1;


% Define Tensor for MSFE
MSFE_tensor = {};
MSFE_save = NaN(T,N,1:5);



%% -------------------------------------------------------------------- %%
%                   SECTION 2: Preliminary Analysis                      %
%  --------------------------------------------------------------------  %

% 2.1] Analasying Correlation of stocks
C = corr(data, 'Rows', 'complete');
sum = summary(data);


% 2.2] generate block subdivision



% 2.3] Preliminary correlation heatmap


n = size(C, 1);
[xGrid, yGrid] = meshgrid(1:n, 1:n);

% --- Plot correlation surface ---
figure;
surf(xGrid, yGrid, C);
title('Correlation Surface by Country');
xlabel('Variable index i');
ylabel('Variable index j');
zlabel('Correlation coefficient');
colormap(jet);
colorbar;
shading interp;
view(45, 30);
axis tight;

% --- Define country groups by variable index ranges ---
countryNames = {'Spain', 'Italy', 'France', 'Germany', ...
                'Belgium', 'Netherlands', 'Sweden', ...
                'Denmark', 'Norway'};

countryIdx = [ ...
     1   13;   % Spain (13)
    14   25;   % Italy (12)
    26   42;   % France (17)
    43   62;   % Germany (20)
    63   71;   % Belgium (9)
    72   80;   % Netherlands (9)
    81   88;   % Sweden (8)
    89   99;   % Denmark (11)
   100  119;   % Norway (20)
];
% --- Overlay visible grid lines to separate regions ---
hold on;
for i = 1:size(countryIdx,1)
    xEnd = countryIdx(i,2);
    yEnd = countryIdx(i,2);

    % vertical line
    plot3([xEnd xEnd], [1 n], [max(C(:)) max(C(:))], 'k-', 'LineWidth', 1.5);
    % horizontal line
    plot3([1 n], [yEnd yEnd], [max(C(:)) max(C(:))], 'k-', 'LineWidth', 1.5);
end

% --- Optional: Add region labels on the axes ---
ax = gca;
ax.XTick = mean(countryIdx,2);
ax.YTick = mean(countryIdx,2);
ax.XTickLabel = countryNames;
ax.YTickLabel = countryNames;
ax.XTickLabelRotation = 45;
ax.YTickLabelRotation = 0;
ax.FontSize = 10;

hold off;



% Stationary test for all the series
sta_test = nan(N,1);

for i = 1:N
    sta_test(i) = adftest(data(:,i));
end

sta_test = array2table(sta_test, "RowNames", col_names);



%% -------------------------------------------------------------------- %%
%                   SECTION 3: PCA and FACTOR MODEL                      %
%  --------------------------------------------------------------------  %


% 3.1] data standardization
data_std = zscore(data);

surf(corr(data_std)) % after standardization we still observe regions


% 3.2] Proceding with PC
cov_data_std = cov(data_std);
[vec, val]  = eig(cov_data_std);
val = diag(val);

% we take the eigenvalues and their indexes
[evals, rindices] = sort(val,'descend') ;
 
evecs       = vec(:,rindices) ; 
scores      =  data_std * evecs / sqrt(N) ; 
Index = 1:T;
cs_m = mean(data_std,2) ; 

figure;

% === (1) Time Series: 1st PC and Cross-Sectional Average ===
subplot(1,2,1);
plot(scores(:,1), 'LineWidth', 1.4, 'Color', [0.2 0.4 0.8]);
title('1st Principal Component');
xlabel('Time'); ylabel('Score'); grid on;

subplot(1,2,2);
plot(cs_m, 'LineWidth', 1.4, 'Color', [0.8 0.3 0.3]);
title('Cross-Sectional Average');
xlabel('Time'); ylabel('Average'); grid on;

sgtitle('1st PC vs. Cross-Sectional Average');

% === (2) Scree / Elbow Plot ===
figure('Color','w');

plot(1:length(evals), evals(:), '-o', ...
    'LineWidth', 1.5, ...
    'Color', [0.2 0.4 0.8], ...
    'MarkerFaceColor', [0.2 0.4 0.8]);

title('Eigenvalue Decay (Scree Plot)');
xlabel('Principal Component');
ylabel('Eigenvalue');
grid on;
box off;




s = 0;
for i = 1:length(val)
    s = s + val(i);
end

tot_evals = s;
rel_evals = sort(val./tot_evals, "descend");
plot(rel_evals);

% 3.3] ABC Criterion
kmax = 10; % we choose 10 since it is not reasonable to have such a big number of factors
nbck = floor(N/3);
cmax = 5;
graph = 1;
    
% use function ABC_crit 
[rhat1 rhat2] = ABC_crit(data_std, kmax, nbck, cmax, graph);
% to be conservative we choose r_min
n_fac = rhat2;


% 3.4] "asymptotic behavior"
    % plot of eigenvalues against n = 0,...,90 
    figure;
    evals_by_n = nan(90,10);
    ctr = 1;
    for n = 1:4:90
        yn = data_std(:,randperm(90,n));                                          % create a dataset with n randomly chosen columns out of 90
        covyn = yn' * yn/ (T-1);                                           % construct covariance matrix
        en = sort( eig(covyn) , 'descend');                                % obtain eigenvalues (in descending order)
        evals_by_n(1:n,ctr) = en;                                          % store eigenvalues
        ctr = ctr+1;
    end
    nseries = 1:4:90;                 % prepare x-axis for graph
    figure;
    plot(nseries', evals_by_n(1:5,:)'); 
    title('Plot of Eigenvalues against N');
    legend('λ_1', 'λ_2', 'λ_3', 'λ_4', 'λ_5', 'Location', 'best');
    xlabel('N (number of assets)');
    ylabel('Eigenvalue');
    grid on;
    


% Stationary test for all the series
sta_test = nan(N,1);
count_pi = 0;
for i = 1:N
    count_pi = count_pi + 1;
    sta_test(i) = adftest(data_std(:,i));
end

sta_test = array2table(sta_test, "RowNames", col_names);







%% -------------------------------------------------------------------- %%
%                      SECTION 4: COVARIANCE TESTS                       %
%  --------------------------------------------------------------------  %

%______________________ COVARIANCE TEST U_hat ___________________________%

% 4.1] Running and set the factor model
Y = data_std;
F_hat = scores(:,1:n_fac);
L_hat = evecs(:,1:n_fac);
U_hat = Y - F_hat*L_hat';
[T, N] = size(U_hat);

% 4.2] Compute residuals off-diagonal to test see 4.3]


% 4.3] Build Pi_hat matrix using computed residuals
Pi_hat = (U_hat' * U_hat) / T;     
Pi_hat = (Pi_hat + Pi_hat') / 2;   


mask = ~eye(N);
Pi_test = Pi_hat(mask);

% Ensure symmetry
Pi_hat = (Pi_hat + Pi_hat') / 2;



% 4.4] Test statistic for off-diagonal elements
mask = ~eye(N);
Pi_test = Pi_hat(mask);
Pi_0 = zeros(size(Pi_test));
dist_norm = sqrt(T) * (Pi_test - Pi_0);
test_stat = max(abs(dist_norm));

fprintf('\nTest statistic: %.4f\n', test_stat);

% 4.5] Build D_Pi matrix for off-diagonal pairs only
pairs = [];
for i = 1:N
    for j = 1:N
        if i ~= j
            pairs = [pairs; i j];
        end
    end
end
d = size(pairs, 1); 

D_Pi = zeros(T, d);
for k = 1:d
    i = pairs(k,1);
    j = pairs(k,2);
    prod_t = U_hat(:,i) .* U_hat(:,j);   
    D_Pi(:,k) = prod_t - Pi_hat(i,j);
end

% 4.6] Long-run covariance (HAC) with Newey-West
fprintf('\nComputing HAC covariance matrix...\n');
b = floor(4 * (T/100)^(2/9));  
fprintf('Using bandwidth b = %d\n', b);

Omega_Pi = (D_Pi' * D_Pi) / T;
for lag = 1:b
    w = 1 - lag/(b+1);
    D_t = D_Pi(lag+1:T, :);
    D_t_lag = D_Pi(1:T-lag, :);
    M_b = (D_t' * D_t_lag) / T;
    Omega_Pi = Omega_Pi + w * (M_b + M_b');
    if mod(lag, 5) == 0 || lag == b
        fprintf('HAC Progress: %.1f%% (%d of %d lags)\n', lag/b*100, lag, b);
    end
end

% Ensure positive semi-definite
Omega_Pi = (Omega_Pi + Omega_Pi') / 2;
[U_eig, D_eig] = eig(Omega_Pi);
D_eig = diag(D_eig);
D_eig = max(D_eig, 1e-10); 
fprintf('Min eigenvalue: %.2e, Max eigenvalue: %.2e\n', min(D_eig), max(D_eig));
fprintf('Condition number: %.2e\n', max(D_eig)/min(D_eig));

C = U_eig * diag(sqrt(D_eig));

% 4.7] Gaussian multiplier bootstrap
fprintf('\nRunning bootstrap...\n');
bt = 2000;
alpha_lev = 0.05;
S_b = zeros(bt, 1);

for b_iter = 1:bt
    z = randn(d, 1);
    Z = C * z;
    S_b(b_iter) = max(abs(Z));
    if mod(b_iter, 200) == 0
        fprintf('Bootstrap iteration %d / %d\n', b_iter, bt);
    end
end

c_alpha = quantile(S_b, 1 - alpha_lev);

% 4.8] Results
fprintf('\n========================================\n');
fprintf('COVARIANCE TEST RESULTS\n');
fprintf('========================================\n');
fprintf('Test statistic:      %.4f\n', test_stat);
fprintf('Critical value (5%%): %.4f\n', c_alpha);
fprintf('P-value:             %.4f\n', mean(S_b >= test_stat));
if test_stat > c_alpha
    fprintf('Decision:            REJECT H0 (residuals are correlated)\n');
else
    fprintf('Decision:            FAIL TO REJECT H0 (residuals are uncorrelated)\n');
end
fprintf('========================================\n');

% 4.9] Diagnostics
figure;
subplot(2,2,1);
histogram(S_b, 50);
hold on;
xline(test_stat, 'r--', 'LineWidth', 2);
xline(c_alpha, 'g--', 'LineWidth', 2);
xlabel('Bootstrap Statistics');
ylabel('Frequency');
title('Bootstrap Distribution');
legend('Bootstrap', 'Test Stat', 'Critical Value');

subplot(2,2,2);
imagesc(Pi_hat);
colorbar;
title('Estimated \Pi Matrix');
xlabel('Asset j');
ylabel('Asset i');

subplot(2,2,3);
histogram(abs(dist_norm), 50);
xlabel('|√T(π_{ij} - 0)|');
ylabel('Frequency');
title('Distribution of Normalized Deviations');

subplot(2,2,4);
qqplot(S_b);
title('Q-Q Plot: Bootstrap vs Normal');

% 4.10] Additional checks
fprintf('\nDiagnostics:\n');
fprintf('Max |π_ij| (off-diag): %.4f\n', max(abs(Pi_test)));
fprintf('Mean |π_ij| (off-diag): %.4f\n', mean(abs(Pi_test)));
fprintf('Max |√T·π_ij| (off-diag): %.4f\n', max(abs(dist_norm)));
fprintf('Number of pairs tested: %d\n', d);

%%


%______________________ COVARIANCE TEST V_hat ___________________________%

% 4.1] Running and set the factor model
Y = data_std;
F_hat = scores(:,1:n_fac);
L_hat = evecs(:,1:n_fac);
U_hat = Y - F_hat*L_hat';
[T, N] = size(U_hat);


V_i_j = cell(N, 1);
fprintf('Computing LASSO residuals with time-series CV (forward)...\n');

K = 5;
fold_size = floor(T / K);

for i = 1:N
    select_idx = setdiff(1:N, i);
    X = U_hat(:, select_idx);
    y = U_hat(:, i);


    [B_path, Fit_path] = lasso(X, y, 'NumLambda', 100, 'Standardize', false);
    lambdas = Fit_path.Lambda;
    nL = numel(lambdas);


    mse_L = nan(nL, K);
    for k = 2:K
        idx_tr = 1 : (k-1)*fold_size;
        idx_va = (k-1)*fold_size + 1 : min(k*fold_size, T);
        if numel(idx_tr) < size(X,2) || numel(idx_va) < 5, continue; end

        Bk = lasso(X(idx_tr,:), y(idx_tr), 'Lambda', lambdas, 'Standardize', false);
        Yhat = X(idx_va,:) * Bk;                       
        err  = y(idx_va) - Yhat;                        
        mse_L(:,k) = mean(err.^2, 1)';                
    end

    mse_mean = mean(mse_L, 2, 'omitnan');
    [~, best_idx] = min(mse_mean);
    best_lambda = lambdas(best_idx);

    % final fit on all data with chosen lambda
    B_best = lasso(X, y, 'Lambda', best_lambda, 'Standardize', false);
    V_i_j{i} = y - X * B_best;

    if mod(i, 10) == 0
        fprintf('Completed %d / %d variables\n', i, N);
    end
end

% 4.3] Build Pi_hat matrix using computed residuals
Pi_hat = zeros(N, N);
for i = 1:N
    for j = 1:N
        if i == j
            Pi_hat(i,i) = var(V_i_j{i}, 1);  
        else
            Pi_hat(i,j) = mean(V_i_j{i} .* V_i_j{j});
        end
    end
end

% Ensure symmetry
Pi_hat = (Pi_hat + Pi_hat') / 2;

% 4.4] Test statistic for off-diagonal elements
mask = ~eye(N);
Pi_test = Pi_hat(mask);
Pi_0 = zeros(size(Pi_test));
dist_norm = sqrt(T) * (Pi_test - Pi_0);
test_stat = max(abs(dist_norm));

fprintf('\nTest statistic: %.4f\n', test_stat);

% 4.5] Build D_Pi matrix for off-diagonal pairs only
pairs = [];
for i = 1:N
    for j = 1:N
        if i ~= j
            pairs = [pairs; i j];
        end
    end
end
d = size(pairs, 1);  

D_Pi = zeros(T, d);
for k = 1:d
    i = pairs(k,1);
    j = pairs(k,2);
    prod_t = V_i_j{i} .* V_i_j{j};
    D_Pi(:,k) = prod_t - Pi_hat(i,j);
end

% 4.6] Long-run covariance (HAC) with Newey-West
fprintf('\nComputing HAC covariance matrix...\n');
b = floor(4 * (T/100)^(2/9));  
fprintf('Using bandwidth b = %d\n', b);

Omega_Pi = (D_Pi' * D_Pi) / T;
for lag = 1:b
    w = 1 - lag/(b+1);
    D_t = D_Pi(lag+1:T, :);
    D_t_lag = D_Pi(1:T-lag, :);
    M_b = (D_t' * D_t_lag) / T;
    Omega_Pi = Omega_Pi + w * (M_b + M_b');
    if mod(lag, 5) == 0 || lag == b
        fprintf('HAC Progress: %.1f%% (%d of %d lags)\n', lag/b*100, lag, b);
    end
end

% Ensure positive semi-definite
Omega_Pi = (Omega_Pi + Omega_Pi') / 2;
[U_eig, D_eig] = eig(Omega_Pi);
D_eig = diag(D_eig);
D_eig = max(D_eig, 1e-10); 
fprintf('Min eigenvalue: %.2e, Max eigenvalue: %.2e\n', min(D_eig), max(D_eig));
fprintf('Condition number: %.2e\n', max(D_eig)/min(D_eig));

C = U_eig * diag(sqrt(D_eig));

% 4.7] Gaussian multiplier bootstrap
fprintf('\nRunning bootstrap...\n');
bt = 2000;
alpha_lev = 0.05;
S_b = zeros(bt, 1);

for b_iter = 1:bt
    z = randn(d, 1);
    Z = C * z;
    S_b(b_iter) = max(abs(Z));
    if mod(b_iter, 200) == 0
        fprintf('Bootstrap iteration %d / %d\n', b_iter, bt);
    end
end

c_alpha = quantile(S_b, 1 - alpha_lev);

% 4.8] Results
fprintf('\n========================================\n');
fprintf('COVARIANCE TEST RESULTS\n');
fprintf('========================================\n');
fprintf('Test statistic:      %.4f\n', test_stat);
fprintf('Critical value (5%%): %.4f\n', c_alpha);
fprintf('P-value:             %.4f\n', mean(S_b >= test_stat));
if test_stat > c_alpha
    fprintf('Decision:            REJECT H0 (residuals are correlated)\n');
else
    fprintf('Decision:            FAIL TO REJECT H0 (residuals are uncorrelated)\n');
end
fprintf('========================================\n');

% 4.9] Diagnostics
figure;
subplot(2,2,1);
histogram(S_b, 50);
hold on;
xline(test_stat, 'r--', 'LineWidth', 2);
xline(c_alpha, 'g--', 'LineWidth', 2);
xlabel('Bootstrap Statistics');
ylabel('Frequency');
title('Bootstrap Distribution');
legend('Bootstrap', 'Test Stat', 'Critical Value');

subplot(2,2,2);
imagesc(Pi_hat);
colorbar;
title('Estimated \Pi Matrix');
xlabel('Asset j');
ylabel('Asset i');

subplot(2,2,3);
histogram(abs(dist_norm), 50);
xlabel('|√T(π_{ij} - 0)|');
ylabel('Frequency');
title('Distribution of Normalized Deviations');

subplot(2,2,4);
qqplot(S_b);
title('Q-Q Plot: Bootstrap vs Normal');

% 4.10] Additional checks
fprintf('\nDiagnostics:\n');
fprintf('Max |π_ij| (off-diag): %.4f\n', max(abs(Pi_test)));
fprintf('Mean |π_ij| (off-diag): %.4f\n', mean(abs(Pi_test)));
fprintf('Max |√T·π_ij| (off-diag): %.4f\n', max(abs(dist_norm)));
fprintf('Number of pairs tested: %d\n', d);

















%% -------------------------------------------------------------------- %%
%                         SECTION 5: AR MODEL                            %
%  --------------------------------------------------------------------  %


Y_ar = data_std; % standardized data: T x N
[T_ar, N_ar] = size(Y_ar);
rol_window_ar = [150, 252, 500];
lags_ar = 4:1:10;
h_ar = 1;
MSFE_ar = NaN(length(rol_window_ar), length(lags_ar));
Forecast_y = NaN(T_ar, N_ar, 5);


% Initialize
AIC_count = zeros(1, length(lags_ar));
BIC_count = zeros(1, length(lags_ar));
bestLag_AIC = NaN(N_ar, 1);
bestLag_BIC = NaN(N_ar, 1);

% Loop over all series
for n = 1:N_ar
    y_ar = Y_ar(:, n);
    aic_vals = NaN(length(lags_ar),1);
    bic_vals = NaN(length(lags_ar),1);

    % Compute AIC/BIC for each lag
    for l = 1:length(lags_ar)
        L_ar = lags_ar(l);
        [~, ~, AIC_val, BIC_val] = ar_ols(y_ar, L_ar);
        aic_vals(l) = AIC_val;
        bic_vals(l) = BIC_val;
    end

    % Record best lag per criterion
    [~, idxA] = min(aic_vals);
    [~, idxB] = min(bic_vals);
    bestLag_AIC(n) = idxA;
    bestLag_BIC(n) = idxB;
end

% Count share of models selecting each lag
for l = 1:length(lags_ar)
    AIC_count(l) = mean(bestLag_AIC == l);
    BIC_count(l) = mean(bestLag_BIC == l);
end

% === Display as tables ===
VarNames_ar = strcat("p=", string(lags_ar));
AIC_share_table = array2table(AIC_count, 'VariableNames', VarNames_ar);
BIC_share_table = array2table(BIC_count, 'VariableNames', VarNames_ar);

disp('=== Share of models selecting each lag (AIC) ===');
disp(AIC_share_table);
disp('=== Share of models selecting each lag (BIC) ===');
disp(BIC_share_table);

% === Visualization: Compact bar charts (blue & yellow) ===
figure('Position',[200 200 700 300]);

% --- AIC (blue) ---
subplot(1,2,1);
bar(lags_ar, AIC_count*100, 'FaceColor',[0 0.4470 0.7410]); % blue
title('AIC-selected lag share');
xlabel('Lag order p');
ylabel('Percentage (%)');
ylim([0 100]);
xticks(lags_ar);
grid on; box off;
for i = 1:length(lags_ar)
    text(lags_ar(i), AIC_count(i)*100 + 2, ...
        sprintf('%.0f%%', AIC_count(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

% --- BIC (yellow) ---
subplot(1,2,2);
bar(lags_ar, BIC_count*100, 'FaceColor',[0.9290 0.6940 0.1250]); % yellow
title('BIC-selected lag share');
xlabel('Lag order p');
ylabel('');
ylim([0 100]);
xticks(lags_ar);
grid on; box off;
for i = 1:length(lags_ar)
    text(lags_ar(i), BIC_count(i)*100 + 2, ...
        sprintf('%.0f%%', BIC_count(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

sgtitle('Percentage of Models Selecting Each Lag AR Model(Full Sample)');

%_________________________ MODEL AR TUNED _______________________________ %
[min_MSFE_ar, idx_min_ar] = min(MSFE_ar(:));
[row_idx_ar, col_idx_ar] = ind2sub(size(MSFE_ar), idx_min_ar);
lag_ar = lags_ar(col_idx_ar);
rol_ar = rol_window_ar(row_idx_ar);

% New
MSFE_grouped = NaN(T_ar, N_ar);

fprintf('\nOptimal lag (p*) = %d | Optimal window (w*) = %d | MSFE = %.6f\n', ...
        lag_ar, rol_ar, min_MSFE_ar);

% Define country index mapping
countryIdx = [ ...
    1   11;  % Spain
    12  22;  % Italy
    23  44;  % France
    45  66;  % Germany
    67  76;  % Belgium
    77  86;  % Netherlands
    87  96;  % Sweden
    97  106; % Denmark
    107 119  % Norway
];

% --- Define grouped blocks ---
groupIdx = { [1 2], [3 4], [6 7] };   % (Spain,Italy), (France,Germany), (Netherlands,Sweden)
groupNames = ["South_Europe","Core_Europe","North_Europe"];
N_groups = numel(groupIdx);




Mean_group_MSFE = NaN(T_ar, N_groups);

% Compute AR tuned model errors
MSE_tot_ar_final = 0;
for n = 1:N_ar
    MSE_n_ar_final = 0;
    count_ar_final = 0;
    for t = 1:(T_ar - rol_ar - h_ar + 1)
        y_ar = Y_ar(t:(t + rol_ar - 1), n);
        [phi_ar, ~, ~, ~] = ar_ols(y_ar, lag_ar);
        yF_ar = y_ar(end - lag_ar + 1:end)' * phi_ar;
        Forecast_y(t,n,1) = yF_ar;
        y_true_ar = Y_ar(t + rol_ar, n);
        err2_ar = (yF_ar - y_true_ar)^2;

        MSFE_save(t, n, 1) = err2_ar;
        MSFE_grouped(t, n) = err2_ar;

        MSE_n_ar_final = MSE_n_ar_final + err2_ar;
        count_ar_final = count_ar_final + 1;
    end
    MSE_n_ar_final = MSE_n_ar_final / count_ar_final;
    MSE_tot_ar_final = MSE_tot_ar_final + MSE_n_ar_final;
end

% Compute mean MSFE per group at each time t
for t = 1:(T_ar - rol_ar - h_ar + 1)
    for g = 1:N_groups
        idx_blocks = [];
        for ci = groupIdx{g}
            idx_blocks = [idx_blocks, countryIdx(ci,1):countryIdx(ci,2)];
        end
        valid_errors = MSFE_grouped(t,idx_blocks); 
        Mean_group_MSFE(t, g) = mean(valid_errors);
    end
end

% Compute mean AIC/BIC across rolling windows
AIC_mean_ar = mean(AIC_models_ar, 1, 'omitnan');
BIC_mean_ar = mean(BIC_models_ar, 1, 'omitnan');




% Find optimal (minimum) AIC/BIC
[min_AIC_ar, idx_AIC_ar] = min(AIC_mean_ar);
[min_BIC_ar, idx_BIC_ar] = min(BIC_mean_ar);

% Plot AIC and BIC vs lag order
figure;
plot(lags_ar, AIC_mean_ar, '-o', 'LineWidth', 2, 'DisplayName', 'AIC');
hold on;
plot(lags_ar, BIC_mean_ar, '-s', 'LineWidth', 2, 'DisplayName', 'BIC');

% Highlight optimal points
plot(lags_ar(idx_AIC_ar), min_AIC_ar, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
plot(lags_ar(idx_BIC_ar), min_BIC_ar, 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');

xlabel('Lag order (p)');
ylabel('Information Criterion');
title('AR Model: AIC and BIC vs Lag Order');
legend('Location', 'best');
grid on;

fprintf('\nOptimal AIC at p = %d | Value = %.4f\n', lags_ar(idx_AIC_ar), min_AIC_ar);
fprintf('Optimal BIC at p = %d | Value = %.4f\n', lags_ar(idx_BIC_ar), min_BIC_ar);





% Create table for group-level MSFE
MeanGroupMSFE_table = array2table(Mean_group_MSFE, 'VariableNames', groupNames);

% Save results
savepath = fullfile(fl_pt, 'Tables');  
writetable(MeanGroupMSFE_table, fullfile(savepath, 'MeanGroup_MSFE_AR.txt'), 'Delimiter', '\t');

fprintf('\nMean Group-level MSFE saved in:\n%s/MeanGroup_MSFE_AR.txt\n', savepath);


% Prepare formatted tables
RowNames_ar = strcat("w=", string(rol_window_ar));
VarNames_ar = strcat("p=", string(lags_ar));

MSFE_table_ar = array2table(MSFE_ar, 'RowNames', RowNames_ar, 'VariableNames', VarNames_ar);
AIC_table_ar  = array2table(AIC_models_ar, 'RowNames', RowNames_ar, 'VariableNames', VarNames_ar);
BIC_table_ar  = array2table(BIC_models_ar, 'RowNames', RowNames_ar, 'VariableNames', VarNames_ar);

% Display results in MATLAB
disp('=== Mean Squared Forecast Error (MSFE) ===');
disp(MSFE_table_ar);

% Save as .txt (plain LaTeX-friendly format)
writetable(MSFE_table_ar, fullfile(savepath, 'MSFE_table_AR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(AIC_table_ar,  fullfile(savepath, 'AIC_table_AR.txt'),  'WriteRowNames', true, 'Delimiter', '\t');
writetable(BIC_table_ar,  fullfile(savepath, 'BIC_table_AR.txt'),  'WriteRowNames', true, 'Delimiter', '\t');

fprintf('\nTables successfully saved in:\n%s\n', savepath);
fprintf('  • MSFE_table_AR.txt\n  • AIC_table_AR.txt\n  • BIC_table_AR.txt\n');



%% -------------------------------------------------------------------- %%
%                           SECTION 6: VAR                               %
%  --------------------------------------------------------------------  %

Y_var = data_std;
[T_var, N_var] = size(Y_var);
lags_var = 1:3;
rol_window_var = 500;
h_var = 1;


% === AIC/BIC computation for each lag and series (VAR) ===
AIC_store_var = NaN(length(lags_var), N_var);
BIC_store_var = NaN(length(lags_var), N_var);
MSFE_var = NaN(length(rol_window_var), length(lags_var));

for l = 1:length(lags_var)
    p_var = lags_var(l);
    T_eff_var = T_var - p_var;

    % Build regressor matrix
    X_var = [];
    for lag = 1:p_var
        X_var = [X_var, Y_var(p_var-lag+1:T_var-lag, :)];
    end
    X_var = [ones(T_eff_var,1), X_var];
    Y_dep_var = Y_var(p_var+1:end, :);

    % Estimate for each variable
    for n = 1:N_var
        y_var = Y_dep_var(:, n);
        beta_var = (X_var' * X_var) \ (X_var' * y_var);
        y_hat_var = X_var * beta_var;
        res_var = y_var - y_hat_var;
        sigma2_var = (res_var' * res_var) / T_eff_var;
        k_var = 1 + N_var*p_var;
        logL_var = -T_eff_var/2 * (log(2*pi) + log(sigma2_var) + 1);
        AIC_store_var(l,n) = -2*logL_var + 2*k_var;
        BIC_store_var(l,n) = -2*logL_var + k_var*log(T_eff_var);
    end
    fprintf('Completed AIC/BIC: lag=%d\n', p_var);
end

% === Identify best lag for each series (Stock & Watson approach) ===
[~, bestLag_AIC] = min(AIC_store_var, [], 1);
[~, bestLag_BIC] = min(BIC_store_var, [], 1);

% Compute share of models selecting each lag
AIC_count_var = zeros(1, length(lags_var));
BIC_count_var = zeros(1, length(lags_var));
for l = 1:length(lags_var)
    AIC_count_var(l) = mean(bestLag_AIC == l);
    BIC_count_var(l) = mean(bestLag_BIC == l);
end

% === Display results ===
VarNames_var = strcat("p=", string(lags_var));
AIC_share_table = array2table(AIC_count_var, 'VariableNames', VarNames_var);
BIC_share_table = array2table(BIC_count_var, 'VariableNames', VarNames_var);

disp('=== Share of models selecting each lag (AIC) ===');
disp(AIC_share_table);
disp('=== Share of models selecting each lag (BIC) ===');
disp(BIC_share_table);

% === Visualization: Compact blue & yellow bar charts ===
figure('Position',[200 200 700 300]);

% --- AIC (blue) ---
subplot(1,2,1);
bar(lags_var, AIC_count_var*100, 'FaceColor',[0 0.4470 0.7410]);
title('AIC-selected lag share');
xlabel('Lag order p');
ylabel('Percentage (%)');
ylim([0 100]); xticks(lags_var); grid on; box off;
for i = 1:length(lags_var)
    text(lags_var(i), AIC_count_var(i)*100 + 2, ...
        sprintf('%.0f%%', AIC_count_var(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

% --- BIC (yellow) ---
subplot(1,2,2);
bar(lags_var, BIC_count_var*100, 'FaceColor',[0.9290 0.6940 0.1250]);
title('BIC-selected lag share');
xlabel('Lag order p');
ylabel('');
ylim([0 100]); xticks(lags_var); grid on; box off;
for i = 1:length(lags_var)
    text(lags_var(i), BIC_count_var(i)*100 + 2, ...
        sprintf('%.0f%%', BIC_count_var(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

sgtitle('Percentage of Models Selecting Each Lag VAR Model (Full Sample)');

for r = 1:length(rol_window_var)
    w_var = rol_window_var(r);
    for l = 1:length(lags_var)
        p_var = lags_var(l);
        MSE_tot_var = 0;
        for n = 1:N_var
            MSE_n_var = 0;
            for t = 1:(T_var - w_var - h_var + 1)
                Y_train_var = Y_var(t:(t+w_var-1), :);
                Y_dep_var = Y_train_var(p_var+1:end, n);
                X_var = [];
                for lag = 1:p_var
                    X_var = [X_var, Y_train_var(p_var+1-lag:end-lag, :)];
                end
                X_var = [ones(size(X_var,1),1), X_var];
                beta_var = (X_var' * X_var) \ (X_var' * Y_dep_var);
                x_last_var = [1];
                for lag = 1:p_var
                    x_last_var = [x_last_var, Y_train_var(end-lag+1, :)];
                end
                yF_var = x_last_var * beta_var;
                y_true_var = Y_var(t+w_var, n);
                MSE_n_var = MSE_n_var + (yF_var - y_true_var)^2;
            end
            MSE_tot_var = MSE_tot_var + MSE_n_var / (T_var - w_var - h_var + 1);
        end
        MSFE_var(r,l) = MSE_tot_var / N_var;
        fprintf('Rolling window=%d | Lag=%d | MSFE=%.6f\n', w_var, p_var, MSFE_var(r,l));
    end
end

RowNames_var = strcat("w=", string(rol_window_var));
VarNames_var = strcat("p=", string(lags_var));
disp('=== Mean Squared Forecast Error (MSFE) ===');
MSFE_table_var = array2table(MSFE_var, 'RowNames', RowNames_var, 'VariableNames', VarNames_var);
disp(MSFE_table_var);

figure;
heatmap(string(lags_var), string(rol_window_var), MSFE_var);
xlabel('Lag order (p)'); ylabel('Rolling window (w)');
title('VAR(OLS): 1-step ahead MSFE');
colorbar;




%________________________ MODEL VAR TUNED _______________________________ %
[min_MSFE_var, idx_min_var] = min(MSFE_var(:));
[row_idx_var, col_idx_var] = ind2sub(size(MSFE_var), idx_min_var);
lag_var = lags_var(col_idx_var);
rol_var = rol_window_var(row_idx_var);
MSFE_grouped_var = NaN(T_ar,N_ar);
fprintf('\nOptimal VAR lag (p*) = %d | Optimal window (w*) = %d | MSFE = %.6f\n', ...
        lag_var, rol_var, min_MSFE_var);
countryIdx = [ ...
    1   11;  
    12  22;  
    23  44;  
    45  66;  
    67  76;  
    77  86;  
    87  96;  
    97  106; 
    107 119  
];
groupIdx = { [1 2], [3 4], [6 7] };
groupNames = ["South_Europe","Core_Europe","North_Europe"];
N_groups = numel(groupIdx);
Mean_group_MSFE_var = NaN(T_var, N_groups);
MSE_tot_var_final = 0;
for n = 1:N_var
    MSE_n_var_final = 0;
    for t = 1:(T_var - rol_var - h_var + 1)
        Y_train_var = Y_var(t:(t + rol_var - 1), :);
        p_var = lag_var;
        X_var = [];
        for lag = 1:p_var
            X_var = [X_var, Y_train_var(p_var+1-lag:end-lag, :)];
        end
        X_var = [ones(size(X_var,1),1), X_var];
        Y_dep_var = Y_train_var(p_var+1:end, n);
        beta_var = (X_var' * X_var) \ (X_var' * Y_dep_var);
        x_last_var = [1];
        for lag = 1:p_var
            x_last_var = [x_last_var, Y_train_var(end-lag+1, :)];
        end
        yF_var = x_last_var * beta_var;
        Forecast_y(t,n,2) = yF_var;
        y_true_var = Y_var(t + rol_var, n);
        err2_var = (yF_var - y_true_var)^2;
        MSFE_save(t, n, 2) = err2_var;
        MSFE_grouped_var(t, n) = err2_var;
        MSE_n_var_final = MSE_n_var_final + err2_var;
    end
    MSE_n_var_final = MSE_n_var_final / (T_var - rol_var - h_var + 1);
    MSE_tot_var_final = MSE_tot_var_final + MSE_n_var_final;
end
for t = 1:(T_var - rol_var - h_var + 1)
    for g = 1:N_groups
        idx_blocks = [];
        for ci = groupIdx{g}
            idx_blocks = [idx_blocks, countryIdx(ci,1):countryIdx(ci,2)];
        end
        valid_errors = MSFE_grouped_var(t, idx_blocks);
        Mean_group_MSFE_var(t, g) = mean(valid_errors, 'omitnan');
    end
end
MeanGroupMSFE_VAR_table = array2table(Mean_group_MSFE_var, 'VariableNames', groupNames);
savepath = fullfile(fl_pt, 'Tables');  
RowNames_var = strcat("w=", string(rol_window_var));
VarNames_var = strcat("p=", string(lags_var));
MSFE_table_var = array2table(MSFE_var, 'RowNames', RowNames_var, 'VariableNames', VarNames_var);
AIC_table_var = array2table(AIC_mean_var, 'VariableNames', "AIC", 'RowNames', string(lags_var));
BIC_table_var = array2table(BIC_mean_var, 'VariableNames', "BIC", 'RowNames', string(lags_var));
disp('=== Mean Squared Forecast Error (MSFE) — VAR ===');
disp(MSFE_table_var);
writetable(MSFE_table_var, fullfile(savepath, 'MSFE_table_VAR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(AIC_table_var,  fullfile(savepath, 'AIC_table_VAR.txt'),  'WriteRowNames', true, 'Delimiter', '\t');
writetable(BIC_table_var,  fullfile(savepath, 'BIC_table_VAR.txt'),  'WriteRowNames', true, 'Delimiter', '\t');
fprintf('\nTables successfully saved in:\n%s\n', savepath);
fprintf('  • MSFE_table_VAR.txt\n  • AIC_table_VAR.txt\n  • BIC_table_VAR.txt\n');







%________________________ MODEL VAR LASSO TUNED _________________________ %

fprintf('\n=== VAR-LASSO using tuned VAR options: p*=%d | w*=%d ===\n', lag_var, rol_var);

Y_var_lasso = Y_var;
[T_var_lasso, N_var_lasso] = size(Y_var_lasso);
h_var_lasso = h_var;
p_var_lasso = lag_var;
rol_opt_var_lasso = rol_var;

MSFE_var_lasso = NaN(length(rol_opt_var_lasso), length(p_var_lasso));
MSFE_save_var_lasso = NaN(T_var_lasso, N_var_lasso);
Forecast_y_var_lasso = NaN(T_var_lasso, N_var_lasso);

countryIdx_var_lasso = [ ...
    1   11;
    12  22;
    23  44;
    45  66;
    67  76;
    77  86;
    87  96;
    97  106;
    107 119
];

groupIdx_var_lasso = { [1 2], [3 4], [6 7] };
groupNames_var_lasso = ["South_Europe","Core_Europe","North_Europe"];
N_groups_var_lasso = numel(groupIdx_var_lasso);

MSFE_grouped_var_lasso = NaN(T_var_lasso, N_var_lasso);
Mean_group_MSFE_var_lasso = NaN(T_var_lasso, N_groups_var_lasso);

MSE_total_var_lasso = 0;
fprintf('Starting VAR-LASSO estimation for %d variables...\n', N_var_lasso);

for n = 1:N_var_lasso
    fprintf('Processing variable %d/%d...\n', n, N_var_lasso);
    MSE_n_var_lasso = 0;
    T_forecast = T_var_lasso - rol_opt_var_lasso - h_var_lasso + 1;

    for t = 1:T_forecast
        if mod(t, 500) == 0
            fprintf('  Variable %d: Forecast %d/%d (%.1f%%)\n', n, t, T_forecast, 100*t/T_forecast);
        end

        Y_train_var_lasso = Y_var_lasso(t:(t + rol_opt_var_lasso - 1), :);
        X_var_lasso = [];
        for lag = 1:p_var_lasso
            X_var_lasso = [X_var_lasso, Y_train_var_lasso(p_var_lasso+1-lag:end-lag, :)];
        end
        Y_dep_var_lasso = Y_train_var_lasso(p_var_lasso+1:end, n);

        [B, FitInfo] = lasso(X_var_lasso, Y_dep_var_lasso, 'CV', 5);
        idxLambda = FitInfo.IndexMinMSE;

        x_last_var_lasso = [];
        for lag = 1:p_var_lasso
            x_last_var_lasso = [x_last_var_lasso, Y_train_var_lasso(end-lag+1, :)];
        end
        yF_var_lasso = FitInfo.Intercept(idxLambda) + x_last_var_lasso * B(:, idxLambda);

        Forecast_y_var_lasso(t,n) = yF_var_lasso;
        y_true_var_lasso = Y_var_lasso(t + rol_opt_var_lasso, n);
        err2_var_lasso = (yF_var_lasso - y_true_var_lasso)^2;

        MSFE_save_var_lasso(t, n) = err2_var_lasso;
        MSFE_grouped_var_lasso(t, n) = err2_var_lasso;
        MSE_n_var_lasso = MSE_n_var_lasso + err2_var_lasso;
    end

    MSE_n_var_lasso = MSE_n_var_lasso / T_forecast;
    MSE_total_var_lasso = MSE_total_var_lasso + MSE_n_var_lasso;
end

MSE_total_var_lasso = MSE_total_var_lasso / N_var_lasso;
fprintf('Final tuned VAR–LASSO MSFE = %.6f\n', MSE_total_var_lasso);

fprintf('\nComputing group-level MSFE...\n');
for t = 1:(T_var_lasso - rol_opt_var_lasso - h_var_lasso + 1)
    for g = 1:N_groups_var_lasso
        idx_blocks_var_lasso = [];
        for ci = groupIdx_var_lasso{g}
            idx_blocks_var_lasso = [idx_blocks_var_lasso, ...
                countryIdx_var_lasso(ci,1):countryIdx_var_lasso(ci,2)];
        end
        valid_errors_var_lasso = MSFE_grouped_var_lasso(t, idx_blocks_var_lasso);
        Mean_group_MSFE_var_lasso(t, g) = mean(valid_errors_var_lasso, 'omitnan');
    end
end

MeanGroupMSFE_VAR_LASSO_table = array2table(Mean_group_MSFE_var_lasso, ...
    'VariableNames', groupNames_var_lasso);

RowNames_var_lasso = strcat("w=", string(rol_opt_var_lasso));
VarNames_var_lasso = strcat("p=", string(p_var_lasso));

MSFE_table_VAR_LASSO = array2table(MSFE_var_lasso, 'RowNames', RowNames_var_lasso, 'VariableNames', VarNames_var_lasso);

AIC_models_VAR_LASSO = NaN(length(p_var_lasso),1);
BIC_models_VAR_LASSO = NaN(length(p_var_lasso),1);
AIC_table_VAR_LASSO  = array2table(AIC_models_VAR_LASSO, 'VariableNames', "AIC", 'RowNames', string(p_var_lasso));
BIC_table_VAR_LASSO  = array2table(BIC_models_VAR_LASSO, 'VariableNames', "BIC", 'RowNames', string(p_var_lasso));

savepath_var_lasso = '/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Tables';

writetable(MeanGroupMSFE_VAR_LASSO_table, fullfile(savepath_var_lasso, 'MeanGroup_MSFE_VAR_LASSO.txt'), 'Delimiter', '\t');
writetable(MSFE_table_VAR_LASSO, fullfile(savepath_var_lasso, 'MSFE_table_VAR_LASSO.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(AIC_table_VAR_LASSO, fullfile(savepath_var_lasso, 'AIC_table_VAR_LASSO.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(BIC_table_VAR_LASSO, fullfile(savepath_var_lasso, 'BIC_table_VAR_LASSO.txt'), 'WriteRowNames', true, 'Delimiter', '\t');

Forecast_table_VAR_LASSO = array2table(Forecast_y_var_lasso);
writetable(Forecast_table_VAR_LASSO, fullfile(savepath_var_lasso, 'Forecast_y_VAR_LASSO.txt'), 'Delimiter', '\t');

fprintf('\nTables successfully saved in:\n%s\n', savepath_var_lasso);
fprintf('  • MeanGroup_MSFE_VAR_LASSO.txt\n  • MSFE_table_VAR_LASSO.txt\n  • AIC_table_VAR_LASSO.txt\n  • BIC_table_VAR_LASSO.txt\n  • Forecast_y_VAR_LASSO.txt\n');


%% -------------------------------------------------------------------- %%
%          SECTION 7: Principal Component Regression                     %
%  --------------------------------------------------------------------  %
Y_pcr = data_std;
[T_pcr, N_pcr] = size(Y_pcr);
F_hat_pcr = scores(:, 1:n_fac);
r_pcr = size(F_hat_pcr, 2);
rol_window_pcr = [150, 252, 500];
lags_pcr = 4:10;
h_pcr = 1;


MSFE_pcr = NaN(length(rol_window_pcr), length(lags_pcr));
AIC_store_pcr = NaN(length(lags_pcr), N_pcr);
BIC_store_pcr = NaN(length(lags_pcr), N_pcr);

for li = 1:length(lags_pcr)
    q_pcr = lags_pcr(li);
    
    for n = 1:N_pcr
        X_pcr = [];
        for j = 1:q_pcr
            X_pcr = [X_pcr, F_hat_pcr(q_pcr+1-j:T_pcr-j, :)];
        end
        X_pcr = [ones(size(X_pcr, 1), 1), X_pcr];
        y_dep_pcr = Y_pcr(q_pcr+1:T_pcr, n);

        ok_pcr = all(isfinite([y_dep_pcr, X_pcr]), 2);
        y__pcr = y_dep_pcr(ok_pcr);
        X__pcr = X_pcr(ok_pcr, :);
        Te_n_pcr = size(X__pcr, 1);
        if Te_n_pcr < (1 + r_pcr*q_pcr + 1)
            continue;
        end

        beta_pcr = X__pcr \ y__pcr;
        res_pcr = y__pcr - X__pcr * beta_pcr;
        sigma2_pcr = (res_pcr' * res_pcr) / Te_n_pcr;
        
        k_pcr = 1 + r_pcr*q_pcr;
        logL_pcr = -Te_n_pcr/2 * (log(2*pi) + log(sigma2_pcr) + 1);
        AIC_store_pcr(li, n) = -2*logL_pcr + 2*k_pcr;
        BIC_store_pcr(li, n) = -2*logL_pcr + k_pcr*log(Te_n_pcr);
    end
    
    fprintf('Computed AIC/BIC for q=%d\n', q_pcr);
end

% === Identify best lag per series (Stock & Watson method) ===
[~, bestLag_AIC_pcr] = min(AIC_store_pcr, [], 1);
[~, bestLag_BIC_pcr] = min(BIC_store_pcr, [], 1);

% === Compute share of models selecting each lag ===
AIC_share_pcr = zeros(1, length(lags_pcr));
BIC_share_pcr = zeros(1, length(lags_pcr));
for li = 1:length(lags_pcr)
    AIC_share_pcr(li) = mean(bestLag_AIC_pcr == li);
    BIC_share_pcr(li) = mean(bestLag_BIC_pcr == li);
end

% === Display shares as tables ===
VarNames_pcr = strcat("q=", string(lags_pcr));
AIC_share_table_pcr = array2table(AIC_share_pcr, 'VariableNames', VarNames_pcr);
BIC_share_table_pcr = array2table(BIC_share_pcr, 'VariableNames', VarNames_pcr);

disp('=== Share of models selecting each lag (AIC) ===');
disp(AIC_share_table_pcr);
disp('=== Share of models selecting each lag (BIC) ===');
disp(BIC_share_table_pcr);

% === Visualization: Compact bar charts (blue & yellow) ===
figure('Position',[200 200 700 300]);

% --- AIC (blue) ---
subplot(1,2,1);
bar(lags_pcr, AIC_share_pcr*100, 'FaceColor',[0 0.4470 0.7410]);
title('AIC-selected lag share');
xlabel('Lag order q');
ylabel('Percentage (%)');
ylim([0 100]); xticks(lags_pcr); grid on; box off;
for i = 1:length(lags_pcr)
    text(lags_pcr(i), AIC_share_pcr(i)*100 + 2, ...
        sprintf('%.0f%%', AIC_share_pcr(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

% --- BIC (yellow) ---
subplot(1,2,2);
bar(lags_pcr, BIC_share_pcr*100, 'FaceColor',[0.9290 0.6940 0.1250]);
title('BIC-selected lag share');
xlabel('Lag order q');
ylabel('');
ylim([0 100]); xticks(lags_pcr); grid on; box off;
for i = 1:length(lags_pcr)
    text(lags_pcr(i), BIC_share_pcr(i)*100 + 2, ...
        sprintf('%.0f%%', BIC_share_pcr(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

sgtitle('Percentage of Models Selecting Each Lag PCR Model (Full Sample)');

for rwin = 1:length(rol_window_pcr)
    w_pcr = rol_window_pcr(rwin);
    
    for li = 1:length(lags_pcr)
        q_pcr = lags_pcr(li);
        MSE_tot_pcr = 0;
        
        for n = 1:N_pcr
            MSE_n_pcr = 0;
            
            for t = 1:(T_pcr - w_pcr - h_pcr + 1)
                Y_train_pcr = Y_pcr(t:(t + w_pcr - 1), :);
                F_train_pcr = F_hat_pcr(t:(t + w_pcr - 1), :);
                
                X_pcr = [];
                for j = 1:q_pcr
                    X_pcr = [X_pcr, F_train_pcr(q_pcr+1-j:end-j, :)];
                end
                X_pcr = [ones(size(X_pcr, 1), 1), X_pcr];
                
                y_dep_pcr = Y_train_pcr(q_pcr+1:end, n);
                
                ok_pcr = all(isfinite([y_dep_pcr, X_pcr]), 2);
                y__pcr = y_dep_pcr(ok_pcr);
                X__pcr = X_pcr(ok_pcr, :);
                
                if size(X__pcr, 1) < (1 + r_pcr*q_pcr + 1)
                    continue;
                end
                
                beta_pcr = X__pcr \ y__pcr;
                
                x_f_pcr = [1];
                for j = 1:q_pcr
                    x_f_pcr = [x_f_pcr, F_train_pcr(end-j+1, :)];
                end
                
                yF_pcr = x_f_pcr * beta_pcr;
                y_true_pcr = Y_pcr(t + w_pcr, n);
                MSE_n_pcr = MSE_n_pcr + (yF_pcr - y_true_pcr)^2;
            end
            
            MSE_n_pcr = MSE_n_pcr / (T_pcr - w_pcr - h_pcr + 1);
            MSE_tot_pcr = MSE_tot_pcr + MSE_n_pcr;
        end
        
        MSFE_pcr(rwin, li) = MSE_tot_pcr / N_pcr;
        fprintf('Lag: %d | Window: %d | MSFE: %.6f\n', q_pcr, w_pcr, MSFE_pcr(rwin, li));
    end
end

RowNames_pcr = strcat("w=", string(rol_window_pcr));
VarNames_pcr = strcat("q=", string(lags_pcr));

disp('=== Mean Squared Forecast Error (MSFE) ===');
MSFE_table_pcr = array2table(MSFE_pcr, 'RowNames', RowNames_pcr, 'VariableNames', VarNames_pcr);
disp(MSFE_table_pcr);

disp('=== AIC and BIC (full data) ===');
IC_table_pcr = array2table([lags_pcr(:), AIC_models_pcr(:), BIC_models_pcr(:)], 'VariableNames', {'Lag', 'AIC', 'BIC'});
disp(IC_table_pcr);

figure;
heatmap(string(lags_pcr), string(rol_window_pcr), MSFE_pcr);
xlabel('Lag order q');
ylabel('Rolling window w');
title('MSFE for PCR (1-step ahead)');
colorbar;

figure;
plot(lags_pcr, AIC_models_pcr, '-o', 'LineWidth', 2, 'DisplayName', 'AIC');
hold on;
plot(lags_pcr, BIC_models_pcr, '-s', 'LineWidth', 2, 'DisplayName', 'BIC');
xlabel('Lag order q');
ylabel('Information Criterion');
title('PCR: Model Selection');
legend('Location', 'best');
grid on;



%________________________ MODEL PCR TUNED _______________________________ %
[min_MSFE_pcr, idx_min_pcr] = min(MSFE_pcr(:));
[row_idx_pcr, col_idx_pcr] = ind2sub(size(MSFE_pcr), idx_min_pcr);
lag_pcr = lags_pcr(col_idx_pcr);
rol_pcr = rol_window_pcr(row_idx_pcr);

MSFE_grouped_pcr = NaN(T_ar, N_ar);

fprintf('\nOptimal PCR lag (q*) = %d | Optimal window (w*) = %d | MSFE = %.6f\n', ...
        lag_pcr, rol_pcr, min_MSFE_pcr);

q_pcr = lag_pcr;

countryIdx = [ ...
    1   11;
    12  22;
    23  44;
    45  66;
    67  76;
    77  86;
    87  96;
    97  106;
    107 119
];

groupIdx = { [1 2], [3 4], [6 7] };
groupNames = ["South_Europe","Core_Europe","North_Europe"];
N_groups = numel(groupIdx);

Mean_group_MSFE_pcr = NaN(T_pcr, N_groups);

MSE_tot_pcr_final = 0;
for n = 1:N_pcr
    MSE_n_pcr_final = 0;
    
    for t = 1:(T_pcr - rol_pcr - h_pcr + 1)
        Y_train_pcr = Y_pcr(t:(t + rol_pcr - 1), :);
        F_train_pcr = F_hat_pcr(t:(t + rol_pcr - 1), :);
        
        X_pcr = [];
        for j = 1:q_pcr
            X_pcr = [X_pcr, F_train_pcr(q_pcr+1-j:end-j, :)];
        end
        X_pcr = [ones(size(X_pcr, 1), 1), X_pcr];
        
        y_dep_pcr = Y_train_pcr(q_pcr+1:end, n);
        
        ok_pcr = all(isfinite([y_dep_pcr, X_pcr]), 2);
        y__pcr = y_dep_pcr(ok_pcr);
        X__pcr = X_pcr(ok_pcr, :);
        
        if size(X__pcr, 1) < (1 + r_pcr*q_pcr + 1)
            continue;
        end
        
        beta_pcr = X__pcr \ y__pcr;
        
        x_f_pcr = [1];
        for j = 1:q_pcr
            x_f_pcr = [x_f_pcr, F_train_pcr(end-j+1, :)];
        end
        
        yF_pcr = x_f_pcr * beta_pcr;
        Forecast_y(t,n,3) = yF_pcr;
        y_true_pcr = Y_pcr(t + rol_pcr, n);
        err2_pcr = (yF_pcr - y_true_pcr)^2;
        MSE_n_pcr_final = MSE_n_pcr_final + err2_pcr;
        MSFE_save(t, n, 3) = err2_pcr;
        MSFE_grouped_pcr(t, n) = err2_pcr;
    end
    
    MSE_n_pcr_final = MSE_n_pcr_final / (T_pcr - rol_pcr - h_pcr + 1);
    MSE_tot_pcr_final = MSE_tot_pcr_final + MSE_n_pcr_final;
end

for t = 1:(T_pcr - rol_pcr - h_pcr + 1)
    for g = 1:N_groups
        idx_blocks = [];
        for ci = groupIdx{g}
            idx_blocks = [idx_blocks, countryIdx(ci,1):countryIdx(ci,2)];
        end
        valid_errors = MSFE_grouped_pcr(t, idx_blocks);
        Mean_group_MSFE_pcr(t, g) = mean(valid_errors, 'omitnan');
    end
end

MeanGroupMSFE_PCR_table = array2table(Mean_group_MSFE_pcr, 'VariableNames', groupNames);

savepath = fullfile(fl_pt, 'Tables');  

writetable(MeanGroupMSFE_PCR_table, fullfile(savepath, 'MeanGroup_MSFE_PCR.txt'), 'Delimiter', '\t');

RowNames_pcr = strcat("w=", string(rol_window_pcr));
VarNames_pcr = strcat("q=", string(lags_pcr));

MSFE_table_pcr = array2table(MSFE_pcr, 'RowNames', RowNames_pcr, 'VariableNames', VarNames_pcr);

AIC_table_pcr = array2table(AIC_models_pcr, 'VariableNames', compose("q=%d", lags_pcr));
BIC_table_pcr = array2table(BIC_models_pcr, 'VariableNames', compose("q=%d", lags_pcr));

disp('=== Mean Squared Forecast Error (MSFE) — PCR ===');
disp(MSFE_table_pcr);
disp('=== AIC Values ===');
disp(AIC_table_pcr);
disp('=== BIC Values ===');
disp(BIC_table_pcr);

writetable(MSFE_table_pcr, fullfile(savepath, 'MSFE_table_PCR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(AIC_table_pcr,  fullfile(savepath, 'AIC_table_PCR.txt'),  'WriteRowNames', false, 'Delimiter', '\t');
writetable(BIC_table_pcr,  fullfile(savepath, 'BIC_table_PCR.txt'),  'WriteRowNames', false, 'Delimiter', '\t');

fprintf('\nTables successfully saved in:\n%s\n', savepath);
fprintf('  • MeanGroup_MSFE_PCR.txt\n  • MSFE_table_PCR.txt\n  • AIC_table_PCR.txt\n  • BIC_table_PCR.txt\n');





%% -------------------------------------------------------------------- %%
%              SECTION 8: AR Principal Component Regression              %
%  --------------------------------------------------------------------  %

Y_arpcr = data_std;
[T_arpcr, N_arpcr] = size(Y_arpcr);

F_hat_arpcr = scores(:, 1:n_fac);
r_arpcr = size(F_hat_arpcr, 2);

rol_window_arpcr = [150, 252, 500];
lags_arpcr = 4:10;
h_arpcr = 1;

MSFE_arpcr = NaN(length(rol_window_arpcr), length(lags_arpcr));
AIC_store_arpcr = NaN(length(lags_arpcr), N_arpcr);
BIC_store_arpcr = NaN(length(lags_arpcr), N_arpcr);

fprintf('\n=== AR–PCR Model Selection and Rolling Forecast ===\n');

% === Step 1: Compute AIC/BIC on full sample ===
for li = 1:length(lags_arpcr)
    p_arpcr = lags_arpcr(li);

    for n = 1:N_arpcr
        % Build regressors
        X_y = []; X_f = [];
        for j = 1:p_arpcr
            X_y = [X_y, Y_arpcr(p_arpcr+1-j:T_arpcr-j, n)];
            X_f = [X_f, F_hat_arpcr(p_arpcr+1-j:T_arpcr-j, :)];
        end
        X = [ones(size(X_y,1),1), X_y, X_f];
        y = Y_arpcr(p_arpcr+1:T_arpcr, n);

        ok = all(isfinite([y, X]), 2);
        y = y(ok); X = X(ok,:);
        T_eff = size(X,1);
        if T_eff <= size(X,2), continue; end

        beta = X \ y;
        res = y - X*beta;
        sigma2 = (res'*res)/T_eff;
        k = size(X,2);
        logL = -T_eff/2 * (log(2*pi) + log(sigma2) + 1);
        AIC_store_arpcr(li,n) = -2*logL + 2*k;
        BIC_store_arpcr(li,n) = -2*logL + k*log(T_eff);
    end

    fprintf('Computed AIC/BIC for p=%d\n', p_arpcr);
end

% === Step 2: Identify best lag per series (Stock & Watson approach) ===
[~, bestLag_AIC_arpcr] = min(AIC_store_arpcr, [], 1);
[~, bestLag_BIC_arpcr] = min(BIC_store_arpcr, [], 1);

% === Step 3: Compute share of models selecting each lag ===
AIC_share_arpcr = zeros(1, length(lags_arpcr));
BIC_share_arpcr = zeros(1, length(lags_arpcr));
for li = 1:length(lags_arpcr)
    AIC_share_arpcr(li) = mean(bestLag_AIC_arpcr == li);
    BIC_share_arpcr(li) = mean(bestLag_BIC_arpcr == li);
end

% === Step 4: Display as tables ===
VarNames_arpcr = strcat("p=", string(lags_arpcr));
AIC_share_table_arpcr = array2table(AIC_share_arpcr, 'VariableNames', VarNames_arpcr);
BIC_share_table_arpcr = array2table(BIC_share_arpcr, 'VariableNames', VarNames_arpcr);

disp('=== Share of models selecting each lag (AIC) ===');
disp(AIC_share_table_arpcr);
disp('=== Share of models selecting each lag (BIC) ===');
disp(BIC_share_table_arpcr);

% === Step 5: Visualization (compact, blue & yellow bar charts) ===
figure('Position',[200 200 700 300]);

% --- AIC (blue) ---
subplot(1,2,1);
bar(lags_arpcr, AIC_share_arpcr*100, 'FaceColor',[0 0.4470 0.7410]);
title('AIC-selected lag share');
xlabel('Lag order p');
ylabel('Percentage (%)');
ylim([0 100]); xticks(lags_arpcr); grid on; box off;
for i = 1:length(lags_arpcr)
    text(lags_arpcr(i), AIC_share_arpcr(i)*100 + 2, ...
        sprintf('%.0f%%', AIC_share_arpcr(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

% --- BIC (yellow) ---
subplot(1,2,2);
bar(lags_arpcr, BIC_share_arpcr*100, 'FaceColor',[0.9290 0.6940 0.1250]);
title('BIC-selected lag share');
xlabel('Lag order p');
ylabel('');
ylim([0 100]); xticks(lags_arpcr); grid on; box off;
for i = 1:length(lags_arpcr)
    text(lags_arpcr(i), BIC_share_arpcr(i)*100 + 2, ...
        sprintf('%.0f%%', BIC_share_arpcr(i)*100), ...
        'HorizontalAlignment','center','FontSize',9);
end

sgtitle('Percentage of Models Selecting Each Lag AR–PCR (Full Sample)');


% === Step 2: Rolling window forecasting ===
for rw = 1:length(rol_window_arpcr)
    w = rol_window_arpcr(rw);

    for li = 1:length(lags_arpcr)
        p = lags_arpcr(li);
        MSE_tot = 0;

        for n = 1:N_arpcr
            MSE_n = 0;
            for t = 1:(T_arpcr - w - h_arpcr + 1)
                % --- training sample ---
                Y_tr = Y_arpcr(t:(t+w-1), :);
                F_tr = F_hat_arpcr(t:(t+w-1), :);

                % --- construct AR-PCR design ---
                X_y = []; X_f = [];
                for j = 1:p
                    X_y = [X_y, Y_tr(p+1-j:end-j, n)];
                    X_f = [X_f, F_tr(p+1-j:end-j, :)];
                end
                X = [ones(size(X_y,1),1), X_y, X_f];
                y_dep = Y_tr(p+1:end, n);

                % --- OLS fit ---
                beta = X \ y_dep;

                % --- 1-step forecast ---
                x_f = [1];
                for j = 1:p
                    x_f = [x_f, Y_tr(end-j+1, n)];
                end
                for j = 1:p
                    x_f = [x_f, F_tr(end-j+1, :)];
                end

                y_pred = x_f * beta;
                y_true = Y_arpcr(t + w, n);
                MSE_n = MSE_n + (y_pred - y_true)^2;
            end

            MSE_tot = MSE_tot + MSE_n / (T_arpcr - w - h_arpcr + 1);
        end

        MSFE_arpcr(rw, li) = MSE_tot / N_arpcr;
        fprintf('Window=%d | Lag=%d | MSFE=%.6f\n', w, p, MSFE_arpcr(rw, li));
    end
end

% === Display Results ===
RowNames_arpcr = strcat("w=", string(rol_window_arpcr));
VarNames_arpcr = strcat("p=", string(lags_arpcr));

disp('=== Mean Squared Forecast Error (MSFE) ===');
MSFE_table_arpcr = array2table(MSFE_arpcr, ...
    'RowNames', RowNames_arpcr, 'VariableNames', VarNames_arpcr);
disp(MSFE_table_arpcr);

disp('=== AIC and BIC (Full Sample) ===');
IC_table_arpcr = array2table([lags_arpcr(:), AIC_models_arpcr(:), BIC_models_arpcr(:)], ...
    'VariableNames', {'Lag','AIC','BIC'});
disp(IC_table_arpcr);

figure;
heatmap(string(lags_arpcr), string(rol_window_arpcr), MSFE_arpcr);
xlabel('Lag order p');
ylabel('Rolling window w');
title('AR–PCR: 1-step Ahead MSFE');
colorbar;

figure;
plot(lags_arpcr, AIC_models_arpcr, '-o', 'LineWidth', 2, 'DisplayName', 'AIC');
hold on;
plot(lags_arpcr, BIC_models_arpcr, '-s', 'LineWidth', 2, 'DisplayName', 'BIC');
xlabel('Lag order p');
ylabel('Information Criterion');
title('AR–PCR: Model Selection via AIC/BIC');
legend('Location','best');
grid on;


%______________________ MODEL AR PCR TUNED ______________________________ %
[min_MSFE_arpcr, idx_min_arpcr] = min(MSFE_arpcr(:));
[row_idx_arpcr, col_idx_arpcr] = ind2sub(size(MSFE_arpcr), idx_min_arpcr);
rol_opt = rol_window_arpcr(row_idx_arpcr);
lag_opt = lags_arpcr(col_idx_arpcr);

MSFE_grouped_arpcr = NaN(T_ar, N_ar);

fprintf('\nOptimal window = %d | Optimal lag = %d | Min MSFE = %.6f\n', ...
        rol_opt, lag_opt, min_MSFE_arpcr);

countryIdx = [ ...
    1   11;
    12  22;
    23  44;
    45  66;
    67  76;
    77  86;
    87  96;
    97  106;
    107 119
];
groupIdx = { [1 2], [3 4], [6 7] };
groupNames = ["South_Europe","Core_Europe","North_Europe"];
N_groups = numel(groupIdx);

Mean_group_MSFE_arpcr = NaN(T_arpcr, N_groups);

MSE_final = 0;
for n = 1:N_arpcr
    MSE_n = 0;
    for t = 1:(T_arpcr - rol_opt - h_arpcr + 1)
        Y_tr = Y_arpcr(t:(t+rol_opt-1), :);
        F_tr = F_hat_arpcr(t:(t+rol_opt-1), :);

        X_y = []; X_f = [];
        for j = 1:lag_opt
            X_y = [X_y, Y_tr(lag_opt+1-j:end-j, n)];
            X_f = [X_f, F_tr(lag_opt+1-j:end-j, :)];
        end
        X = [ones(size(X_y,1),1), X_y, X_f];
        y_dep = Y_tr(lag_opt+1:end, n);
        beta = X \ y_dep;

        x_f = [1];
        for j = 1:lag_opt
            x_f = [x_f, Y_tr(end-j+1, n)];
        end
        for j = 1:lag_opt
            x_f = [x_f, F_tr(end-j+1, :)];
        end

        y_pred = x_f * beta;
        Forecast_y(t,n,4)= y_pred;
        y_true = Y_arpcr(t + rol_opt, n);
        err2 = (y_pred - y_true)^2;
        MSE_n = MSE_n + err2;
        MSFE_save(t, n, 4) = err2;
        MSFE_grouped_arpcr(t, n) = err2;
    end
    MSE_final = MSE_final + MSE_n / (T_arpcr - rol_opt - h_arpcr + 1);
end

MSE_final = MSE_final / N_arpcr;
fprintf('Final tuned AR–PCR MSFE = %.6f\n', MSE_final);

for t = 1:(T_arpcr - rol_opt - h_arpcr + 1)
    for g = 1:N_groups
        idx_blocks = [];
        for ci = groupIdx{g}
            idx_blocks = [idx_blocks, countryIdx(ci,1):countryIdx(ci,2)];
        end
        valid_errors = MSFE_grouped_arpcr(t, idx_blocks);
        Mean_group_MSFE_arpcr(t, g) = mean(valid_errors, 'omitnan');
    end
end

MeanGroupMSFE_ARPCR_table = array2table(Mean_group_MSFE_arpcr, 'VariableNames', groupNames);

% === Save Tables (FIXED: Don't overwrite AIC/BIC) ===
RowNames_arpcr = strcat("w=", string(rol_window_arpcr));
VarNames_arpcr = strcat("p=", string(lags_arpcr));

MSFE_table_arpcr = array2table(MSFE_arpcr, 'RowNames', RowNames_arpcr, 'VariableNames', VarNames_arpcr);

% Create AIC/BIC tables using the COMPUTED values (not NaN!)
AIC_table_arpcr = array2table(AIC_models_arpcr', 'VariableNames', "AIC", 'RowNames', string(lags_arpcr));
BIC_table_arpcr = array2table(BIC_models_arpcr', 'VariableNames', "BIC", 'RowNames', string(lags_arpcr));

savepath = fullfile(fl_pt, 'Tables');  

writetable(MeanGroupMSFE_ARPCR_table, fullfile(savepath, 'MeanGroup_MSFE_ARPCR.txt'), 'Delimiter', '\t');
writetable(MSFE_table_arpcr, fullfile(savepath, 'MSFE_table_ARPCR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(AIC_table_arpcr, fullfile(savepath, 'AIC_table_ARPCR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(BIC_table_arpcr, fullfile(savepath, 'BIC_table_ARPCR.txt'), 'WriteRowNames', true, 'Delimiter', '\t');

fprintf('\nTables successfully saved in:\n%s\n', savepath);
fprintf('  • MeanGroup_MSFE_ARPCR.txt\n  • MSFE_table_ARPCR.txt\n  • AIC_table_ARPCR.txt\n  • BIC_table_ARPCR.txt\n');




%% -------------------------------------------------------------------- %%
%          SECTION 9: FARM Predict (Rolling Forecast Evaluation)         %
%  --------------------------------------------------------------------  %

Y = data_std;
[T, N] = size(Y);
p = 4;
rol_window_farm = 252;
h_farm = 1;
step = 1;
MSFE_FARM = NaN(length(rol_window_farm), 1);
MSFE_grouped_Farm = NaN(T,N);

fprintf('\n=== FARM Predict: Rolling window (step=%d) ===\n', step);

% === Define country and regional groupings (same as other models) ===
countryIdx = [ ...
    1   11;  % Spain (11)
    12  22;  % Italy (21)
    23  44;  % France (22)
    45  66;  % Germany (22)
    67  76;  % Belgium (10)
    77  86;  % Netherlands (10)
    87  96;  % Sweden (10)
    97  106; % Denmark (10)
    107 119  % Norway (13)
];
groupIdx = { [1 2], [3 4], [6 7] };   % (Spain,Italy), (France,Germany), (Netherlands,Sweden)
groupNames = ["South_Europe","Core_Europe","North_Europe"];
N_groups = numel(groupIdx);

% === Initialize tensors ===
MSFE_tensor = cell(T, N, 5);   % 5th index for FARM
Mean_group_MSFE_FARM = NaN(T, N_groups);

for r = 1:length(rol_window_farm)
    w_farm = rol_window_farm(r);
    MSE_tot_farm = 0;
    count_farm = 0;
    n_iter = floor((T - w_farm - h_farm + 1) / step);

    fprintf('\n--- Rolling window = %d | Total iterations ≈ %d ---\n', w_farm, n_iter);

    for t = 1:step:(T - w_farm - h_farm + 1)
        iter = ceil(t / step);
        pct_done = 100 * iter / n_iter;
        fprintf('\rWindow %d | Iteration %d/%d (%.1f%% done)', w_farm, iter, n_iter, pct_done);
        drawnow;

        % === 1. Factor extraction ===
        Y_train = Y(t:(t + w_farm - 1), :);
        [U, S, V] = svd(Y_train, 'econ');
        F_tr = U(:,1) * S(1,1);
        L_tr = V(:,1);
        U_tr = Y_train - F_tr * L_tr';

        % === 2. Factor AR(p) ===
        if w_farm - p <= 1, continue; end
        Xf = [];
        for j = 1:p
            Xf = [Xf, F_tr(p+1-j:end-j)];
        end
        Xf = [ones(size(Xf,1),1), Xf];
        yf = F_tr(p+1:end);
        beta_f = (Xf' * Xf) \ (Xf' * yf);

        % === 3. Predict next factor ===
        x_last = [1];
        for j = 1:p
            x_last = [x_last, F_tr(end-j+1)];
        end
        F_pred_next = x_last * beta_f;

                % === 4. Common and idiosyncratic predictions ===
        Y_common_pred = L_tr * F_pred_next;
        Y_idio_pred = zeros(N,1);

        L = 1;  % number of residual lags to use

        for i = 1:N
            % --- Build response and lagged predictor matrix ---
            y_i = U_tr(L+1:end, i);   % drop first L obs for alignment
            X_i = [];

            % Stack the 4 lagged residuals of all other series
            for lag = 1:L
                X_i = [X_i, U_tr(L+1-lag:end-lag, setdiff(1:N, i))];
            end

            % --- blocked time-series CV for lambda selection ---
            lambda_grid = logspace(-4, 1, 30);
            K_blocks = 3;
            T_i = size(X_i,1);
            block_size = floor(T_i / (K_blocks + 1));
            mse_lambda = zeros(length(lambda_grid), K_blocks);

            for k = 1:K_blocks
                train_idx = 1:(k*block_size);
                val_idx = (k*block_size+1):min((k+1)*block_size, T_i);
                if isempty(val_idx), break; end

                X_train = X_i(train_idx,:);
                y_train = y_i(train_idx);
                X_val = X_i(val_idx,:);
                y_val = y_i(val_idx);

                [B_tmp, Fit_tmp] = lasso(X_train, y_train, ...
                    'Lambda', lambda_grid, 'Standardize', true);
                Y_val_pred = X_val * B_tmp + Fit_tmp.Intercept;
                mse_lambda(:,k) = mean((y_val - Y_val_pred).^2, 1);
            end

            mean_mse = mean(mse_lambda, 2, 'omitnan');
            [~, idx_best] = min(mean_mse);
            lambda_best = lambda_grid(idx_best);

            % --- Fit final LASSO on full sample ---
            [B, FitInfo] = lasso(X_i, y_i, 'Lambda', lambda_best, 'Standardize', true);
            beta_i = B;
            b0_i  = FitInfo.Intercept;

            % --- Build 4-lag predictor for next period ---
            W_i_t1 = [];
            for lag = 1:L
                W_i_t1 = [W_i_t1, U_tr(end-lag+1, setdiff(1:N, i))];
            end

            % --- Predict next idiosyncratic residual ---
            Y_idio_pred(i) = W_i_t1 * beta_i + b0_i;
        end

        % === 5. Forecast and error ===
        Y_FARM_pred = Y_common_pred + Y_idio_pred;
        Y_true_next = Y(t + w_farm, :)';
        err_vec = Y_FARM_pred - Y_true_next;
        Forecast_y(t,:,5) = Y_FARM_pred;
        
        % Store individual squared errors in tensor (FIXED)
        sq_errors = err_vec'.^2;  % 1 x N vector
        for i = 1:N
            MSFE_save(t, i, 5) = sq_errors(i);
            MSFE_grouped_Farm(t, i) = sq_errors(i);
        end

MSE_tot_farm = MSE_tot_farm + mean(err_vec.^2);
count_farm = count_farm + 1;
    end

    fprintf('\nWindow %d completed\n', w_farm);
    MSFE_FARM(r) = MSE_tot_farm / count_farm;
    fprintf('Rolling window = %d | FARM MSFE = %.6f\n', w_farm, MSFE_FARM(r));
end

% === Compute mean MSFE per group at each time t ===
for t = 1:(T - rol_window_farm - h_farm + 1)
    for g = 1:N_groups
        idx_blocks = [];
        for ci = groupIdx{g}
            idx_blocks = [idx_blocks, countryIdx(ci,1):countryIdx(ci,2)];
        end
        valid_errors = MSFE_grouped_Farm(t, idx_blocks);
        Mean_group_MSFE_FARM(t, g) = mean(valid_errors, 'omitnan');
    end
end

% === Create table for group-level MSFE ===
MeanGroupMSFE_FARM_table = array2table(Mean_group_MSFE_FARM, 'VariableNames', groupNames);

% === Save tables ===
savepath = fullfile(fl_pt, 'Tables');  

% Save main FARM MSFE summary and group means
RowNames_FARM = strcat("w=", string(rol_window_farm));
MSFE_table_FARM = array2table(MSFE_FARM, 'RowNames', RowNames_FARM, 'VariableNames', {'MSFE'});
writetable(MSFE_table_FARM, fullfile(savepath, 'MSFE_table_FARM.txt'), 'WriteRowNames', true, 'Delimiter', '\t');
writetable(MeanGroupMSFE_FARM_table, fullfile(savepath, 'MeanGroup_MSFE_FARM.txt'), 'Delimiter', '\t');

fprintf('\nTables successfully saved in:\n%s\n', savepath);
fprintf('  • MSFE_table_FARM.txt\n  • MeanGroup_MSFE_FARM.txt\n');


% _____________ FARM for a specific stock (g) __________________________ %

data_matrix = data_std;
[n_obs, n_assets] = size(data_matrix);
lag_grid = 1:10;  % Grid for AR lag order
rolling_win = 252;
forecast_horizon = 1;
time_step = 1;

% === Focus on single asset ===
target_asset = 10;  % Specific asset index
fprintf('\n=== FARM Single Asset Prediction: asset=%d (p grid search) ===\n', target_asset);

% === Initialize storage ===
n_lags = length(lag_grid);
MSFE_by_lag = NaN(n_lags, 1);
MSFE_timeseries = NaN(n_obs, n_lags);
Predictions = NaN(n_obs, n_lags);

for lag_idx = 1:n_lags
    current_lag = lag_grid(lag_idx);
    
    cumulative_mse = 0;
    valid_count = 0;
    n_iterations = floor((n_obs - rolling_win - forecast_horizon + 1) / time_step);

    fprintf('\n--- AR lag p=%d | Rolling window=%d | Iterations ≈ %d ---\n', ...
        current_lag, rolling_win, n_iterations);

    for time_idx = 1:time_step:(n_obs - rolling_win - forecast_horizon + 1)
        iter_num = ceil(time_idx / time_step);
        progress_pct = 100 * iter_num / n_iterations;
        fprintf('\rLag p=%d | Iteration %d/%d (%.1f%% done)', ...
            current_lag, iter_num, n_iterations, progress_pct);
        drawnow;

        % === 1. Extract common factor via SVD ===
        training_data = data_matrix(time_idx:(time_idx + rolling_win - 1), :);
        [left_sv, singular_vals, right_sv] = svd(training_data, 'econ');
        
        common_factor = left_sv(:, 1) * singular_vals(1, 1);
        factor_loadings = right_sv(:, 1);
        idiosync_residuals = training_data - common_factor * factor_loadings';

        % === 2. Fit AR(p) model to common factor ===
        if rolling_win - current_lag <= 1, continue; end
        
        % Build lagged matrix
        factor_lags = [];
        for lag = 1:current_lag
            factor_lags = [factor_lags, common_factor(current_lag+1-lag:end-lag)];
        end
        factor_lags = [ones(size(factor_lags, 1), 1), factor_lags];
        
        factor_response = common_factor(current_lag+1:end);
        ar_coefficients = (factor_lags' * factor_lags) \ (factor_lags' * factor_response);

        % === 3. Forecast next period factor ===
        last_observation = [1];
        for lag = 1:current_lag
            last_observation = [last_observation, common_factor(end-lag+1)];
        end
        factor_forecast = last_observation * ar_coefficients;

        % === 4. Common component prediction ===
        common_component_pred = factor_loadings * factor_forecast;

        % === 5. Idiosyncratic prediction for target asset only ===
        asset_idx = target_asset;
        response_vec = idiosync_residuals(2:end, asset_idx);
        predictor_mat = idiosync_residuals(1:end-1, setdiff(1:n_assets, asset_idx));

        % --- Blocked time-series CV for lambda tuning ---
        lambda_sequence = logspace(-4, 1, 30);
        n_cv_blocks = 3;
        n_train_obs = size(predictor_mat, 1);
        block_length = floor(n_train_obs / (n_cv_blocks + 1));
        cv_errors = zeros(length(lambda_sequence), n_cv_blocks);

        for block_idx = 1:n_cv_blocks
            train_indices = 1:(block_idx * block_length);
            valid_indices = (block_idx*block_length+1):min((block_idx+1)*block_length, n_train_obs);
            if isempty(valid_indices), break; end

            X_train_block = predictor_mat(train_indices, :);
            y_train_block = response_vec(train_indices);
            X_valid_block = predictor_mat(valid_indices, :);
            y_valid_block = response_vec(valid_indices);

            [lasso_coefs, lasso_info] = lasso(X_train_block, y_train_block, ...
                'Lambda', lambda_sequence, 'Standardize', true);
            validation_preds = X_valid_block * lasso_coefs + lasso_info.Intercept;
            cv_errors(:, block_idx) = mean((y_valid_block - validation_preds).^2, 1);
        end

        avg_cv_error = mean(cv_errors, 2, 'omitnan');
        [~, optimal_lambda_idx] = min(avg_cv_error);
        optimal_lambda = lambda_sequence(optimal_lambda_idx);

        % Refit with optimal lambda
        [final_coefs, final_info] = lasso(predictor_mat, response_vec, ...
            'Lambda', optimal_lambda, 'Standardize', true);
        lasso_beta = final_coefs;
        lasso_intercept = final_info.Intercept;
        
        most_recent_residuals = idiosync_residuals(end, setdiff(1:n_assets, asset_idx));
        idiosync_forecast = most_recent_residuals * lasso_beta + lasso_intercept;

        % === 6. Combined forecast and error ===
        total_forecast = common_component_pred(asset_idx) + idiosync_forecast;
        actual_value = data_matrix(time_idx + rolling_win, asset_idx);
        forecast_error = total_forecast - actual_value;
        
        % Store results
        Predictions(time_idx, lag_idx) = total_forecast;
        MSFE_timeseries(time_idx, lag_idx) = forecast_error^2;

        cumulative_mse = cumulative_mse + forecast_error^2;
        valid_count = valid_count + 1;
    end

    fprintf('\n');
    MSFE_by_lag(lag_idx) = cumulative_mse / valid_count;
    fprintf('Lag p=%d completed | MSFE = %.6f\n', current_lag, MSFE_by_lag(lag_idx));
end

% === Find optimal lag ===
[min_msfe, optimal_lag_idx] = min(MSFE_by_lag);
optimal_p = lag_grid(optimal_lag_idx);

% === Create summary table ===
Results_table = array2table([lag_grid', MSFE_by_lag], ...
    'VariableNames', {'AR_Lag_p', sprintf('MSFE_Asset_%d', target_asset)});

% === Display results ===
fprintf('\n=== FARM Results for Asset %d ===\n', target_asset);
disp(Results_table);
fprintf('\nOptimal lag: p = %d (MSFE = %.6f)\n', optimal_p, min_msfe);

% === Save results ===
savepath = fullfile(fl_pt, 'Tables');  
writetable(Results_table, fullfile(output_dir, sprintf('FARM_Asset_%d_LagGrid.txt', target_asset)), ...
    'Delimiter', '\t');

fprintf('\nTable saved: FARM_Asset_%d_LagGrid.txt\n', target_asset);

% === Visualization 1: MSFE vs Lag Order ===
figure('Name', 'MSFE vs AR Lag Order');
plot(lag_grid, MSFE_by_lag, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
plot(optimal_p, min_msfe, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('AR Lag Order (p)', 'FontSize', 12);
ylabel('Mean Squared Forecast Error', 'FontSize', 12);
title(sprintf('FARM Model Performance - Asset %d', target_asset), 'FontSize', 13);
legend('MSFE', sprintf('Optimal p=%d', optimal_p), 'Location', 'best');
grid on;
hold off;

% === Visualization 2: Forecast vs Actual (optimal p) ===
figure('Name', sprintf('FARM Forecast - Asset %d (p=%d)', target_asset, optimal_p));
valid_times = ~isnan(Predictions(:, optimal_lag_idx));
time_vector = find(valid_times);

plot(time_vector, data_matrix(valid_times, target_asset), 'b-', ...
    'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(time_vector, Predictions(valid_times, optimal_lag_idx), 'r--', ...
    'LineWidth', 1.5, 'DisplayName', sprintf('FARM Forecast (p=%d)', optimal_p));
legend('show', 'Location', 'best');
title(sprintf('FARM Forecast vs Actual - Asset %d (Optimal p=%d)', target_asset, optimal_p), ...
    'FontSize', 12);
xlabel('Time Index', 'FontSize', 11);
ylabel('Standardized Value', 'FontSize', 11);
grid on;
hold off;

% === Visualization 3: Forecast Errors Over Time ===
figure('Name', 'Forecast Errors');
plot(time_vector, sqrt(MSFE_timeseries(valid_times, optimal_lag_idx)), 'k-', 'LineWidth', 1);
title(sprintf('Absolute Forecast Errors - Asset %d (p=%d)', target_asset, optimal_p), ...
    'FontSize', 12);
xlabel('Time Index', 'FontSize', 11);
ylabel('Absolute Error', 'FontSize', 11);
grid on;




%% -------------------------------------------------------------------- %%
%                Section 10: Cumulative MSFE                             %
%  --------------------------------------------------------------------  %
k  = find(isnan(Forecast_y(:,1,1)), 1, 'first');
kk = find(isnan(Forecast_y(:,1,5)), 1, 'first');

MSFE_save_final(:,:,1) = MSFE_save(1:5167,:,1);                 % AR
MSFE_save_final(:,:,2) = MSFE_save(1:5167,:,2);                 % VAR
MSFE_save_final(:,:,3:4) = MSFE_save(1:5167,:,3:4);             % PCR - AR PCR
MSFE_save_final(:,:,5) = MSFE_save(249:5415,:,5);    % FARM
MSFE_save_final(:,:,6) = MSFE_save_var_lasso(1:5167,:);         % VAR LASSO
MSFE_save_backup = MSFE_save_final;


% _________________________ Figure Cumulative __________________________ %
% (1) Cumulative MSFE tensor
[T, N, S] = size(MSFE_save_final);
MSFE_cum = NaN(T, N, S);

for i = 1:S
    MSFE_cum(:, :, i) = cumsum(MSFE_save_final(:, :, i), 1, 'omitnan');
end

% (2) Load time vector
savepath = fullfile(fl_pt, 'Data');  
time_data = readtable(fullfile(savepath,"/dates_daily_new.csv"));
dates = datetime(time_data.Date);
dates = dates(end - T + 1 : end);

% (3) Choose specific series g
g = 14;

% (4) Compute ratios: numerator = FARM Predict (slice 5)
ratio_AR       = MSFE_cum(:, g, 5) ./ MSFE_cum(:, g, 1);   % FARM / AR
ratio_SR       = MSFE_cum(:, g, 5) ./ MSFE_cum(:, g, 2);   % FARM / VAR
ratio_PCR      = MSFE_cum(:, g, 5) ./ MSFE_cum(:, g, 3);   % FARM / PCR
ratio_ARPCR    = MSFE_cum(:, g, 5) ./ MSFE_cum(:, g, 4);   % FARM / AR-PCR
ratio_VARLASSO = MSFE_cum(:, g, 5) ./ MSFE_cum(:, g, 6);   % FARM / VAR-LASSO

% (5) Plot (like Figure 6)
figure('Position',[200 200 950 450]);
hold on; grid on; box on;

plot(dates, ratio_AR,       'LineWidth',1.2);
plot(dates, ratio_SR,       'LineWidth',1.2);
plot(dates, ratio_PCR,      'LineWidth',1.2);
plot(dates, ratio_ARPCR,    'LineWidth',1.2);
plot(dates, ratio_VARLASSO, 'LineWidth',1.2);

yline(1,'--k','LineWidth',0.8);
ylim([0.5 1.5]);
xlim([datetime(2003,1,1) datetime(2019,1,1)]);

legend({'FARM/AR','FARM/VAR','FARM/PCR','FARM/AR-PCR','FARM/VAR-LASSO'},'Location','best');
xlabel('Date');
ylabel('Cumulative MSFE Ratio');
title(sprintf('Ratios of Cumulative MSFE (FARM Predict as numerator, g = %d)', g));
%%
% _________________________ Group-Level Cumulative Ratios __________________________ %

% Define trimming indices as in your main code
k  = find(isnan(Forecast_y(:,1,1)), 1, 'first');
kk = find(isnan(Forecast_y(:,1,5)), 1, 'first');

T_trim = 5167;     % same as your per-stock trimmed window
MSFE_group_final = NaN(T_trim, 3, 6);

% --- Apply windsorization consistent with MSFE_save_final structure ---
MSFE_group_final(:,:,1) = Mean_group_MSFE(1:T_trim,:);                    % AR
MSFE_group_final(:,:,3) = Mean_group_MSFE_pcr(1:T_trim,:);                % PCR
MSFE_group_final(:,:,4) = Mean_group_MSFE_arpcr(1:T_trim,:);              % AR-PCR
MSFE_group_final(:,:,6) = Mean_group_MSFE_var_lasso(1:T_trim,:);          % VAR-LASSO
MSFE_group_final(:,:,5) = Mean_group_MSFE_FARM(249:5415,:);    % FARM
MSFE_group_final(:,:,2) = Mean_group_MSFE_var(1:T_trim,:);                % VAR

MSFE_group_backup = MSFE_group_final;

for i = 1:3
 mean(MSFE_group_backup(:,i))
end



[Tg, G, Sg] = size(MSFE_group_final);
MSFE_group_cum = NaN(Tg, G, Sg);
for i = 1:Sg
    MSFE_group_cum(:, :, i) = cumsum(MSFE_group_final(:, :, i), 1, 'omitnan');
end


dates = datetime(time_data.Date);
dates = dates(end - Tg + 1 : end);

groupNames = ["South_Europe","Core_Europe","North_Europe"];

figure('Position',[200 200 1000 700]);

for g = 1:G
    % ---- Compute ratios: FARM Predict (slice 5) as numerator ----
    ratio_AR       = MSFE_group_cum(:, g, 5) ./ MSFE_group_cum(:, g, 1); % FARM / AR
    ratio_PCR      = MSFE_group_cum(:, g, 5) ./ MSFE_group_cum(:, g, 3); % FARM / PCR
    ratio_ARPCR    = MSFE_group_cum(:, g, 5) ./ MSFE_group_cum(:, g, 4); % FARM / AR-PCR
    ratio_VARLASSO = MSFE_group_cum(:, g, 5) ./ MSFE_group_cum(:, g, 6); % FARM / VAR-LASSO
    ratio_VAR = MSFE_group_cum(:, g, 5) ./ MSFE_group_cum(:, g, 2);

    % ---- Plot ----
    subplot(3,1,g);
    hold on; grid on; box on;
    plot(dates, ratio_AR,       'LineWidth',1.2);
    plot(dates, ratio_PCR,      'LineWidth',1.2);
    plot(dates, ratio_ARPCR,    'LineWidth',1.2);
    plot(dates, ratio_VARLASSO, 'LineWidth',1.2);
    plot(dates, ratio_VAR,      'LineWidth',1.2);
    yline(1,'--k','LineWidth',0.8);

    title(groupNames(g), 'Interpreter','none');
    ylabel('Cumulative MSFE Ratio');
    if g == G
        xlabel('Date');
    end
    ylim([0.5 1.5]);
    xlim([datetime(2003,1,1) datetime(2019,1,1)]);
end

legend({'FARM/AR','FARM/PCR','FARM/AR-PCR','FARM/VAR-LASSO'}, ...
       'Location','best','Box','off','Orientation','horizontal');
sgtitle('Ratios of Cumulative MSFE (FARM Predict as numerator, by region)');

%%

% _________________________ Figure Forecasting (No VAR) __________________________ %
% Find index of first NaN in first model and FARM
k  = find(isnan(Forecast_y(:,1,1)), 1, 'first');
kk = find(isnan(Forecast_y(:,1,5)), 1, 'first');

g = 14;

% --- Define crisis periods ---
crises = [datetime(2007,7,1)  datetime(2009,6,30);
          datetime(2010,5,1)  datetime(2012,6,30);
          datetime(2020,2,1)  datetime(2020,6,30)];
crisis_colors = [0.9 0.7 0.7;
                 0.9 0.8 0.6;
                 0.7 0.7 0.9];

% --- Extract aligned data (excluding VAR) ---
Forecast_y_tot = cat(3, ...
    Forecast_y(1:5167,:,1), ...      % AR
    Forecast_y(1:5167,:,3:4), ...    % PCR, AR-PCR
    Forecast_y(((kk)-(k-1)):(kk-1),:,5), ... % FARM
    Forecast_y_var_lasso(1:5167,:)); % VAR-LASSO

Y_true = Y(501:end, g);
dates_plot = datetime(time_data.Date);
dates_plot = dates_plot(end - T + 1 : end);
[Tf, Nf, S] = size(Forecast_y_tot);
Forecast_trim = Forecast_y_tot(:, g, :);

% --- Your color scheme ---
colors_models = [
    0.85 0.15 0.15;  % AR - Red
    0.20 0.50 0.85;  % PCR - Blue
    0.95 0.75 0.15;  % AR-PCR - Yellow
    0.15 0.70 0.25;  % FARM - Green (highlighted)
    0.65 0.65 0.65]; % VAR-LASSO - Gray

color_true = [0.10 0.10 0.10];  % Dark for true values

model_names = {'AR','PCR','AR-PCR','FARM','VAR-LASSO'};

% --- Plot ---
figure('Position',[200 200 950 450]);
hold on; grid on; box on;

yl = [-8 10];
for j = 1:size(crises,1)
    fill([crises(j,1) crises(j,2) crises(j,2) crises(j,1)], ...
         [yl(1) yl(1) yl(2) yl(2)], crisis_colors(j,:), ...
         'FaceAlpha',0.25,'EdgeColor','none','HandleVisibility','off');
end

% True value
h_true = plot(dates_plot, Y_true, 'Color', color_true, 'LineWidth', 1);

% Baseline forecasts (thinner, dotted)
h_models = gobjects(5,1);
for i = 1:3
    h_models(i) = plot(dates_plot, Forecast_trim(:,1,i), ...
        'LineWidth', 1.3, 'Color', colors_models(i,:), 'LineStyle', ':');
end

% FARM - highlighted (bold, solid)
h_models(4) = plot(dates_plot, Forecast_trim(:,1,4), ...
    'Color', colors_models(4,:), 'LineWidth', 1.5, 'LineStyle', ':');

% VAR-LASSO (dotted)
h_models(5) = plot(dates_plot, Forecast_trim(:,1,5), ...
    'LineWidth', 1.3, 'Color', colors_models(5,:), 'LineStyle', ':');

% Legend
legend_entries = [h_true, h_models'];
legend_labels = {'True Value', model_names{:}};
legend(legend_entries, legend_labels, 'Location', 'best', 'Box', 'off');

xlabel('Date'); ylabel('Value');
title(sprintf('Forecast Comparison for g = %d', g));
xlim([datetime(2005,1,1) datetime(2025,1,1)]);
ylim(yl);
hold off;


%%
% _________________________ Figure Forecasting by Country __________________________ %


% --- Find index of first NaN in first model and FARM ---
k  = find(isnan(Forecast_y(:,1,1)), 1, 'first');
kk = find(isnan(Forecast_y(:,1,5)), 1, 'first');

% --- Crisis periods ---
crises = [datetime(2007,7,1)  datetime(2009,6,30);
          datetime(2010,5,1)  datetime(2012,6,30);
          datetime(2020,2,1)  datetime(2020,6,30)];
crisis_colors = [0.9 0.7 0.7;
                 0.9 0.8 0.6;
                 0.7 0.7 0.9];

% --- Concatenate aligned forecasts (excluding VAR) ---
Forecast_y_tot = cat(3, ...
    Forecast_y(1:5167,:,1), ...      % AR
    Forecast_y(1:5167,:,3:4), ...    % PCR, AR-PCR
    Forecast_y(((kk)-(k-1)):(kk-1),:,5), ... % FARM
    Forecast_y_var_lasso(1:5167,:)); % VAR-LASSO

% --- Country representatives ---
biggest_idx = [9, 15, 29, 47, 68, 78, 88, 98, 110]; 
countryNames = {'Spain','Italy','France','Germany','Belgium','Netherlands','Sweden','Denmark','Norway'};
stockNames   = {'TEF\_MC','ISP\_MI','BNP\_PA','VOW3\_DE','LOTB\_BR','SBMO\_AS','ASSA\_B\_ST','VWS\_CO','MOWI\_OL'};


% --- Country representatives ---
biggest_idx = [9, 15, 29, 47, 68, 78, 88, 98, 110]; 
countryNames = {'Spain','Italy','France','Germany','Belgium','Netherlands','Sweden','Denmark','Norway'};

% --- Dates ---
dates_plot = datetime(time_data.Date);
dates_plot = dates_plot(end - size(Forecast_y_tot,1) + 1 : end);

% --- Colors ---
colors_models = [
    0.85 0.15 0.15;  % AR - red
    0.20 0.50 0.85;  % PCR - blue
    0.95 0.75 0.15;  % AR-PCR - yellow
    0.15 0.70 0.25;  % FARM - green
    0.65 0.65 0.65]; % VAR-LASSO - gray
color_true = [0.10 0.10 0.10];

% --- Plot layout ---
figure('Position',[100 100 1200 800]);
tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
yl = [-8 10];

for c = 1:numel(biggest_idx)
    g = biggest_idx(c);
    nexttile(c);
    hold on; box on; grid on;

    % Crisis shading
    for j = 1:size(crises,1)
        fill([crises(j,1) crises(j,2) crises(j,2) crises(j,1)], ...
             [yl(1) yl(1) yl(2) yl(2)], crisis_colors(j,:), ...
             'FaceAlpha',0.25,'EdgeColor','none');
    end

    % True series
    Y_true = Y(501:end, g);
    plot(dates_plot, Y_true, 'Color', color_true, 'LineWidth', 1);

    % Forecasts
    Forecast_trim = squeeze(Forecast_y_tot(:, g, :));

    % Baseline models (dotted)
    for i = 1:3
        plot(dates_plot, Forecast_trim(:, i), ...
             'LineWidth', 1.2, 'LineStyle', ':', 'Color', colors_models(i,:));
    end

    % FARM (solid)
    plot(dates_plot, Forecast_trim(:, 4), ...
         'Color', colors_models(4,:), 'LineWidth', 1.8, 'LineStyle', ':');

    % VAR-LASSO (dotted)
    plot(dates_plot, Forecast_trim(:, 5), ...
         'LineWidth', 1.2, 'LineStyle', ':', 'Color', colors_models(5,:));

    title(sprintf('%s (%s)', countryNames{c}, stockNames{c}), 'FontWeight','bold');
    ylim(yl);
    xlim([datetime(2005,1,1) datetime(2025,1,1)]);
    if c > 6, xlabel('Date'); end
    if mod(c,3)==1, ylabel('Value'); end
end

sgtitle('Forecast Comparison for the Biggest Stock of Each Country (No VAR, No Legend)', 'FontWeight','bold');











%% -------------------------------------------------------------------- %%
%               SECTION 11: Time-Series Visualization                   %
%  --------------------------------------------------------------------  %

% Convert dates
dates = datetime(time_data.Date);

% --- Define crisis periods ---
crises = [datetime(2007,7,1)  datetime(2009,6,30);   % Global Financial Crisis
          datetime(2010,5,1)  datetime(2012,6,30);   % Eurozone Debt Crisis
          datetime(2020,2,1)  datetime(2020,6,30)];  % COVID-19 Crash
crisis_labels = {'GFC', 'Eurozone Crisis', 'COVID-19'};
crisis_colors = [0.9 0.7 0.7;  % Light red
                 0.9 0.8 0.6;  % Light orange
                 0.7 0.7 0.9]; % Light blue

% -------------------------------------------------------------------- %%
%                    (1) All Returns with Crisis Shading                 %
figure('Position', [100 100 1200 500], 'Color', 'w');
hold on; box on; grid on; grid minor;

max_abs = max(abs(data_std(:))) * 1.1;
ylim([-max_abs, max_abs]);

crisis_colors = [0.85 0.90 0.75;
                 0.85 0.90 0.75;
                 0.80 0.88 0.95];
crisis_labels = {'Financial Crises', 'Eurozone Crisis', 'COVID-19'};

for c = 1:size(crises, 1)
    fill([crises(c,1) crises(c,2) crises(c,2) crises(c,1)], ...
         [-max_abs -max_abs max_abs max_abs], crisis_colors(c,:), ...
         'EdgeColor', 'none', 'FaceAlpha', 0.3);
    mid_date = crises(c,1) + (crises(c,2)-crises(c,1))/2;
    text(mid_date, max_abs*0.92, crisis_labels{c}, ...
         'HorizontalAlignment', 'center', 'FontSize', 9, ...
         'FontWeight', 'bold', 'Color', [0.4 0.4 0.4]);
end

plot(dates, data_std, 'Color', [0.5 0.05 0.05 0.25], 'LineWidth', 0.5);
yline(0, 'k--', 'LineWidth', 1.0);

title('Standardized Daily Returns – All European Stocks', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Date', 'FontSize', 11);
ylabel('Standardized Returns', 'FontSize', 11);
xlim([min(dates) max(dates)]);
set(gca, 'Layer', 'top');
hold off;


figure('Position', [100 100 1400 900], 'Color', 'w');
tiledlayout(3, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

max_abs_global = max(abs(data_std(:))) * 1.1;
crisis_colors = [0.85 0.90 0.75;
                 0.85 0.90 0.75;
                 0.80 0.88 0.95];
series_color = [0.50 0.05 0.05];

for k = 1:length(countryNames)
    nexttile;
    hold on; box on; grid on; grid minor;
    
    idx_range = countryIdx(k,1):countryIdx(k,2);
    n_stocks = length(idx_range);
    country_data = data_std(:, idx_range);
    
    for c = 1:size(crises,1)
        fill([crises(c,1) crises(c,2) crises(c,2) crises(c,1)], ...
             [-max_abs_global -max_abs_global max_abs_global max_abs_global], ...
             crisis_colors(c,:), 'EdgeColor', 'none', 'FaceAlpha', 0.25);
    end
    
    for j = 1:n_stocks
        alpha = 0.25 + 0.3*(j/n_stocks);
        plot(dates, country_data(:, j), ...
             'Color', [series_color alpha], 'LineWidth', 0.6);
    end
    
    mean_ret = mean(country_data, 2, 'omitnan');
    plot(dates, mean_ret, 'Color', series_color, 'LineWidth', 2.0);
    
    yline(0, 'k--', 'LineWidth', 0.8);
    ylim([-max_abs_global, max_abs_global]);
    xlim([min(dates) max(dates)]);
    
    title(sprintf('%s (n = %d)', countryNames{k}, n_stocks), ...
          'FontWeight', 'bold', 'FontSize', 11, 'Color', [0.1 0.1 0.1]);
    xlabel('Date', 'FontSize', 9, 'FontWeight', 'bold');
    ylabel('Returns', 'FontSize', 9, 'FontWeight', 'bold');
    set(gca, 'FontSize', 9, 'LineWidth', 0.8, 'Layer', 'top', ...
        'GridColor', [0.75 0.75 0.75], 'MinorGridColor', [0.88 0.88 0.88]);
    
    vol = std(country_data(:), 'omitnan');
    text(0.02, 0.95, sprintf('\\sigma = %.3f', vol), ...
         'Units', 'normalized', 'FontSize', 8, 'FontWeight', 'bold', ...
         'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7], ...
         'Color', [0.25 0.25 0.25], 'VerticalAlignment', 'top');
    
    hold off;
end

sgtitle('Standardized Daily Returns by Country (Shaded = Crisis Periods)', ...
        'FontSize', 15, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1]);


figure('Position', [100 100 1200 500], 'Color', 'w');
hold on; box on; grid on; grid minor;

window = 30;
volatility = zeros(length(dates)-window+1, length(countryNames));

for k = 1:length(countryNames)
    idx_range = countryIdx(k,1):countryIdx(k,2);
    country_data = data_std(:, idx_range);
    country_avg = mean(country_data, 2, 'omitnan');
    
    for t = window:length(dates)
        volatility(t-window+1, k) = std(country_avg(t-window+1:t), 'omitnan');
    end
end

vol_dates = dates(window:end);
crisis_colors = [0.85 0.90 0.75;
                 0.85 0.90 0.75;
                 0.80 0.88 0.95];
crisis_labels = {'Financial Crises','Eurozone Crisis','COVID-19'};
country_colors = lines(length(countryNames));

yl_vol = [0, max(volatility(:))*1.1];
ylim(yl_vol);

for c = 1:size(crises,1)
    fill([crises(c,1) crises(c,2) crises(c,2) crises(c,1)], ...
         [yl_vol(1) yl_vol(1) yl_vol(2) yl_vol(2)], crisis_colors(c,:), ...
         'EdgeColor','none','FaceAlpha',0.25);
    mid_date = crises(c,1) + (crises(c,2)-crises(c,1))/2;
    text(mid_date, yl_vol(2)*0.92, crisis_labels{c}, ...
         'HorizontalAlignment','center','FontSize',10, ...
         'FontWeight','bold','Color',[0.25 0.25 0.25]);
end

for k = 1:length(countryNames)
    plot(vol_dates, volatility(:,k), 'LineWidth',1.5, ...
         'DisplayName',countryNames{k}, 'Color',country_colors(k,:));
end

yline(0,'k--','LineWidth',1);
title(sprintf('%d-Day Rolling Volatility by Country', window), ...
      'FontSize',14,'FontWeight','bold','Color',[0.1 0.1 0.1]);
xlabel('Date','FontSize',11,'FontWeight','bold');
ylabel('Volatility (Std Dev)','FontSize',11,'FontWeight','bold');
legend('Location','northwest','FontSize',9);
set(gca,'FontSize',10,'LineWidth',0.9,'Layer','top', ...
    'GridColor',[0.75 0.75 0.75],'MinorGridColor',[0.88 0.88 0.88]);
xlim([min(vol_dates) max(vol_dates)]);

hold off;





%% -------------------------------------------------------------------- %%
%          SECTION 12: MODEL COMPARISON SUMMARY TABLE                   %
%  --------------------------------------------------------------------  %

fprintf('\n');
fprintf('========================================================================\n');
fprintf('                   OPTIMAL MODEL CONFIGURATIONS                         \n');
fprintf('========================================================================\n');

% --- Collect optimal parameters and MSFE for each model ---

% AR Model
[min_MSFE_ar, idx_ar] = min(MSFE_ar(:));
[row_ar, col_ar] = ind2sub(size(MSFE_ar), idx_ar);
optimal_window_ar = rol_window_ar(row_ar);
optimal_lag_ar = lags_ar(col_ar);

% VAR Model
[min_MSFE_var, idx_var] = min(MSFE_var(:));
[row_var, col_var] = ind2sub(size(MSFE_var), idx_var);
optimal_window_var = rol_window_var(row_var);
optimal_lag_var = lags_var(col_var);

% VAR-LASSO Model (uses tuned VAR parameters)
optimal_window_var_lasso = rol_var;
optimal_lag_var_lasso = lag_var;
min_MSFE_var_lasso = MSE_total_var_lasso;

% PCR Model
[min_MSFE_pcr, idx_pcr] = min(MSFE_pcr(:));
[row_pcr, col_pcr] = ind2sub(size(MSFE_pcr), idx_pcr);
optimal_window_pcr = rol_window_pcr(row_pcr);
optimal_lag_pcr = lags_pcr(col_pcr);

% AR-PCR Model
[min_MSFE_arpcr, idx_arpcr] = min(MSFE_arpcr(:));
[row_arpcr, col_arpcr] = ind2sub(size(MSFE_arpcr), idx_arpcr);
optimal_window_arpcr = rol_window_arpcr(row_arpcr);
optimal_lag_arpcr = lags_arpcr(col_arpcr);

% FARM Model
optimal_window_farm = rol_window_farm;
optimal_lag_farm = p;  % Fixed at p=5 in original code
min_MSFE_farm = MSFE_FARM(1);

% --- Create Summary Table ---
model_names = {'AR'; 'VAR'; 'VAR-LASSO'; 'PCR'; 'AR-PCR'; 'FARM'};
optimal_windows = [optimal_window_ar; optimal_window_var; optimal_window_var_lasso; 
                   optimal_window_pcr; optimal_window_arpcr; optimal_window_farm];
optimal_lags = [optimal_lag_ar; optimal_lag_var; optimal_lag_var_lasso; 
                optimal_lag_pcr; optimal_lag_arpcr; optimal_lag_farm];
optimal_MSFE = [min_MSFE_ar; min_MSFE_var; min_MSFE_var_lasso; 
                min_MSFE_pcr; min_MSFE_arpcr; min_MSFE_farm];

% Create table
Summary_table = table(model_names, optimal_windows, optimal_lags, optimal_MSFE, ...
    'VariableNames', {'Model', 'Optimal_Window', 'Optimal_Lags', 'MSFE'});

% Display table
disp(Summary_table);

% --- Find best overall model ---
[best_MSFE, best_idx] = min(optimal_MSFE);
best_model = model_names{best_idx};

fprintf('\n');
fprintf('========================================================================\n');
fprintf('                         BEST PERFORMING MODEL                          \n');
fprintf('========================================================================\n');
fprintf('Model:          %s\n', best_model);
fprintf('Window:         %d\n', optimal_windows(best_idx));
fprintf('Lags:           %d\n', optimal_lags(best_idx));
fprintf('MSFE:           %.6f\n', best_MSFE);
fprintf('========================================================================\n');

% --- Compute relative performance (MSFE ratios relative to best model) ---
MSFE_ratios = optimal_MSFE / best_MSFE;

fprintf('\n');
fprintf('========================================================================\n');
fprintf('              RELATIVE PERFORMANCE (Ratio to Best Model)               \n');
fprintf('========================================================================\n');
for i = 1:length(model_names)
    fprintf('%-12s: %.4f', model_names{i}, MSFE_ratios(i));
    if i == best_idx
        fprintf('  ← BEST\n');
    else
        fprintf('  (%.2f%% worse)\n', (MSFE_ratios(i) - 1) * 100);
    end
end
fprintf('========================================================================\n');

% --- Save summary table ---
savepath = fullfile(fl_pt, 'Tables'); 
writetable(Summary_table, fullfile(savepath, 'Model_Summary_Optimal.txt'), 'Delimiter', '\t');

% Add relative performance column
Summary_table.MSFE_Ratio = MSFE_ratios;
Summary_table.Percent_Worse = (MSFE_ratios - 1) * 100;

writetable(Summary_table, fullfile(savepath, 'Model_Summary_Complete.txt'), 'Delimiter', '\t');

fprintf('\nSummary tables saved:\n');
fprintf('  • Model_Summary_Optimal.txt\n');
fprintf('  • Model_Summary_Complete.txt\n\n');

% --- Visualization: Bar chart of MSFE by model ---
figure('Name', 'Model Comparison', 'Position', [200 200 900 500]);

% Subplot 1: MSFE values
subplot(1,2,1);
bar_handles = bar(optimal_MSFE, 'FaceColor', 'flat');
bar_handles.CData = repmat([0.3 0.5 0.8], length(model_names), 1);
bar_handles.CData(best_idx, :) = [0.2 0.7 0.3];  % Highlight best model in green
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
ylabel('MSFE', 'FontSize', 12);
title('Mean Squared Forecast Error by Model', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

% Add value labels on bars
for i = 1:length(optimal_MSFE)
    text(i, optimal_MSFE(i), sprintf('%.4f', optimal_MSFE(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9, 'FontWeight', 'bold');
end

% Subplot 2: Relative performance
subplot(1,2,2);
bar_handles2 = bar(MSFE_ratios, 'FaceColor', 'flat');
bar_handles2.CData = repmat([0.8 0.4 0.4], length(model_names), 1);
bar_handles2.CData(best_idx, :) = [0.2 0.7 0.3];  % Highlight best model
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
ylabel('MSFE Ratio (relative to best)', 'FontSize', 12);
title('Relative Model Performance', 'FontSize', 13, 'FontWeight', 'bold');
yline(1, '--k', 'LineWidth', 1.5, 'Label', 'Best Model', 'LabelHorizontalAlignment', 'left');
grid on;
ylim([0.95 max(MSFE_ratios)*1.05]);

% Add percentage labels
for i = 1:length(MSFE_ratios)
    if i == best_idx
        label_text = 'BEST';
    else
        label_text = sprintf('+%.1f%%', (MSFE_ratios(i) - 1) * 100);
    end
    text(i, MSFE_ratios(i), label_text, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9, 'FontWeight', 'bold');
end

sgtitle('Comprehensive Model Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% --- Additional: Parameter comparison table ---
fprintf('\n');
fprintf('========================================================================\n');
fprintf('                    OPTIMAL PARAMETER DETAILS                           \n');
fprintf('========================================================================\n');
fprintf('%-12s | Window | Lags | MSFE      | Rank\n', 'Model');
fprintf('------------------------------------------------------------------------\n');

[~, rank_order] = sort(optimal_MSFE);
for rank = 1:length(model_names)
    idx = rank_order(rank);
    fprintf('%-12s | %6d | %4d | %.6f | %d\n', ...
        model_names{idx}, optimal_windows(idx), optimal_lags(idx), ...
        optimal_MSFE(idx), rank);
end
fprintf('========================================================================\n');

% --- AIC/BIC comparison where available ---
fprintf('\n');
fprintf('========================================================================\n');
fprintf('            INFORMATION CRITERIA (where available)                      \n');
fprintf('========================================================================\n');

% AR Model
[~, idx_aic_ar] = min(AIC_mean_ar);
[~, idx_bic_ar] = min(BIC_mean_ar);
fprintf('AR Model:\n');
fprintf('  AIC suggests p = %d (AIC = %.4f)\n', lags_ar(idx_aic_ar), AIC_mean_ar(idx_aic_ar));
fprintf('  BIC suggests p = %d (BIC = %.4f)\n', lags_ar(idx_bic_ar), BIC_mean_ar(idx_bic_ar));
fprintf('  MSFE optimal: p = %d\n\n', optimal_lag_ar);

% VAR Model
[~, idx_aic_var] = min(AIC_mean_var);
[~, idx_bic_var] = min(BIC_mean_var);
fprintf('VAR Model:\n');
fprintf('  AIC suggests p = %d (AIC = %.4f)\n', lags_var(idx_aic_var), AIC_mean_var(idx_aic_var));
fprintf('  BIC suggests p = %d (BIC = %.4f)\n', lags_var(idx_bic_var), BIC_mean_var(idx_bic_var));
fprintf('  MSFE optimal: p = %d\n\n', optimal_lag_var);

fprintf('========================================================================\n');





%% -------------------------------------------------------------------- %%
%                FUNCTIONS: Functions for the code                       %
%  --------------------------------------------------------------------  %

% Function 1] AR model
function [phi, sigma2, AIC_val, BIC_val] = ar_ols(y, L)
    % Build lag matrix
    T = length(y);
    Ylag = lagmatrix(y, 1:L);
    Ylag = Ylag(L+1:end, :);
    y_t = y(L+1:end);
    
    % Estimate AR(L) by OLS
    phi = (Ylag' * Ylag) \ (Ylag' * y_t);
    res = y_t - Ylag * phi;
    
    % Variance of residuals
    T_eff = length(res);  % effective sample size
    sigma2 = (res' * res) / T_eff;  % o usa var(res, 1)
    
    % Log-likelihood (Gaussian)
    logL = -T_eff/2 * log(2*pi) - T_eff/2 * log(sigma2) - T_eff/2;
    
    % Information criteria
    k = L;  % number of parameters
    AIC_val = -2*logL + 2*k;
    BIC_val = -2*logL + log(T_eff) * k;
end


rng(123);
y_test = cumsum(randn(200,1));  % random walk
[phi, sigma2, AIC, BIC] = ar_ols(y_test, 2);

fprintf('phi: [%.4f, %.4f]\n', phi(1), phi(2));
fprintf('sigma2: %.4f\n', sigma2);
fprintf('AIC: %.2f, BIC: %.2f\n', AIC, BIC);




% Function 2] VAR
function [PI_hat, BIC, AIC, Forecast_1] = Var(Y, Y_f, p)
    % Y: T x N time series matrix
    % Y_f: last p observations stacked as column vector (N*p x 1)
    % p: lag order

    [T, N] = size(Y);

    W = Y(p+1:T, :);                      
    X = [];
    for i = 1:p
        X = [X, Y(p+1-i:T-i, :)];         
    end

    X = [ones(T-p, 1), X];

    PI_hat = (X' * X) \ (X' * W);         
    PI_hat = PI_hat';                    

    % Compute AIC and BIC
    E = W - X * (PI_hat');               
    Sigma = (E' * E) / (T - p - N*p - 1);
    k = N * (N*p + 1);
    logL = -(T - p) * N / 2 * (log(2*pi) + 1) - (T - p)/2 * log(det(Sigma));
    AIC = -2*logL / (T - p) + 2*k / (T - p);
    BIC = -2*logL / (T - p) + k*log(T - p) / (T - p);


    X_f = [1; Y_f(:)];                   
    Forecast_1 = (PI_hat * X_f)';        
end




% Function 3] Sparse Regression
function [beta, forecast_1, bic, aic] = Sparse_regression(Y, Y_for, p)
    [T, ~] = size(Y);
    
    X = lagmatrix(Y, 1:p);
    X = X(p+1:end, :);
    y = Y(p+1:end);
    
    [B, FitInfo] = lasso(X, y, 'CV', 10);
    idx = FitInfo.IndexMinMSE;
    beta_lags = B(:, idx);
    intercept = FitInfo.Intercept(idx);
    
    beta = [intercept; beta_lags];
    
    if nargin > 1 && ~isempty(Y_for)
        X_for = Y_for(end:-1:end-p+1);
    else
        X_for = Y(end:-1:end-p+1);
    end
    
    forecast_1 = intercept + X_for' * beta_lags;
    
    y_pred = FitInfo.Intercept(idx) + X * beta_lags;
    res = y - y_pred;
    T_eff = length(res);
    sigma2 = (res' * res) / T_eff;
    
    k = sum(beta_lags ~= 0);
    if intercept ~= 0
        k = k + 1;
    end
    
    logL = -T_eff/2 * log(2*pi) - T_eff/2 * log(sigma2) - T_eff/2;
    
    aic = -2*logL + 2*k;
    bic = -2*logL + log(T_eff)*k;
end


% ABC_Criteria - Determines the Number of Common Factors in the Static
% Approximate Factor Model
% 
%  This function implement the information criteria described in Alessi,
%  Barigozzi and Capasso (2010) 
% 
% [rhat1 rhat2]=ABC_crit(X, kmax, nbck, cmax)
% 
% Inputs:
% 	X - (T x n) stationary data matrix
%   kmax - maximum number of factors
%   nbck - number of sublocks to be used (default floor(n/10))
%   cmax - maximum value for the penalty constant (default = 3) 
%   graph - if set to 1 show graphs as in the paper (default = 0)
% Outputs:
%  rhat1 - determines the number of shocks using a large window
%  rhat2 - determines the number of shocks using a small window
% 
% Written by Matteo Barigozzi 
% 
% Reference: Alessi, L., M. Barigozzi, and M. Capasso (2010). 
% Improved penalization for determining the number of factors 
% in approximate static factor models. 
% Statistics and Probability Letters 80, 1806?1813.

function [rhat1 rhat2] = ABC_crit(X, kmax, nbck, cmax, graph)

npace=1; 
step=500; 

[T,n] = size(X);                                                            % Size of the datatset

if nargin < 2
    disp(sprintf('Too few input arguments, must provide a value for kmax'))
elseif nargin == 2
    nbck = floor(n/10);
    cmax = 3;
    graph = 0;
elseif nargin == 3
    cmax = 3;
    graph = 0;
elseif nargin == 4
    graph = 0;
end

x = (X - ones(T,1)*mean(X))./(ones(T,1)*std(X,1));
    
% Running the Criteria %%
s=0;
for N = n-nbck:npace:n
    s=s+1; 
    [~, Ns]=sort(rand(n,1));
    xs = x(1:T,Ns(1:N));
    xs = (xs - ones(T,1)*mean(xs))./(ones(T,1)*std(xs,1));
    eigv = flipud(eig(cov(xs)));                                            % Eigenvalues of the Covariance Matrix
    
    for k=1:kmax+1
        IC1(k,1) = sum(eigv(k:N)); 
    end
    
    p = ((N+T)/(N*T))*log((N*T)/(N+T));                                     % penalty
    
    T0=repmat((0:kmax)',1).*p;
    for c = 1:floor(cmax*step);            
        cc = c/step;
        IC = (IC1./N) + T0*cc;   [~, rr]=min(IC);                           % criterion            
        abc(s,c)=rr-1;                                  
    end
end    

%%% ----------------------------------------- %%%
%%% Select Automatically the Number of Shocks %%%
%%% ----------------------------------------- %%%

cr=(1:floor(cmax*500))'/500;

for ll=1:2; 
    ABC(1,1)=kmax;
    ABC(1,2:3)=0;
end
        
sabc = std(abc);
c1=2;
for ii=1:size(cr,1);
    if sabc(1,ii)==0;                                                       % If the number of factors is always the same across sub-blocks
        if abc(end,ii)==ABC(c1-1,1);
            ABC(c1-1,3)=cr(ii);
        else
            ABC(c1,1)=abc(end,ii);
            ABC(c1,2:3)=cr(ii);
            c1=c1+1;
        end;
    end;
end;
ABC(:,4) = ABC(:,3)-ABC(:,2);                                               % Computes Window Size
q = ABC(find(ABC(2:end,4)>.05)+1,1);                                        % Number of Factors with Large Window
rhat1 = q(1);                                                               
q = ABC(find(ABC(2:end,4)>.01)+1,1);                                        % Number of Shocks with Small Window
rhat2 = q(1);                                              

if graph == 1
    set(0,'DefaultLineLineWidth',1.5);
    figure
    plot(cr,abc(end,:),'r-')
    axis tight
    hold all
    plot(cr,5*sabc,'b--')
    xlabel('c')
    axis tight
    xlim([0 cmax])
    legend('r^{*T}_{c;N}','S_c')
    title('ABC estimated number of factors')
    grid on
end

end









%%
save("/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Machine Learning for Economists/Project/Data/workspace_code.mat")