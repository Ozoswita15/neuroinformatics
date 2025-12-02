%% --- Cluster-based Permutation Test (with Multiple Comparisons Correction) ---
% This code compares two conditions (A and B) using a non-parametric
% cluster-based permutation test, correcting for multiple comparisons.

% Input: 
%   dataA, dataB: [subjects x time] (or [subjects x time x channels])
% Output:
%   cluster_pvals : p-values for significant clusters
%   stat_obs      : observed t-map
%   sig_mask      : mask of significant clusters (1 = significant)

clear; clc;

% -----------------------------
% Example simulated data
% -----------------------------
nSubs = 20;
nTime = 500;  % samples (e.g. -200 to 800 ms)
dataA = randn(nSubs, nTime);      % Condition A
dataB = randn(nSubs, nTime);      % Condition B
dataB(:, 200:250) = dataB(:, 200:250) + 0.8;  % true effect window

nPerms = 1000;
alpha_cluster = 0.05; % pre-cluster threshold
alpha_sig = 0.05;     % cluster-level significance threshold

% -----------------------------
% Step 1: Compute observed t-map
% -----------------------------
[~,~,~,stats] = ttest(dataA, dataB);
t_obs = squeeze(stats.tstat);

% -----------------------------
% Step 2: Threshold to form clusters
% -----------------------------
tcrit = tinv(1 - alpha_cluster/2, nSubs - 1);
cluster_mask = abs(t_obs) > tcrit;

% Identify clusters (using bwconncomp)
CC = bwconncomp(cluster_mask);
cluster_sums_obs = zeros(1, length(CC.PixelIdxList));
for i = 1:length(CC.PixelIdxList)
    cluster_sums_obs(i) = sum(abs(t_obs(CC.PixelIdxList{i})));
end

% -----------------------------
% Step 3: Permutation loop to build null distribution
% -----------------------------
max_cluster_sums = zeros(1, nPerms);
disp('Running permutations...');
for p = 1:nPerms
    flip_mask = rand(nSubs, 1) > 0.5;
    permA = dataA; permB = dataB;
    permA(flip_mask, :) = dataB(flip_mask, :);
    permB(flip_mask, :) = dataA(flip_mask, :);

    [~,~,~,stats_perm] = ttest(permA, permB);
    t_perm = squeeze(stats_perm.tstat);

    % Pre-cluster threshold
    cluster_mask_perm = abs(t_perm) > tcrit;
    CCp = bwconncomp(cluster_mask_perm);

    if CCp.NumObjects > 0
        cluster_sums_perm = zeros(1, CCp.NumObjects);
        for j = 1:CCp.NumObjects
            cluster_sums_perm(j) = sum(abs(t_perm(CCp.PixelIdxList{j})));
        end
        max_cluster_sums(p) = max(cluster_sums_perm);
    else
        max_cluster_sums(p) = 0;
    end
end

% -----------------------------
% Step 4: Determine corrected threshold
% -----------------------------
thr = prctile(max_cluster_sums, 100 * (1 - alpha_sig));

% -----------------------------
% Step 5: Keep only clusters above threshold
% -----------------------------
sig_mask = zeros(size(t_obs));
cluster_pvals = nan(1, length(CC.PixelIdxList));

for i = 1:length(CC.PixelIdxList)
    cluster_sum = cluster_sums_obs(i);
    pval = mean(max_cluster_sums >= cluster_sum);
    cluster_pvals(i) = pval;
    if cluster_sum > thr
        sig_mask(CC.PixelIdxList{i}) = 1;
    end
end

% -----------------------------
% Step 6: Visualization
% -----------------------------
figure('Color','w');
subplot(2,1,1);
plot(t_obs, 'k'); hold on;
plot(find(sig_mask), t_obs(sig_mask), 'r', 'LineWidth', 2);
xlabel('Time points'); ylabel('t-values');
title('Observed t-map (red = cluster-corrected significant)');
box off;

subplot(2,1,2);
histogram(max_cluster_sums,50); hold on;
xline(thr,'r','LineWidth',2);
title('Null distribution of max cluster sums');
xlabel('Cluster sum'); ylabel('Count');
legend({'Null','95th percentile'},'box','off');

disp('--- Cluster-based permutation test complete ---');
disp(['Significant clusters: ' num2str(sum(cluster_pvals < alpha_sig))]);
