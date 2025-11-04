clc;
clear;

load sampleEEGdata.mat

% Input parameters
chan2use = 'fcz';
min_freq = 3;
max_freq = 30;
num_freq = 20;

%To Do: TFR with morlet wavelet number 6 in freqs * time * trials matrix in
%power_all variable

waveletnum = 6;

% Find channel index
chanIdx = find(strcmpi({EEG.chanlocs.labels}, chan2use));
if isempty(chanIdx)
    error(['Channel ' chan2use ' not found in EEG.chanlocs']);
end

% Get data: time x trials
data = squeeze(EEG.data(chanIdx, :, :));  % (time x trials)
fs = EEG.srate;                           % sampling frequency
timeVec = EEG.times / 1000;               % convert ms → s
cfreqs = linspace(min_freq, max_freq, num_freq); % center frequencies

%% Perform Morlet wavelet transform for each trial
nTrials = size(data, 2);
power_all = zeros(num_freq, size(data, 1), nTrials);

for tr = 1:nTrials
    x = data(:, tr);
    [cfx, cfreqso] = morletWaveletTransform(x, fs, cfreqs, waveletnum, 1);
    power_all(:, :, tr) = abs(cfx).^2;  % power = |complex coefficients|^2
end

%% Average across trials for visualization
power_avg = mean(power_all, 3);

%% Plot Time-Frequency Power Map
figure('Color','w');
imagesc(timeVec, cfreqso, power_avg);
set(gca, 'YDir', 'normal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(sprintf('TFR (%s) – Morlet wavelet #%d', upper(chan2use), waveletnum));
colorbar;

%Edge trimming
time_s = dsearchn(EEG.times', -500);
time_e = dsearchn(EEG.times', 1200);
eegpower = power_all(:, time_s:time_e, :);
tftimes = EEG.times(time_s:time_e);
nTimepoints = length(tftimes);

%Plot
%% Parameters
voxel_pval = 0.01;
cluster_pval = 0.05;
n_permutes = 2000;

baseidx = [dsearchn(tftimes', -500), dsearchn(tftimes', -100)]; %baseline range

%% Trial-level baseline normalization
% Initialize output
normed_power = zeros(size(eegpower));  % same size: [freq x time x trials]

% Loop through trials
for tr = 1:size(eegpower, 3)
    trialPower = eegpower(:, :, tr);  % [freq x time]
    basePower = mean(trialPower(:, baseidx(1):baseidx(2)), 2); % mean over baseline timepoints
    normed_power(:, :, tr) = 10 * log10(bsxfun(@rdivide, trialPower, basePower)); % dB normalization
end

% Average across trials
realmean = mean(normed_power, 3);  % [freq x time]

%% Plot the baseline-normalized average TFR
figure('Color','w');
contourf(tftimes, cfreqso, realmean, 20, 'LineColor', 'none'); % 20 levels for smooth contours
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('Baseline-normalized TFR (%s, %.1f–%.1f Hz)', upper(chan2use), min_freq, max_freq));
colorbar;


% To Do: shuffle data i.e. destroy time-locking to the stimuli to obtain 1000
% equivalents of realmean and store in permuted_vals

%% Create null distribution by time-shuffling 

n_permutes = 1000;  % number of permutations
[nFreqs, nTimes, nTrials] = size(normed_power);

permuted_vals = zeros(nFreqs, nTimes, n_permutes);

for perm_i = 1:n_permutes
    shuffled_trials = zeros(nFreqs, nTimes, nTrials);
    
    % Shuffle each trial's time points independently
    for tr = 1:nTrials
        shift_amt = randi(nTimes);  % random circular shift
        shuffled_trials(:,:,tr) = circshift(normed_power(:,:,tr), [0 shift_amt]);
    end
    
    % Average across trials after shuffling
    permuted_vals(:,:,perm_i) = mean(shuffled_trials, 3);
end

% To Do: Create a z-score metric 
% that stores how deviant the observed realmean is w.r.t permuted_vals and 
% Name it zmap

%% Compute z-score map relative to null distribution
% Compute mean and std across permutations for each voxel
perm_mean = mean(permuted_vals, 3);
perm_std  = std(permuted_vals, 0, 3);

% Avoid divide-by-zero errors
perm_std(perm_std == 0) = eps;

% Compute z-map: how deviant the observed map is from the permutation null
zmap = (realmean - perm_mean) ./ perm_std;

%% Visualize the z-map
figure('Color','w');
contourf(tftimes, cfreqso, zmap, 20, 'LineColor', 'none'); % 20 levels for smooth contours
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('Z-score map of observed power vs. shuffled null');
colorbar;

% To Do: Calculate those bins where z-score threshold exceeds p val say 0.05 and
% store results in threshmean

%% Threshold z-map based on p-value criterion
p_threshold = 0.05;             % desired significance level
two_tailed = true;              % set false for one-tailed test

if two_tailed
    z_thresh = norminv(1 - p_threshold/2);
else
    z_thresh = norminv(1 - p_threshold);
end

% Initialize thresholded map
threshmean = zeros(size(zmap));

% Apply threshold (keep only significant voxels)
threshmean(abs(zmap) >= z_thresh) = zmap(abs(zmap) >= z_thresh);

%% Visualize thresholded z-map
figure('Color','w');
contourf(tftimes, cfreqso, threshmean, 20, 'LineColor', 'none'); % 20 levels for smooth contours
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('Thresholded Z-map (p < %.2f, |z| > %.2f)', p_threshold, z_thresh));
colorbar;



%% morlet wavelet transform function copied:
function [cfx,cfreqso] = morletWaveletTransform(x, fs, cfreqs, morletParam, dim, plotFlag)
% Takes the complex Morlet wavelet transform of time series data and plots
% spectrogram
% 
% ARGUMENTS:
%       x -- 1xT vector time series or multi-dimensional vector with time
%           specified in dimension DIM
%       fs -- Sampling frequency (in Hz)
%       cfreqs -- Vector of centre frequencies
%       morletParam -- (Optional) Morlet wavelet parameter, allows trade
%           between time and frequency resolution (higher is better
%           frequency resolution). Default value 7
%       dim -- (Optional) Specify the dimension of time in X
%       plotFlag -- (Optional) morletWaveletTransform(..., 'plot') means
%          function will create time-frequency plots, with or without
%          morletParam specified. Plots will be averaged across channels.
%          Default is no plotting
%
% OUTPUTS:
%       cfs -- FxT matrix of complex Morlet wavelet coefficients, where F
%           is the number of centre frequencies. If X is multi-dimensional,
%           CFS will be Fx(size(X))
%       cfreqso -- Vector of centre frequencies. Should always be equal to
%           input argument cfreqs, but this can be used to check that CWTFT
%           is not changing the input frequencies.
%
% USAGE:
%{
        fs = 1000;
        t = 0:(1/fs):2;
        x = chirp(t,3,1,8,'quadratic');
        cfreqs = linspace(1, 10, 100);
        % Use built-in spectrogram plotting with default Morlet parameter
        morletWaveletTransform(x, fs, cfreqs, 'plot');
        % Manually plot spectrogram with smaller parameter
        [cfx, cfreqs] = morletWaveletTransform(x, fs, cfreqs, 3);
        figure
        imagesc(t, cfreqs, abs(cfx))
        axis xy
%}
%
% Rory Townsend, Oct 2017
% rory.townsend@sydney.edu.au


% Set up paramters for morlet wavelet transform
if exist('morletParam', 'var') && strcmp(morletParam, 'plot')
    plotFlag = 'plot';
end
if ~exist('morletParam', 'var') || strcmp(morletParam, 'plot')
    morletParam = 7;
end
if ~isvector(x) && (~exist('dim', 'var') || strcmp(morletParam, 'plot'))
    dim = 2;
end

if ~isvector(x)
    % Reshape input so that time is in the second dimension and other 
    % dimensions are combined
    permOrder = [dim, 1:dim-1, dim+1:ndims(x)];
    x = permute(x, permOrder);
    sx = size(x);
    x = x(:,:);
end

dt = 1/fs;
morletFourierFactor = 4*pi/(morletParam+sqrt(2+morletParam^2));

% Set up a larger output matrix if X has multiple channels / trials
if ~isvector(x)
    cfx = zeros([length(cfreqs), size(x)]);
end

% Old code using CWT rather than CWTFT
% wname = 'morl';
% scales = centfrq(wname)./(cfreqs*dt);

% Set up structure defining scales between min and max pseudo-frequencies
scales = 1./(morletFourierFactor * cfreqs);

% Calculate wavelet coefficients for each channel
if ~isvector(x)
    for ichan = 1:size(x,2)
        %icfs = cwt(x(ichan,:), scales, wname);
        cfstruct = cwtft({x(:,ichan),dt},'scales',scales,'wavelet',{'morl', morletParam});
        cfx(:, :, ichan) = cfstruct.cfs;
    end
    plotVal = squeeze(mean(cfx, 3));
else
    %cfs = cwt(x, scales, wname);
    cfstruct = cwtft({x,dt},'scales',scales,'wavelet',{'morl',morletParam});
    cfx = cfstruct.cfs;
    plotVal = cfx;
end

sc = cfstruct.scales;
cfreqso = 1./(sc*morletFourierFactor);

% Plot data 
if exist('plotFlag', 'var') && (strcmp(plotFlag, 'plot'))
    figure
    % Generate time-frequency power plot
    time = (1:length(x))/fs;
    imagesc(time, cfreqso, abs(plotVal))
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Morlet wavelet power')
    axis xy
    
    % Generate time-frequency phase plot
    figure
    imagesc(time, cfreqso, angle(plotVal))
    % Note: hsv is MATLAB's only built-in colormap suitable for circular
    % data, but it is very ugly
    if exist('pmkmp_new', 'file') == 2
        colormap(pmkmp_new('ostwald_o', 256));
    else
        colormap(hsv)
    end
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    title('Morlet wavelet phase')
    axis xy
end

% Reshape output to input size
if ~isvector(x)
    cfx = reshape(cfx, [length(cfreqs), sx]);
    cfx = ipermute(cfx, [1, 1+permOrder]);
end

end
