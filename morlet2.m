%% Time-Frequency Analysis using Morlet Wavelets 
% Frequencies to analyze (logarithmically spaced)
min_freq = 2;
max_freq = 13;
num_freqs = 30;
frex = logspace(log10(min_freq), log10(max_freq), num_freqs);
% Number of wavelet cycles 
n_cycles = logspace(log10(3), log10(10), num_freqs);
% Time vector and dimensions
[num_chans, num_time_points, min_epochs] = size(epochs_remembered);
time_points = (0:num_time_points - 1) / new_srate;
% Baseline Period 
basel_start_t = -0.2; 
basel_end_t = -0.1; 

% CHANNELS:
chan_names_of_interest = {'F3', 'P3', 'FZ', 'Pz'};
chan_indices = find(ismember(chan_labels, chan_names_of_interest));
if isempty(chan_indices)
    warning('Specified channels not found. Using the first 4 channels.');
    chan_indices = 1:min(4, num_chans);
    chan_names_of_interest = chan_labels(chan_indices);
end
fprintf(' Analyzing Time-Frequency Power for channels: %s\n', strjoin(chan_names_of_interest, ', '));
% edge artifact
s_min_freq = n_cycles(1) / (2 * pi * frex(1));
samples_to_cut = ceil(3 * s_min_freq * new_srate); 

valid_time_indices = (samples_to_cut + 1) : (num_time_points - samples_to_cut);
valid_time_points = time_points(valid_time_indices);

% --- 2. Time-Frequency Calculation

conditions_epochs = {epochs_remembered, epochs_recognised, epochs_not_recognised};
condition_names = {'Remembered', 'Recognised', 'Not Recognised'};
all_tf_power = zeros(num_freqs, num_time_points, length(conditions_epochs));
all_ersp = cell(1, length(conditions_epochs)); 
n_chans_filtered = length(chan_indices);

% baseline indices
[~, basel_start_idx_valid] = min(abs(valid_time_points - basel_start_t));
[~, basel_end_idx_valid] = min(abs(valid_time_points - basel_end_t));

for cond_idx = 1:length(conditions_epochs)
    fprintf(' Calculating power for Condition: %s\n', condition_names{cond_idx});
    current_epochs = conditions_epochs{cond_idx}(chan_indices, :, :);
    [~, ~, n_trials_filtered] = size(current_epochs);
    % Pre-allocate (freqs x time x channels x trials)
    cond_tf_power_linear = zeros(num_freqs, num_time_points, n_chans_filtered, n_trials_filtered);
    % Pre-allocate for per-trial ERSP
    cond_ersp = zeros(num_freqs, length(valid_time_indices), n_trials_filtered);
    
    for trial_idx = 1:n_trials_filtered
        for chan_idx = 1:n_chans_filtered
            data_to_analyze = squeeze(current_epochs(chan_idx, :, trial_idx));
            for fi = 1:num_freqs
                s = n_cycles(fi) / (2 * pi * frex(fi));
                t_wavelet = -3*s*new_srate:1/new_srate:3*s*new_srate;
                morlet_wavelet = exp(2*1i*pi*frex(fi)*t_wavelet) .* exp(-t_wavelet.^2./(2*s^2));
                % CONVOLUTION (using FFT)
                n_data = length(data_to_analyze);
                n_kernel = length(morlet_wavelet);
                n_conv = n_data + n_kernel - 1;
                convolution_result = ifft(fft(data_to_analyze, n_conv) .* fft(morlet_wavelet, n_conv));
                % TRIMMING to original data length
                convolution_result = convolution_result(floor(n_kernel/2)+1 : floor(n_kernel/2)+n_data);
                cond_tf_power_linear(fi, :, chan_idx, trial_idx) = abs(convolution_result).^2; % Power
            end
        end
        % per-trial ERSP
        trial_power_linear = mean(cond_tf_power_linear(:, :, :, trial_idx), 3); % (freqs x times x 1)
        trial_power_db = 10 * log10(trial_power_linear);
        trial_power_db_valid = trial_power_db(:, valid_time_indices);
        baseline_mean = mean(trial_power_db_valid(:, basel_start_idx_valid:basel_end_idx_valid), 2);
        cond_ersp(:, :, trial_idx) = bsxfun(@minus, trial_power_db_valid, baseline_mean);
    end
   
    all_ersp{cond_idx} = cond_ersp;
    % Average power across trials and channels
    all_tf_power(:, :, cond_idx) = mean(mean(cond_tf_power_linear, 4), 3);
end
fprintf('Time-Frequency calculations and per-trial ERSP computation complete.\n');


% --- 3. Baseline Correction and Plotting 

figure('Name', 'Morlet Wavelet Time-Frequency Power Comparison (ERSP)', 'Position', [100 100 1200 800]);

clim_range = [-1 1];
for cond_idx = 1:length(conditions_epochs)
    subplot(length(conditions_epochs), 1, cond_idx);
    current_tf_power = all_tf_power(:, :, cond_idx);
    
    power_db = 10 * log10(current_tf_power);
    
    current_power_db_valid = power_db(:, valid_time_indices);
    % Define Baseline Indices within the *VALID* time vector
    [~, basel_start_idx_valid] = min(abs(valid_time_points - basel_start_t));
    [~, basel_end_idx_valid] = min(abs(valid_time_points - basel_end_t));
    % Compute mean power during baseline for each frequency
    baseline_power_mean = mean(current_power_db_valid(:, basel_start_idx_valid:basel_end_idx_valid), 2);
    %  Subtract baseline to get ERSP
    tf_power_baseline_corrected = bsxfun(@minus, current_power_db_valid, baseline_power_mean);
    % Plotting 
    imagesc(valid_time_points, frex, tf_power_baseline_corrected);
    set(gca, 'YDir', 'normal');
    caxis(clim_range); 
    colorbar;
    title(sprintf('%s ERSP (%s Avg.)', condition_names{cond_idx}, strjoin(chan_names_of_interest, '/')));
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    line([0 0], get(gca, 'YLim'), 'Color', 'k', 'LineWidth', 1.5, 'LineStyle', '--');
    ylim([min_freq max_freq]);
end
sgtitle('Morlet Wavelet Time-Frequency Power (Event-Related Spectral Perturbation)');

% Save the Time-Frequency plot
tf_plot_file = fullfile(pwd, 'morlet_ersp_comparison_fixed.png');
saveas(gcf, tf_plot_file);
close(gcf);
fprintf('Fixed Morlet wavelet ERSP plots saved. If still flat, re-verify t=0 timing.\n');


%Plot Normality Distribution Curves 

if ~exist('all_ersp', 'var') || isempty(all_ersp)
    error('all_ersp not found or empty. Ensure per-trial ERSP is computed in the time-frequency loop.');
end

% Verify trial counts across conditions
n_trials = zeros(1, length(condition_names));
for cond_idx = 1:length(condition_names)
    if isempty(all_ersp{cond_idx})
        error('all_ersp{%d} (%s) is empty. Check data for condition.', cond_idx, condition_names{cond_idx});
    end
    [n_f, n_t, n_trials(cond_idx)] = size(all_ersp{cond_idx});
    fprintf('Condition %s: %d trials\n', condition_names{cond_idx}, n_trials(cond_idx));
end
if ~isequal(n_trials, n_trials(1) * ones(1, length(condition_names)))
    warning('Trial counts differ across conditions: %s. Proceeding with available trials.', mat2str(n_trials));
end

% Average over a frequency/time range 
freq_range = frex >= 4 & frex <= 12; % Alpha/theta band
time_range = valid_time_points >= 0 & valid_time_points <= 1; % 0 to 1 s

if ~any(freq_range) || ~any(time_range)
    error('Invalid frequency or time range. Check frex (%d elements) and valid_time_points (%d elements).', ...
          length(frex), length(valid_time_points));
end

% Extract ERSP data averaged over the selected frequency and time range
data_conditions = cell(1, length(condition_names));
for cond_idx = 1:length(condition_names)
    % Average over frequency and time dimensions
    data_subset = all_ersp{cond_idx}(freq_range, time_range, :);
    if isempty(data_subset)
        error('Data subset for condition %s is empty. Check freq_range and time_range.', condition_names{cond_idx});
    end
    data_conditions{cond_idx} = squeeze(mean(data_subset, [1, 2])); % (trials x 1)
    % Verify output is a vector
    if ~isvector(data_conditions{cond_idx})
        error('Averaged data for condition %s is not a vector. Size: %s', ...
              condition_names{cond_idx}, mat2str(size(data_conditions{cond_idx})));
    end
end

% Compute mean and standard deviation for each condition
means = zeros(1, length(condition_names));
stds = zeros(1, length(condition_names));
for cond_idx = 1:length(condition_names)
    if isempty(data_conditions{cond_idx})
        error('data_conditions{%d} (%s) is empty.', cond_idx, condition_names{cond_idx});
    end
    means(cond_idx) = mean(data_conditions{cond_idx});
    stds(cond_idx) = std(data_conditions{cond_idx});
    % Check for zero standard deviation 
    if stds(cond_idx) == 0
        warning('Standard deviation for condition %s is zero. Setting to small value to avoid normpdf error.', ...
                condition_names{cond_idx});
        stds(cond_idx) = eps; % Small non-zero value
    end
end

% Define x-axis range for plotting 
all_data = vertcat(data_conditions{:});
if isempty(all_data)
    error('No data available in data_conditions. Check all_ersp contents.');
end
x_range = linspace(min(all_data) - 1, max(all_data) + 1, 100);

% Compute normal distribution PDFs
pdfs = zeros(length(x_range), length(condition_names));
for cond_idx = 1:length(condition_names)
    pdfs(:, cond_idx) = normpdf(x_range, means(cond_idx), stds(cond_idx));
end

% Plot histograms and normality curves
figure('Name', 'Normality Distribution (4-12 Hz, 0-1 s)', ...
       'Position', [100 100 800 600]);

hold on;
colors = lines(length(condition_names)); 
for cond_idx = 1:length(condition_names)
    % Plot histogram (normalized to probability density)
    histogram(data_conditions{cond_idx}, 20, 'Normalization', 'pdf', ...
              'FaceColor', colors(cond_idx, :), 'FaceAlpha', 0.3, ...
              'DisplayName', sprintf('%s (Histogram)', condition_names{cond_idx}));
    % Plot normal distribution curve
    plot(x_range, pdfs(:, cond_idx), 'Color', colors(cond_idx, :), 'LineWidth', 2, ...
         'DisplayName', sprintf('%s (Normal PDF)', condition_names{cond_idx}));
end

% Customize plot
title('Distribution of ERSP (4-12 Hz, 0-1 s)');
xlabel('ERSP (dB)');
ylabel('Probability Density');
legend('show');
grid on;
hold off;

fprintf('Normality distribution plot complete.\n');