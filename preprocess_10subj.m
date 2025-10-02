clear;
eeglab;

% Define subject list (adjust as needed)
subjects = {'sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', ...
            'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-011', ...
            }; % 10 of 27 subjects

% Initialize array to store trial counts for all subjects
all_num_trials = zeros(1, length(subjects));

% Initialize metadata storage before subject loop
rejection_metadata = table('Size', [length(subjects) 8], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double', 'cell'}, ...
    'VariableNames', {'Subject', 'TotalChannels', 'RejectedChannels', ...
                      'FinalChannels', 'TotalTrials', 'RejectedTrials', 'FinalTrials', 'GoodTrialIndices'});


for subj_idx = 1:length(subjects)
    subj_id = subjects{subj_idx};
    fprintf('Processing %s...\n', subj_id);

    % File paths
    edf_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_eeg.edf', subj_id, subj_id);
    event_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_events.tsv', subj_id, subj_id);
    output_dir = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/spectral_data', subj_id);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Load EEG data
    EEG = pop_biosig(edf_file);
    [ALLEEG, EEG, CURRENTSET] = eeg_store([], EEG, 0);

    % --- Load and insert events ---
    event_data = readtable(event_file, 'FileType', 'text', 'Delimiter', '\t');
    EEG.event = struct([]);
    for i = 1:height(event_data)
        EEG.event(i).latency = event_data.sample(i);
        EEG.event(i).type = string(event_data.trial_type(i));
        EEG.event(i).duration = event_data.duration(i) * EEG.srate;
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    
    % --- Channels ---
    chan_labels = {EEG.chanlocs.labels};
    
    % --- Spectral plots (initial inspection) ---
    figure;
    [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
    saveas(gcf, fullfile(output_dir, 'spectro_raw.png'), 'png');
    %save(fullfile(output_dir, 'spectro_raw.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    

    % --- Notch filter (50 Hz) ---
    wo = 50/(EEG.srate/2); bo = wo/35;
    [bn,an] = iirnotch(wo, bo);
    EEG.data = filtfilt(bn,an,EEG.data')';
    EEG = eeg_checkset(EEG);
    figure;
    [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
    saveas(gcf, fullfile(output_dir, 'spectro_notch.png'), 'png');
    %save(fullfile(output_dir, 'spectro_notch.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    

    % --- Epoch extraction ---
    types = {EEG.event.type};
    start_idx = find(strcmp(types, 'trialStart'));
    end_idx = find(strcmp(types, 'trialEnd'));
    pre_buffer = round(-0.2 * EEG.srate); % 200 ms pre-stimulus
    post_buffer = round(0.8 * EEG.srate); % 800 ms post-stimulus
    num_trials = min(length(start_idx), length(end_idx));
    epochs = cell(1, num_trials);
    trial_durations = zeros(1, num_trials);
    for t = 1:num_trials
        start_lat = round(EEG.event(start_idx(t)).latency) + pre_buffer;
        end_lat = round(EEG.event(end_idx(t)).latency) + post_buffer;
        if start_lat < 1 || end_lat > size(EEG.data,2) || start_lat >= end_lat
            warning('Skipping trial %d: start=%d, end=%d', t, start_lat, end_lat);
            continue;
        end
        epochs{t} = EEG.data(:, start_lat:end_lat);
        trial_durations(t) = end_lat - start_lat + 1;
        fprintf('Extracted trial %d with %d samples\n', t, trial_durations(t));
    end

    % --- Downsample ---
    new_srate = 512;
    for t = 1:num_trials
        if isempty(epochs{t}), continue; end
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN before downsampling, skipping.', t);
            continue;
        end
        epochs{t} = epochs{t}'; % Transpose to time x channels
        pad_size = 100;
        epochs{t} = [zeros(size(epochs{t},1), pad_size) epochs{t} zeros(size(epochs{t},1), pad_size)];
        epochs{t} = resample(epochs{t}, new_srate, EEG.srate);
        epochs{t} = epochs{t}(:, pad_size+1:end-pad_size); % Remove padding
        epochs{t} = epochs{t}'; % Transpose back to channels x time
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN after downsampling.', t);
            continue;
        end
        fprintf('Downsampled trial %d to %d samples\n', t, size(epochs{t}, 2));
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_downsampled.png'), 'png');
    %save(fullfile(output_dir, 'spectro_downsampled.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    
    % --- Bandpass filter (0.5-40 Hz) ---
    for t = 1:num_trials
        if isempty(epochs{t}), continue; end
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN before filtering, skipping.', t);
            continue;
        end
        epochs{t} = epochs{t}'; % Transpose to time x channels
        pad_size = 100;
        temp_data = [epochs{t}(end-pad_size+1:end, :); epochs{t}; epochs{t}(1:pad_size, :)];
        [b,a] = butter(2, [0.5 40]/(new_srate/2), 'bandpass');
        epochs{t} = filtfilt(b, a, temp_data);
        epochs{t} = epochs{t}(pad_size+1:end-pad_size, :); % Remove padding
        epochs{t} = epochs{t}'; % Transpose back to channels x time
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN after filtering.', t);
            continue;
        end
        fprintf('Filtered trial %d\n', t);
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_bandpass.png'), 'png');
    %save(fullfile(output_dir, 'spectro_bandpass.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    
    manual_bad_indices = {
    'sub-001', [20];
    'sub-002', [20, 2, 3];
    'sub-003', [57, 60];
    'sub-004', [57];
    'sub-005', [20];
    'sub-006', [31, 30];
    'sub-007', [18, 19, 27];
    'sub-008', [31];
    'sub-009', [20, 26, 35];
    'sub-011', [52, 61, 64, 31]
};
    
    % --- Channel Rejection (before CAR) ---
    % Find bad channel indices for the current subject
    bad_indices = [];
    for i = 1:size(manual_bad_indices, 1)
        if strcmp(manual_bad_indices{i, 1}, subj_id)
            bad_indices = manual_bad_indices{i, 2};
            break;
        end
    end

    % Convert epochs to 3D array for variance computation
    num_channels = size(epochs{1}, 1);

    max_samples = max(cellfun(@(x) size(x, 2), epochs));
    epochs_3d = zeros(size(epochs{1}, 1), max_samples, num_trials);
    for t = 1:num_trials
        if ~isempty(epochs{t})
            epochs_3d(:,:,t) = [epochs{t}, zeros(size(epochs{t}, 1), max_samples - size(epochs{t}, 2))];
        end
    end
    
    % Reject bad channels (flat or high variance across trials)
    chan_var = squeeze(var(epochs_3d, 0, [2 3]));
    bad_chans = (chan_var < 1e-6) | (chan_var > mean(chan_var) + 3 * std(chan_var));
    
    % add manual bad channels
    bad_indices = bad_indices(bad_indices > 0 & bad_indices <= num_channels);
    bad_chans(bad_indices) = true;

    good_chans = find(~bad_chans);
    num_bad_chans = sum(bad_chans);
    num_total_chans = length(chan_labels);
    
    % Print rejected channels
    if num_bad_chans > 0
        rejected_indices = find(bad_chans);
        rejected_labels = chan_labels(rejected_indices);
        fprintf('Rejected %d/%d channels (indices: %s, labels: %s)\n', ...
            num_bad_chans, num_total_chans, mat2str(rejected_indices), strjoin(rejected_labels, ', '));
    else
        fprintf('Rejected %d/%d channels (no channels rejected)\n', num_bad_chans, num_total_chans);
    end
    
    % Update epochs_3d with only good channels
    epochs_3d = epochs_3d(good_chans, :, :);
    chan_labels_clean = chan_labels(good_chans);
    
    % Convert back to cell array with correct trial count
    epochs = cell(1, num_trials); % Reinitialize to avoid size issues
    for t = 1:num_trials
        if ~isempty(epochs_3d(:,:,t))
            epochs{t} = epochs_3d(:,:,t);
            non_zero_idx = find(any(epochs{t} ~= 0, 1), 1, 'last');
            if ~isempty(non_zero_idx)
                epochs{t} = epochs{t}(:, 1:non_zero_idx);
            end
        end
    end
    
    % Prepare and validate data for spectrogram
    temp_data = cat(2, epochs{:});
    if size(temp_data, 1) ~= length(good_chans)
        warning('Mismatch in number of channels in temp_data (%d) vs good_chans (%d). Subsetting to good channels.', ...
            size(temp_data, 1), length(good_chans));
        temp_data = temp_data(good_chans, :); % Ensure only good channels
    end
    
    % Plot spectrogram for good channels only
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    title(['Spectrogram for ' subj_id ' (Good Channels Only)']);
    saveas(gcf, fullfile(output_dir, 'spectro_channel_rejected.png'), 'png');
    fprintf('Inspect the spectrogram for good channels. Press any key to continue...\n');
        
 

    % --- Common Average Reference (CAR) ---
    for t = 1:num_trials
        if isempty(epochs{t}), continue; end
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN before CAR, skipping.', t);
            continue;
        end
        mean_signal = mean(epochs{t}, 1); % Average over channels
        epochs{t} = epochs{t} - repmat(mean_signal, size(epochs{t}, 1), 1);
        fprintf('Applied CAR to trial %d\n', t);
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_car.png'), 'png');
    %save(fullfile(output_dir, 'spectro_car.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    

    % --- Subtract Baseline ---
    baseline_window = [-0.2 0]; % 200 ms pre-stimulus as baseline (in seconds)
    for t = 1:num_trials
        if isempty(epochs{t}) || any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN or is empty before baseline correction, skipping.', t);
            continue;
        end
        baseline_samples = round((baseline_window(1) + 0.2) * new_srate : (baseline_window(2) + 0.2) * new_srate);
        baseline_samples = max(1, min(size(epochs{t}, 2), baseline_samples)); % Ensure within bounds
        baseline_mean = mean(epochs{t}(:, baseline_samples), 2);
        epochs{t} = epochs{t} - repmat(baseline_mean, 1, size(epochs{t}, 2));
        fprintf('Baseline corrected trial %d\n', t);
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_baseline.png'), 'png');
    %save(fullfile(output_dir, 'spectro_baseline.mat'), 'spec', 'freqs', 'spec_time', 'pow');
    

    %--- Trial Rejection (after baseline) ---
    %--- Trial Rejection (after baseline) ---
    max_samples = max(cellfun(@(x) size(x, 2), epochs));
    epochs_3d = zeros(size(epochs{1}, 1), max_samples, num_trials);
    for t = 1:num_trials
        if ~isempty(epochs{t})
            epochs_3d(:,:,t) = [epochs{t}, zeros(size(epochs{t}, 1), max_samples - size(epochs{t}, 2))];
        end
    end
    trial_var = squeeze(var(epochs_3d, 0, 2)); % Variance across time for each channel
    trial_mean_var = mean(trial_var, 1);       % Mean variance across channels per trial

    % Calculate threshold based on subject's own data
    subject_mean_var = mean(trial_mean_var);
    subject_std_var = std(trial_mean_var);
    trial_reject = trial_mean_var > (subject_mean_var + 3 * subject_std_var);

    %max_samples = max(cellfun(@(x) size(x, 2), epochs));
    %epochs_3d = zeros(size(epochs{1}, 1), max_samples, num_trials);
    %for t = 1:num_trials
       % if ~isempty(epochs{t})
        %    epochs_3d(:,:,t) = [epochs{t}, zeros(size(epochs{t}, 1), max_samples - size(epochs{t}, 2))];
        %end
    %end
    %trial_var = squeeze(var(epochs_3d, 0, 2)); % Variance across time for each channel, then mean across channels
    %trial_mean_var = mean(trial_var, 1); % Mean variance across channels per trial
    %global_mean_var = mean(trial_mean_var);
    %global_std_var = std(trial_mean_var);
    %trial_reject = trial_mean_var > (global_mean_var + 3 * global_std_var);
    % Use a less strict threshold for trial rejection
%trial_reject = trial_mean_var > (global_mean_var + 5 * global_std_var);
    %good_trials = ~trial_reject;
    %epochs_3d = epochs_3d(:,:,good_trials);
    %fprintf('Rejected %d/%d trials\n', sum(trial_reject), length(trial_reject));

    % Convert back to cell array
    epochs = cell(1, sum(good_trials));
    for t = 1:sum(good_trials)
        epochs{t} = epochs_3d(:,:,t);
        non_zero_idx = find(any(epochs{t} ~= 0, 1), 1, 'last');
        if ~isempty(non_zero_idx)
            epochs{t} = epochs{t}(:, 1:non_zero_idx);
        end
    end

    old_num_trials = num_trials;
    num_trials = sum(good_trials);
    all_num_trials(subj_idx) = num_trials; % Update with post-rejection trial count

    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_trial_reject.png'), 'png');
    
    % Numeric indices of good trials
    good_trial_idx = find(good_trials); 
    
    % Update metadata table
    rejection_metadata.Subject(subj_idx)        = subj_id;
    rejection_metadata.TotalChannels(subj_idx)  = num_total_chans;
    rejection_metadata.RejectedChannels(subj_idx) = num_bad_chans;
    rejection_metadata.FinalChannels(subj_idx)   = length(good_chans);
    rejection_metadata.TotalTrials(subj_idx)    = old_num_trials; % before rejection
    rejection_metadata.RejectedTrials(subj_idx) = sum(trial_reject);    % how many rejected
    rejection_metadata.FinalTrials(subj_idx)     = num_trials;       % after rejection
    rejection_metadata.GoodTrialIndices{subj_idx} = good_trial_idx;  % store indices

    % --- Save pre-ICA data for each subject ---
    save(fullfile(output_dir, 'pre_ica_epochs.mat'), 'epochs', 'trial_durations', 'new_srate', 'chan_labels_clean');
end
fprintf('Preprocessing completed for all subjects.\n');


% Bar chart of number of trials per subject
figure;
bar(all_num_trials, 'FaceColor', [0.2 0.6 0.8]);
xticks(1:length(subjects));
xticklabels(subjects);
xtickangle(45); % Rotate labels if needed
xlabel('Subjects');
ylabel('Number of Trials');
title('Number of Trials per Subject');
grid on;

saveas(gcf, fullfile('/home/ozoswita/Dataset/Essex_Movie/output', 'trials_per_subject.png'), 'png');

% --- Save metadata to CSV ---
metadata_file = '/home/ozoswita/Dataset/Essex_Movie/output/rejection_metadata.csv';
rejection_metadata.GoodTrialIndices = cellfun(@mat2str, rejection_metadata.GoodTrialIndices, 'UniformOutput', false);
writetable(rejection_metadata, metadata_file);
fprintf('Rejection metadata saved to %s\n', metadata_file);


