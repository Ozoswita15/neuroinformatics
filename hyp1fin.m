
clear;
close all;
clc;

subj_id = 'sub-001';

% Define the base directory 
base_dir = '/home/ozoswita/Dataset/Essex_Movie';

% Define the output directory 
erp_output_dir = fullfile(base_dir, 'erp_analysis');
if ~exist(erp_output_dir, 'dir')
    mkdir(erp_output_dir);
end

fprintf('Starting ERP analysis for %s...\n', subj_id);

% Define file paths
ica_file = fullfile(base_dir, subj_id, 'spectral_data', 'pre_ica_epochs.mat');
event_file = fullfile(base_dir, subj_id, 'eeg', [subj_id '_task-MovieMemory_events.tsv']);
metadata_file = fullfile(base_dir, 'output', 'rejection_metadata.csv');

% Load preprocessed epochs, channel labels, and event data
try
    load(ica_file, 'epochs_final', 'new_srate', 'chan_labels_clean');
catch ME
    warning('Could not load preprocessed data for %s: %s', subj_id, ME.message);
    return;
end
event_data = readtable(event_file, 'FileType', 'text', 'Delimiter', '\t');
rejection_metadata = readtable(metadata_file);

% Extract good trial indices
subj_row = rejection_metadata(strcmp(rejection_metadata.Subject, subj_id), :);
good_trial_indices_str = subj_row.GoodTrialIndices{1};
good_trial_indices = eval(good_trial_indices_str);
num_total_trials = subj_row.TotalTrials;

%% 2. Re-Epoch Data Based on Specific Clip Events

% Define the event types for the start of each clip condition
event_types = {'startOfNotRecognisedClip', 'startOfRememberedClipFirstWatch', 'startOfRecognisedClipFirstWatch'};

% Find the shortest clip duration
min_duration_samples = Inf;
for i = 1:height(event_data)
    current_event_type = event_data.trial_type{i};
    if ismember(current_event_type, event_types)
        clip_start_sample = event_data.sample(i);
        
        end_idx = find(strcmp(event_data.trial_type, 'endOfClip') & event_data.sample > clip_start_sample, 1, 'first');
        
        if ~isempty(end_idx)
            clip_end_sample = event_data.sample(end_idx);
            duration = clip_end_sample - clip_start_sample;
            if duration < min_duration_samples
                min_duration_samples = duration;
            end
        end
    end
end

if isinf(min_duration_samples)
    warning('No valid clip durations found. Skipping.');
    return;
end

min_duration_samples_new_srate = round(min_duration_samples / 2048 * new_srate);
fprintf('The shortest clip duration is %d original samples (or %d at the new sampling rate).\n', min_duration_samples, min_duration_samples_new_srate);

% Trim and Extract all valid epochs for each condition first
all_epochs_struct_full = struct('remembered', [], 'recognised', [], 'not_recognised', []);

for i = 1:height(event_data)
    current_event_type = event_data.trial_type{i};
    
    if ismember(current_event_type, event_types)
        trial_start_event_idx = find(strcmp(event_data.trial_type, 'trialStart') & event_data.sample < event_data.sample(i), 1, 'last');
        if isempty(trial_start_event_idx), continue; end
        
        trial_number = sum(strcmp(event_data.trial_type(1:trial_start_event_idx), 'trialStart'));
        
        [~, epoch_index] = ismember(trial_number, good_trial_indices);
        if epoch_index == 0, continue; end
        
        current_epoch = epochs_final{epoch_index};
        
        start_sample_orig = event_data.sample(i);
        trial_start_sample_orig = event_data.sample(trial_start_event_idx);
        
        offset_samples_orig = start_sample_orig - trial_start_sample_orig;
        start_idx_new = round(offset_samples_orig / 2048 * new_srate) + round(0.2 * new_srate) + 1;
        
        if start_idx_new < 1 || start_idx_new + min_duration_samples_new_srate - 1 > size(current_epoch, 2)
            warning('Invalid indices for re-epoching trial %d. Skipping.', trial_number);
            continue;
        end
        
        new_epoch_data = current_epoch(:, start_idx_new : start_idx_new + min_duration_samples_new_srate - 1);
        
        current_condition = '';
        if contains(current_event_type, 'NotRecognised')
            current_condition = 'not_recognised';
        elseif contains(current_event_type, 'Remembered')
            current_condition = 'remembered';
        elseif contains(current_event_type, 'Recognised')
            current_condition = 'recognised';
        end
        
        if isempty(current_condition), continue; end
        
        if isempty(all_epochs_struct_full.(current_condition))
            all_epochs_struct_full.(current_condition) = new_epoch_data;
        else
            all_epochs_struct_full.(current_condition) = cat(3, all_epochs_struct_full.(current_condition), new_epoch_data);
        end
    end
end
fprintf('Extracted all available epochs: %d remembered, %d recognised, and %d not_recognised epochs.\n', ...
    size(all_epochs_struct_full.remembered, 3), ...
    size(all_epochs_struct_full.recognised, 3), ...
    size(all_epochs_struct_full.not_recognised, 3));

% Find the minimum number of epochs across all conditions
num_recognised = size(all_epochs_struct_full.recognised, 3);
num_remembered = size(all_epochs_struct_full.remembered, 3);
num_not_recognised = size(all_epochs_struct_full.not_recognised, 3);
min_epochs = min([num_recognised, num_remembered, num_not_recognised]);

% Randomly sample to balance the number of epochs
epochs_remembered = all_epochs_struct_full.remembered;
epochs_not_recognised = all_epochs_struct_full.not_recognised;
epochs_recognised = all_epochs_struct_full.recognised;

if num_remembered > min_epochs
    rand_idx_rem = randperm(num_remembered, min_epochs);
    epochs_remembered = epochs_remembered(:,:,rand_idx_rem);
end
if num_not_recognised > min_epochs
    rand_idx_not_rec = randperm(num_not_recognised, min_epochs);
    epochs_not_recognised = epochs_not_recognised(:,:,rand_idx_not_rec);
end
if num_recognised > min_epochs
    rand_idx_rec = randperm(num_recognised, min_epochs);
    epochs_recognised = epochs_recognised(:,:,rand_idx_rec);
end

fprintf('After random sampling, using %d epochs for each condition.\n', min_epochs);

%% 3. Calculate and Plot ERPs

% Calculate ERPs (average across trials for each condition)
erp_remembered = mean(epochs_remembered, 3);
erp_recognised = mean(epochs_recognised, 3);
erp_not_recognised = mean(epochs_not_recognised, 3);

% time points
data_size = size(erp_remembered, 2);
time_points = (0:data_size-1) / new_srate;

% Plot the ERPs for key channels
figure('Name', ['ERP Comparison for ' subj_id]);
chan_to_plot = {'F3', 'P3', 'T8'};

for c_idx = 1:length(chan_to_plot)
    chan_label = chan_to_plot{c_idx};
    chan_idx = find(strcmpi(chan_labels_clean, chan_label));
    if isempty(chan_idx), continue; end
    
    subplot(length(chan_to_plot), 1, c_idx);
    hold on;
    
    plot(time_points, erp_remembered(chan_idx, :), 'b', 'LineWidth', 1.5);
    plot(time_points, erp_recognised(chan_idx, :), 'g', 'LineWidth', 1.5);
    plot(time_points, erp_not_recognised(chan_idx, :), 'r', 'LineWidth', 1.5);
    
    title(['ERP at ' chan_label]);
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    grid on;
    legend({'Remembered', 'Recognised', 'Not Recognised'}, 'Location', 'Best');
    hold off;
end


% Save the ERP plot
erp_plot_file = fullfile(erp_output_dir, [subj_id '_erp_comparison.png']);
saveas(gcf, erp_plot_file);
fprintf('ERP plot saved to %s\n', erp_plot_file);
close(gcf);

% Save the calculated ERP data
erp_data_file = fullfile(erp_output_dir, [subj_id '_erp_data.mat']);
save(erp_data_file, 'erp_remembered', 'erp_recognised', 'erp_not_recognised', 'time_points', 'chan_labels_clean');
fprintf('ERP data saved to %s\n', erp_data_file);

fprintf('ERP analysis completed for %s.\n', subj_id);
