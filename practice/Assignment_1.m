 %% -------------------------
% Load EEG dataset
% -------------------------
eeglab
EEG = pop_loadset('eeglab_data.set', fullfile(fileparts(which('eeglab')), 'sample_data'));

data = EEG.data;       % channels x samples
chan_labels = {EEG.chanlocs.labels};

% -------------------------
% Step 1: Filtering: Notch filter at 60 Hz
% -------------------------
wo = 60/(EEG.srate/2); 
bo = wo/35;
[bn,an] = designNotchPeakIIR(CenterFrequency = wo, Bandwidth = bo, Response = 'notch');
data_filt = filtfilt(bn,an,data')';

% -------------------------
% Step 2: Plot PSD before & after filtering
% -------------------------
figure;
[~,~,~,~,~] = spectopo(data, size(data,2), EEG.srate);
figure;
[~,~,~,~,~] = spectopo(data_filt, size(data_filt,2), EEG.srate);


%% -------------------------
% Step 3: Referencing: Common Average Reference
% -------------------------
avg_ref = mean(data_filt, 1);
data_car = data_filt - avg_ref;

figure;
[~,~,~,~,~] = spectopo(data_car, size(data_filt,2), EEG.srate);

%% -------------------------
% Step 4: Epoching around "rt"
% -------------------------
event_latencies = [EEG.event.latency]; 
event_types     = {EEG.event.type};
target_idx = find(strcmp(event_types,'rt'));
epoch_window = round([-0.2 0.8]*EEG.srate);  
epoch_len = diff(epoch_window)+1;

% Preallocate
epochs = nan(size(data_car,1), epoch_len, length(target_idx));
for i = 1:length(target_idx)
    center = round(event_latencies(target_idx(i)));
    idx = center+epoch_window(1):center+epoch_window(2);
    if idx(1)>0 && idx(end)<=size(data_car,2)
        epochs(:,:,i) = data_car(:,idx);
    end
end

%% -------------------------
% Step 5: Baseline correction
% -------------------------
baseline_idx = 1:round(0.2*EEG.srate);
baseline = mean(epochs(:,baseline_idx,:),2);
epochs_bc = epochs - baseline;

%% -------------------------
% Step 6: Improved Artifact Rejection
% -------------------------
% (a) Reject bad trials
% Computes the variance of each channel within each trial, averages that across channels per trial, 
% and then rejects trials whose average variance is more than 3 standard deviations 
% above the global variance mean (across trials).
trial_var = squeeze(var(epochs_bc,0,2));
trial_reject = (mean(trial_var,1) > mean(mean(trial_var))+3*std(mean(trial_var)));

good_trials = ~trial_reject;
epochs_clean = epochs_bc(:,:,good_trials);

fprintf('Rejected %d/%d trials\n', sum(trial_reject), length(trial_reject));

% (b) Reject bad channels (flat or high variance across trials)
chan_var = squeeze(var(epochs_clean,0,[2 3]));
bad_chans = (chan_var < 1e-6) | (chan_var > mean(chan_var)+3*std(chan_var));
good_chans = find(~bad_chans);
epochs_clean = epochs_clean(good_chans,:,:);

fprintf('Rejected %d/%d channels\n', sum(bad_chans), length(chan_labels));

chan_labels_clean = chan_labels(good_chans);