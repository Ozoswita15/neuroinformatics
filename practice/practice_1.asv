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
bo = wo/20;
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
[~,~,~,~,~] = spectopo(data_car, size(data_car,2), EEG.srate);

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


% Spectogram

for j = 1:size(epochs_bc, 3)
    []