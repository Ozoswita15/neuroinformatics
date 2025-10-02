clear; clc;
eeglab;

subjects = {'sub-001'};

%subjects = {'sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', ...
            %'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-011'};

eeglabpath = fileparts(which('eeglab.m'));

for subj_idx = 1:length(subjects)
    subj_id = subjects{subj_idx};
    fprintf('Running ICA for %s...\n', subj_id);

    output_dir = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/spectral_data', subj_id);

    % Load preprocessed epochs
    load(fullfile(output_dir, 'pre_ica_epochs.mat'), 'epochs', 'chan_labels_clean', 'new_srate');

    % Convert epochs to continuous data (channels x time)
    data_ica = cat(2, epochs{:}); 

    % --- Convert cleaned labels to proper struct for EEGLAB ---
    num_chans = length(chan_labels_clean);
    chanlocs_struct = struct('labels', cell(1,num_chans));
    for c = 1:num_chans
        chanlocs_struct(c).labels = chan_labels_clean{c};
    end

    % Assign dummy xyz positions from standard 10-20 montage
    chanlocs_std = readlocs(fullfile(eeglabpath, 'plugins', 'dipfit', ...
                                    'standard_BEM', 'elec', 'standard_1005.elc'));
    % Match labels from cleaned channels
    for c = 1:num_chans
        idx_std = find(strcmpi({chanlocs_std.labels}, chan_labels_clean{c}));
        if ~isempty(idx_std)
            chanlocs_struct(c).X = chanlocs_std(idx_std).X;
            chanlocs_struct(c).Y = chanlocs_std(idx_std).Y;
            chanlocs_struct(c).Z = chanlocs_std(idx_std).Z;
        else
            chanlocs_struct(c).X = NaN;
            chanlocs_struct(c).Y = NaN;
            chanlocs_struct(c).Z = NaN;
        end
    end

    % --- Import data into EEGLAB ---
    EEG_ica = pop_importdata('data', data_ica, 'setname', [subj_id '_ICA'], ...
                             'srate', new_srate, 'chanlocs', chanlocs_struct);
    EEG_ica = eeg_checkset(EEG_ica);

    % --- Run ICA with only 10 components ---
    EEG_ica = pop_runica(EEG_ica, 'extended', 1, 'interrupt', 'on');

    % --- Compute IC activations if needed ---
    EEG_ica.icaact = (EEG_ica.icaweights * EEG_ica.icasphere) * EEG_ica.data;

    % --- Topoplot for 10 ICs ---
    figure;
    pop_topoplot(EEG_ica, 0, 1:10, [subj_id ' - Top 10 ICA Components'], 0, 'electrodes', 'on');

    numICs = size(EEG_ica.icaweights,1);
    plotsPerFig = 5;
    
    for startIC = 1:plotsPerFig:numICs
        figure;
        for k = 1:plotsPerFig
            ic = startIC + k - 1;
            if ic > numICs, break; end
            
            % --- Time vector in seconds ---
            timeVec = (0:size(EEG_ica.icaact,2)-1)/EEG_ica.srate + EEG_ica.xmin;
            
            % --- Time series subplot ---
            subplot(plotsPerFig, 2, (k-1)*2 + 1);
            plot(timeVec, EEG_ica.icaact(ic,:));
            xlabel('Time (s)');
            ylabel(['IC ' num2str(ic)]);
            title(['IC ' num2str(ic) ' Time Series']);

            
            % --- Power spectrum subplot ---
            subplot(plotsPerFig, 2, (k-1)*2 + 2);
            [pxx,f] = pwelch(EEG_ica.icaact(ic,:), [], [], [], EEG_ica.srate);
            plot(f, 10*log10(pxx));
            xlim([1 40]);
            xlabel('Frequency (Hz)');
            ylabel('Power (dB)');
            title(['IC ' num2str(ic) ' Power Spectrum']);
        end
        sgtitle([subj_id ' - ICs ' num2str(startIC) ' to ' ...
                 num2str(min(startIC+plotsPerFig-1,numICs))]);
    end




    % --- Pause for manual inspection ---
    fprintf('Pause for manual inspection of %s. Close figures to continue.\n', subj_id);
    uiwait(msgbox('After inspecting topographies and time series, close this message to continue.'));

    % --- Optional: Remove selected ICs manually ---
    %prompt = ['1, 3, 4, 5, 9, 10, 20, 17, 25, 27, 30, 35, 40, 45, 48, 61'];
    prompt = 'Enter ICA component numbers to remove (e.g., [1 3]): ';
    remove_comps = input(prompt);
    if ~isempty(remove_comps)
        EEG_ica = pop_subcomp(EEG_ica, remove_comps, 0);
        fprintf('Removed components: %s\n', mat2str(remove_comps));
    end

    % --- Save ICA result ---
    save(fullfile(output_dir, 'ICA_results.mat'), 'EEG_ica');
    fprintf('ICA completed and saved for %s.\n\n', subj_id);
end

fprintf('ICA processing completed for all subjects.\n');

