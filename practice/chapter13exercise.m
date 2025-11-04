clear; clc;

% --- Load EEG data ---
load sampleEEGdata
fs      = EEG.srate;
nElect  = EEG.nbchan;   % 64 electrodes
nTime   = EEG.pnts;     % 640 time points
trial   = 10;           % pick one trial
freqs   = linspace(2,30,5); % 5 frequencies
m       = 6;            % wavelet cycles

% === Define Morlet function (your code) ===
function w = morlet_wavelet(f0, m, t)
    sigma = m / (2*pi*f0);
    w = exp(2*1i*pi*f0*t) .* exp(-t.^2/(2*sigma^2));
    % normalize
    w = w ./ sqrt(sum(abs(w).^2)) / sqrt(sigma);
end

% preallocate result: time × freq × electrode × (power/phase)
tfmatrix = zeros(nTime, length(freqs), nElect, 2);

% === Loop over frequencies ===
for fi = 1:length(freqs)
    f = freqs(fi);
    sigma = m/(2*pi*f);
    t_wav = -6*sigma : 1/fs : 6*sigma; % wavelet support
    
    % construct wavelet using YOUR function
    w = morlet_wavelet(f, m, t_wav);
    

    % convolution length
    n_conv = nTime + length(w) - 1;
    fft_w  = fft(w, n_conv);
    
    % === Loop over electrodes ===
    for elec = 1:nElect
        eegdata = squeeze(EEG.data(elec,:,trial));
        fft_e   = fft(eegdata, n_conv);
        
        % convolution
        convres = ifft(fft_e .* fft_w, n_conv);
        
        % chop edges
        halfw   = floor(length(w)/2);
        convres = convres(halfw+1:end-halfw);
        
        % store results
        tfmatrix(:,fi,elec,1) = abs(convres).^2; % power
        tfmatrix(:,fi,elec,2) = angle(convres);  % phase
    end
end