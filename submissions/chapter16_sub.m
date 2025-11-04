% Compare Morlet wavelet, filter->Hilbert, stFFT, and multitaper power

clearvars; close all
load sampleEEGdata  

% ===== parameters =====
channel2plot   = 'Fz';    
frequency      = 15;     
times2save     = -300:25:1000;  % timepoints to evaluate (ms)
timewin_stFFT  = 400;     % ms window for short-time FFT
timewin_mt     = 400;     % ms window for multitaper
baseline_range = [-200 0];% ms baseline for dB-correction
wavelet_cycles = 6;       % cycles for Morlet
filter_bw      = 4;       % half-bandwidth (Hz) for bandpass (center ±bw)
nw_product     = 3;       % NW for multitaper (time-bandwidth)
% =======================

% convert times to indices
times2saveidx = dsearchn(EEG.times',times2save'); 
timewinidx_stFFT = round(timewin_stFFT/(1000/EEG.srate));
timewinidx_mt    = round(timewin_mt/(1000/EEG.srate));

chanidx = strcmpi(channel2plot,{EEG.chanlocs.labels});
if ~any(chanidx)
    error('Channel %s not found in EEG.chanlocs.labels',channel2plot)
end

% find baseline indices within times2save
[~,baseidx(1)] = min(abs(times2save-baseline_range(1)));
[~,baseidx(2)] = min(abs(times2save-baseline_range(2)));


% 1) Complex Morlet wavelet convolution (time-domain via FFT)

% --- Wavelet definition (Gaussian-modulated complex sine) ---
sf = frequency;
s = wavelet_cycles / (2*pi*sf);
half_wav_time = round(3 * s * EEG.srate);  % +/- 3*std approx
t = -half_wav_time : half_wav_time;
morlet = exp(2*1i*pi*sf.*(t/EEG.srate)) .* exp(-(t/EEG.srate).^2 ./ (2*s^2));

% --- FFT parameters for convolution ---
nConv = length(morlet) + EEG.pnts - 1;
nFFT  = 2^nextpow2(nConv);

% --- FFT of wavelet ---
W = fft(morlet, nFFT);
W = W ./ max(W);  % normalize

% --- Data for the channel (time x trials) ---
data_chan = squeeze(EEG.data(chanidx,:,:));  % [time x trials]

% --- FFT of data (zero-padded) ---
dataX = fft(double(data_chan), nFFT, 1);     % [freq x trials]

% --- Convolution per trial ---
analytic_alltrials = zeros(EEG.pnts, EEG.trials);
half_wave = floor(length(morlet) / 2);

for tr = 1:EEG.trials
    convRes = ifft(W .* dataX(:, tr));  % elementwise multiply
    convRes = convRes(half_wave + 1 : half_wave + EEG.pnts);  % trim edges
    analytic_alltrials(:, tr) = convRes;
end

% --- Power: average over trials ---
power_morlet = mean(abs(analytic_alltrials).^2, 2);  % [time x 1]

% --- Reduce to requested timepoints ---
power_morlet_ts = power_morlet(times2saveidx);


% 2) Filter -> Hilbert method

% design bandpass (FIR) around frequency +/- filter_bw
nyq = EEG.srate/2;
fl = max(0.1,(frequency-filter_bw)/nyq);
fh = min(0.99,(frequency+filter_bw)/nyq);
filterOrder = round(3*(EEG.srate/(frequency))); % heuristic
b = fir1(filterOrder,[fl fh],'bandpass');  % FIR filter coefficients

% apply filter to each trial (use filtfilt to avoid phase distort)
analytic_trials = zeros(EEG.pnts, EEG.trials);
for tr = 1:EEG.trials
    sig = double(squeeze(EEG.data(chanidx,:,tr)));
    % pad to reduce edge artifacts
    sigf = filtfilt(b,1,sig);
    analytic_trials(:,tr) = hilbert(sigf);
end
power_fh = mean(abs(analytic_trials).^2,2);
power_fh_ts = power_fh(times2saveidx);

% 3) Short-time FFT (stFFT) method (from your Figure 15.2)

hann_win = .5*(1-cos(2*pi*(0:timewinidx_stFFT-1)/(timewinidx_stFFT-1)));
frex = linspace(0,EEG.srate/2,floor(timewinidx_stFFT/2)+1);

tf_stfft = zeros(length(frex),length(times2save));
for tpi = 1:length(times2save)
    % extract window centered on times2saveidx(tpi)
    idx_center = times2saveidx(tpi);
    idx_start  = idx_center - floor(timewinidx_stFFT/2) + 1;
    idx_end    = idx_center + ceil(timewinidx_stFFT/2);
    % guard boundaries
    if idx_start<1 || idx_end>EEG.pnts
        tf_stfft(:,tpi) = NaN;
        continue
    end
    tempdat = squeeze(EEG.data(chanidx,idx_start:idx_end,:)); % time x trials
    taperdat = bsxfun(@times,double(tempdat),hann_win');
    fdat = fft(taperdat,[],1)/timewinidx_stFFT;
    tf_stfft(:,tpi) = mean(abs(fdat(1:floor(timewinidx_stFFT/2)+1,:)).^2,2);
end
% find freq index closest to desired frequency
[~,freqidx_stfft] = min(abs(frex-frequency));
power_stfft_ts = squeeze(tf_stfft(freqidx_stfft,:))';


% 4) Multitaper (from Chapter 16 code)

timewinidx = timewinidx_mt;
tapers = dpss(timewinidx,nw_product);
f = linspace(0,EEG.srate/2,floor(timewinidx/2)+1);
multitaper_tf = zeros(length(f),length(times2save));

for ti=1:length(times2saveidx)
    idx_center = times2saveidx(ti);
    idx_start  = idx_center - floor(timewinidx/2) + 1;
    idx_end    = idx_center + ceil(timewinidx/2);
    if idx_start<1 || idx_end>EEG.pnts
        multitaper_tf(:,ti) = NaN;
        continue
    end
    taperpow = zeros(floor(timewinidx/2)+1,1);
    for tapi = 1:size(tapers,2)
        data = bsxfun(@times,double(squeeze(EEG.data(chanidx,idx_start:idx_end,:))),tapers(:,tapi));
        pow = fft(data,timewinidx)/timewinidx;
        pow = pow(1:floor(timewinidx/2)+1,:);
        taperpow = taperpow + mean(pow.*conj(pow),2);
    end
    multitaper_tf(:,ti) = taperpow / size(tapers,2);
end
[~,freqidx_mt] = min(abs(f-frequency));
power_mt_ts = multitaper_tf(freqidx_mt,:);

% Before baseline correction, confirming data integrity

% 1) Confirm first signal isn't flat or empty
fprintf('\n[DEBUG] Checking signal variance before baseline correction...\n');
fprintf('Variance of Morlet power: %g\n', var(power_morlet_ts));
fprintf('Variance of Filter->Hilbert power: %g\n', var(power_fh_ts));
fprintf('Variance of Short-time FFT power: %g\n', var(power_stfft_ts(~isnan(power_stfft_ts))));
fprintf('Variance of Multitaper power: %g\n', var(power_mt_ts(~isnan(power_mt_ts))));

if var(power_morlet_ts)==0
    warning('Morlet power is constant (zero variance) — check the convolution or data input.');
end

% 2) Plot before normalization/log scaling to ensure raw power exists
figure('Name','Raw power before dB normalization','units','normalized','position',[0.1 0.1 0.6 0.4])
subplot(1,4,1)
plot(times2save, power_morlet_ts, 'LineWidth', 1.2)
title('Raw Morlet Power'); xlabel('Time (ms)'); ylabel('Power')

subplot(1,4,2)
plot(times2save, power_fh_ts, 'LineWidth', 1.2)
title('Raw Filter->Hilbert Power'); xlabel('Time (ms)')

subplot(1,4,3)
plot(times2save, power_stfft_ts, 'LineWidth', 1.2)
title('Raw Short-time FFT Power'); xlabel('Time (ms)')

subplot(1,4,4)
plot(times2save, power_mt_ts, 'LineWidth', 1.2)
title('Raw Multitaper Power'); xlabel('Time (ms)')

sgtitle('Checking if raw power is nonzero before normalization')

% Baseline correction (dB) for all methods (same baseline)

base_morlet = mean(power_morlet_ts(baseidx(1):baseidx(2)));
base_fh     = mean(power_fh_ts(baseidx(1):baseidx(2)));
base_stfft  = mean(power_stfft_ts(baseidx(1):baseidx(2)));
base_mt     = mean(power_mt_ts(baseidx(1):baseidx(2)));

db_morlet = 10*log10( power_morlet_ts ./ base_morlet );
db_fh     = 10*log10( power_fh_ts     ./ base_fh );
db_stfft  = 10*log10( power_stfft_ts  ./ base_stfft );
db_mt     = 10*log10( power_mt_ts     ./ base_mt );


% Plot all four timecourses

figure('units','normalized','position',[0.05 0.05 0.7 0.7])
subplot(2,2,1)
plot(times2save,db_morlet,'LineWidth',1.2); xlabel('Time (ms)'); ylabel('dB'); title(sprintf('Morlet (%d cycles) %g Hz @ %s',wavelet_cycles,frequency,channel2plot))
set(gca,'xlim',[times2save(1) times2save(end)]); grid on

subplot(2,2,2)
plot(times2save,db_fh,'LineWidth',1.2); xlabel('Time (ms)'); ylabel('dB'); title(sprintf('Filter->Hilbert (±%g Hz) @ %s',filter_bw,channel2plot))
set(gca,'xlim',[times2save(1) times2save(end)]); grid on

subplot(2,2,3)
plot(times2save,db_stfft,'LineWidth',1.2); xlabel('Time (ms)'); ylabel('dB'); title(sprintf('short-time FFT (win %d ms) @ %s',timewin_stFFT,channel2plot))
set(gca,'xlim',[times2save(1) times2save(end)]); grid on

subplot(2,2,4)
plot(times2save,db_mt,'LineWidth',1.2); xlabel('Time (ms)'); ylabel('dB'); title(sprintf('Multitaper (NW=%g) @ %s',nw_product,channel2plot))
set(gca,'xlim',[times2save(1) times2save(end)]); grid on

% tighten y-limits to same range for easier visual comparison
yl = [min([db_morlet(:);db_fh(:);db_stfft(:);db_mt(:)]) max([db_morlet(:);db_fh(:);db_stfft(:);db_mt(:)])];
for sp = 1:4, subplot(2,2,sp), ylim(yl + [-1 1]); end
