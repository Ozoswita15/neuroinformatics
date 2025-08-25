%Given:

duration = 4;
sampling_rate = 1000; % 1/(1*10^-3)
t = 0:1/sampling_rate:duration; 
freq = [2, 8, 12, 25];

%amp and phase of choice
amp = [1, 2, 3, 4];
phase = [0, pi/4, pi/2, pi];


%Generate individual sine waves
sineWaves = zeros(length(freq), length(t));
for i = 1:length(freq)
    sineWaves(i, :) = amp(i) * sin(2 * pi * freq(i) * t + phase(i));
end

%Mixed wave i.e. average of above sine waves
mixedSineWave = mean(sineWaves, 1);

%Plotting all sine waves
figure;

for i = 1:length(freq)
    subplot(5, 1, i);
    plot(t, sineWaves(i, :));
    title(['Sine Wave ' num2str(i) ' - Frequency: ' num2str(freq(i)) ' Hz']);
    xlabel('Time (s)');
    ylabel('Amplitude');
end


subplot(5, 1, 5);
plot(t, mixedSineWave);
title('Mixed Sine Wave');
xlabel('Time (s)');
ylabel('Amplitude');


% Can you recognize the involved frequencies in the mixed wave from the plot? 
% No, individual frequencies cannot be recognized because the mixed wave
% has superposition of different frequency, amplitude and phase

%Recovering the frequencies using FFT

N = length(mixedSineWave);
Y = fft(mixedSineWave);       
P2 = abs(Y/N);                 % two-sided spectrum
P1 = P2(1:N/2+1);              % single-sided spectrum
P1(2:end-1) = 2*P1(2:end-1);   % correct amplitudes

% frequency axis
f = sampling_rate*(0:(N/2))/N; 

figure;
plot(f, P1)
xlabel('Frequency (Hz)')
ylabel('|Amplitude|')
title('Recovered Frequencies from Mixed Wave')
xlim([0 50])   % only to show up to 50 Hz

%Continuing question 3 here

rng(0);  % for repeatable results

noise_level = [0.2, 0.5, 1.0, 1.5];

noisySineWaves = zeros(size(sineWaves));
for i = 1:length(freq)
    noisySineWaves(i, :) = sineWaves(i, :) + (amp(i)*noise_level(i)) * randn(1, length(t));
end

% Average of the noisy waves
mixedNoisy = mean(noisySineWaves, 1);

figure;
for i = 1:length(freq)
    subplot(5, 1, i);
    plot(t, noisySineWaves(i, :));
    title(['Noisy Wave ' num2str(i) ' (f = ' num2str(freq(i)) ' Hz)']);
    xlabel('Time (s)'); ylabel('Amp');
end
subplot(5, 1, 5);
plot(t, mixedNoisy);
title('Mixed Noisy Wave');
xlabel('Time (s)'); ylabel('Amp');

% Frequency recovery again using FFT

N = length(mixedNoisy);
Y = fft(mixedNoisy);
P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = sampling_rate*(0:(N/2))/N;

figure;
plot(f, P1)
xlabel('Frequency (Hz)')
ylabel('|Amplitude|')
title('Recovered Frequencies from Mixed Noisy Wave')
xlim([0 50]); 
grid on;