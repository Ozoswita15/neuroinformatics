power_data = load('power.mat');
disp(power_data)

data = power_data.Power;
disp(size(data));

%Transposing the matrix to work intuitively and convert it to 300 (rows) * 2 (columns) format
data = data.';
disp(size(data));

%Storing successful and unsuccessful condition data
success_power = data(:,1);
unsuccess_power = data(:,2);

disp(size(success_power));
disp(size(unsuccess_power));

%mean and std of the distributions
mean_success_power = mean(success_power);
std_success_power = std(success_power);
disp(['Mean (success power): ', num2str(mean_success_power)]);
disp(['Std (success power): ', num2str(std_success_power)]);

mean_unsuccess_power = mean(unsuccess_power);
std_unsuccess_power = std(unsuccess_power);
disp(['Mean (unsuccess power): ', num2str(mean_unsuccess_power)]);
disp(['Std (unsuccess power): ', num2str(std_unsuccess_power)]);

%plotting histogram to check distribution
figure;

subplot(1, 2, 1);
histogram(success_power, 20, 'DisplayName', 'Success Power');
hold on;
xline(mean_success_power, 'Color', 'g', 'LineWidth', 2, 'DisplayName', 'Mean');
xline(mean_success_power - std_success_power, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'Std');
xline(mean_success_power + std_success_power, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'HandleVisibility','off');
title('Spectral power distribution: Successful')
xlabel('Spectral Power');
ylabel('Frequency');
legend show;
hold off;

subplot(1, 2, 2);
histogram(unsuccess_power, 20, 'DisplayName', 'Unsuccess Power');
hold on;
xline(mean_unsuccess_power, 'Color', 'g', 'LineWidth', 2, 'DisplayName', 'Mean');
xline(mean_unsuccess_power - std_unsuccess_power, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'std');
xline(mean_unsuccess_power + std_unsuccess_power, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'HandleVisibility','off');
title('Spectral power distribution: Unsuccessful')
xlabel('Spectral Power');
ylabel('Frequency');
legend show;
hold off;

% perform t-test between 2 conditions to find significance
[h, p] = ttest2(success_power, unsuccess_power);

disp(['h : ', num2str(h)])
if h == 0
    disp('No significant difference found between successful and unsuccessful power conditions.');
else
    disp('Significant difference found between successful and unsuccessful power conditions.');
end

disp(['T-test p-value: ', num2str(p)]);

if p < 0.05
    disp('Significant difference found between successful and unsuccessful power conditions.');
else
    disp('No significant difference found between successful and unsuccessful power conditions.');
end

% Answer: No significant difference found between the 2 conditions
% Choice for statistical test: There were 2 conditions only with high
% equal sample size (300) and variance close to 1, hence a basic t-test is
% sufficient to test for significance between the two conditions. The
% histograms show overlapping means or distribution as well, supporting the non-significance.