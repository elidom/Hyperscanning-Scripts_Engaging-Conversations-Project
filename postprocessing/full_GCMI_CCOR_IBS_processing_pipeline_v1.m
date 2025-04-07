clear, clc, close all
tic;
% This version of the pipeline will calculate the inter-brain coordination
% for each speaker-listener direction

%% Step 1: Load and epoch each dataset
[file_names, file_paths] = uigetfile('ready_data_v0\\*.set',  'set Files (*.set*)','MultiSelect','on');

sub_A = hyper_epoching(file_names{1});
sub_B = hyper_epoching(file_names{2});

if size(sub_A, 1) == size(sub_B, 1)
    if sum(string(sub_A(:,2)) == string(sub_B(:,2))) == size(sub_A, 1)
        disp('# ----------------- Two datasets epoched successfully ----------------- #')
    else
        disp('!!! Error: One or more mismatches found between datasets.')
    end
else
    disp('!!! Error: Datasets are of different size.')
end

%% Step 2: Prepare EEG-only Cell Arrays and Confirm that Trial Length Matches

% Misc Parameters
Nblock =   size(sub_A, 1);
Nch    = size(sub_B{1,1}, 1); 
eeg_fs = 500;

% Initialize arrays
info_A = cell(Nblock, 1);
eeg_A = cell(Nblock, 1);
stimtrack = cell(Nblock, 1);

info_B = cell(Nblock, 1);
eeg_B = cell(Nblock, 1);

% Extract EEG data from trials (SUB A)
for i = 1:Nblock
    % Subject A
    curr_data_temp_A = sub_A{i, 1};                 % Current trial EEG data
    curr_data_A = curr_data_temp_A([1:32 34], :);   % Stim track (channel 65)
    info_A{i} = sub_A{i, 2};                        % Store trial information
    eeg_A{i} = curr_data_A;                        % Store EEG data
    stimtrack{i} = curr_data_temp_A(33, :);
    
    % Subject B
    curr_data_temp_B = sub_B{i, 1};                 % Current trial EEG data
    curr_data_B = curr_data_temp_B(1:33, :);   % Stim track (channel 65)
    info_B{i} = sub_B{i, 2};                        % Store trial information
    eeg_B{i} = curr_data_B;                        % Store EEG data
end

temp_count = 0;
for s = 1:Nblock
    size1 = size(eeg_A{s},2);
    size2 = size(eeg_B{s},2);
    if size1 == size2
        temp_count = temp_count + 1;
    end
end
if temp_count == Nblock
    disp('# - - - - - - - - - Trial length matches across datasets! - - - - - - - - - #')
end


%% Step 3: Apply Filters and Compute Gradients

% - - - - - - - - - - Alpha
fs = 500;         
order = 2000; % Filter order
low_cutoff = 8 / (fs/2);    
high_cutoff = 12 / (fs/2);  
b = fir1(order, [low_cutoff high_cutoff], 'bandpass');

% diagnose
figure;
freqz(b, 1, 1024, fs);
title('Frequency Response of the Alpha Band FIR Filter');

% Prepare for filtering
alpha_A   = cell(Nblock, 1);
d_alpha_A = cell(Nblock, 1);
alpha_B   = cell(Nblock, 1);
d_alpha_B = cell(Nblock, 1);

% Filter the data and compute gradients
for bi = 1:Nblock
    % Filter the sub-A
    filteredData = filtfilt(b, 1, double(eeg_A{bi})');
    alpha_A{bi} = [filteredData'; stimtrack{bi}];
    
    % Filter the sub-B
    filteredData = filtfilt(b, 1, double(eeg_B{bi})');
    alpha_B{bi} = [filteredData'; stimtrack{bi}];
    
    % optional, add gradient to 2d calculation (this could smooth out filter oscilations)
    d_alpha_A{bi} = gradient_dim1(alpha_A{bi});
    d_alpha_B{bi} = gradient_dim1(alpha_B{bi});
end

disp('# -- Alpha done -- #')

% - - - - - - - - - - BETA

fs = 500;         
order = 2000; % Filter order
low_cutoff = 13 / (fs/2);    
high_cutoff = 30 / (fs/2);  
b = fir1(order, [low_cutoff high_cutoff], 'bandpass');

% diagnose
figure;
freqz(b, 1, 1024, fs);
title('Frequency Response of the Beta Band FIR Filter');

% Prepare for filtering
beta_A   = cell(Nblock, 1);
d_beta_A = cell(Nblock, 1);
beta_B   = cell(Nblock, 1);
d_beta_B = cell(Nblock, 1);

% Filter the data and compute gradients
for bi = 1:Nblock
    % Filter the sub-A
    filteredData = filtfilt(b, 1, double(eeg_A{bi})');
    beta_A{bi} = [filteredData'; stimtrack{bi}];
    
    % Filter the sub-B
    filteredData = filtfilt(b, 1, double(eeg_B{bi})');
    beta_B{bi} = [filteredData'; stimtrack{bi}];
    
    % optional, add gradient to 2d calculation (this could smooth out filter oscilations)
    d_beta_A{bi} = gradient_dim1(beta_A{bi});
    d_beta_B{bi} = gradient_dim1(beta_B{bi});
end

disp('# -- Beta done -- #')


% - - - - - - - - - - - Theta
fs = 500;         
order = 2000;      % Filter order 

low_cutoff = 2 / (fs/2);    
high_cutoff = 8 / (fs/2); 
b = fir1(order, [low_cutoff high_cutoff], 'bandpass');

% diagnose
figure;
freqz(b, 1, 1024, fs);
title('Frequency Response of the Theta Band FIR Filter');

% Prepare for filtering EEG and audio signals between 2-10 Hz
theta_A   = cell(Nblock, 1);
d_theta_A = cell(Nblock, 1);
theta_B   = cell(Nblock, 1);
d_theta_B = cell(Nblock, 1);

% Filter the data and compute gradients
for bi = 1:Nblock

    % Filter the sub-A
    filteredData = filtfilt(b, 1, double(eeg_A{bi})');
    theta_A{bi} = [filteredData'; stimtrack{bi}];
    
    % Filter the sub-B
    filteredData = filtfilt(b, 1, double(eeg_B{bi})');
    theta_B{bi} = [filteredData'; stimtrack{bi}];

    % optional, add gradient to 2d calculation (this could smooth out filter oscilations)
    d_theta_A{bi} = gradient_dim1(theta_A{bi});
    d_theta_B{bi} = gradient_dim1(theta_B{bi});
end

disp('# -- Theta done -- #')

%% Separate data by speaker/listener relationship for each frequency band

% Define which trials are conversations
isConversation = string(sub_A(:,2)) == "HighInterest" | string(sub_A(:,2)) == "LowInterest";
convIdx = find(isConversation);

% Organize your frequency band data into cell arrays for subject A and B
bandNames = {'alpha', 'beta', 'theta'};
data_A = {alpha_A, beta_A, theta_A};
data_B = {alpha_B, beta_B, theta_B};

% Preallocate output cell arrays for each frequency band.
% Each cell in these arrays will itself be a cell array for each conversation trial.
brain_A_WhenSpeaks_A = cell(length(bandNames), 1);
brain_A_WhenSpeaks_B = cell(length(bandNames), 1);
brain_B_WhenSpeaks_A = cell(length(bandNames), 1);
brain_B_WhenSpeaks_B = cell(length(bandNames), 1);

% Initialize the output for each band
for b = 1:length(bandNames)
    brain_A_WhenSpeaks_A{b} = cell(length(convIdx),1);
    brain_A_WhenSpeaks_B{b} = cell(length(convIdx),1);
    brain_B_WhenSpeaks_A{b} = cell(length(convIdx),1);
    brain_B_WhenSpeaks_B{b} = cell(length(convIdx),1);
end

% Loop over each conversation trial
counter = 1;
for i = convIdx'
    % Get the shared trial information once (masks and speaker segmentations)
    mask_data   = sub_A{i,4};
    spkrA_sgmts = sub_A{i,5};
    spkrB_sgmts = sub_A{i,6};
    
    % Loop over each frequency band
    for b = 1:length(bandNames)
        % Select current frequency band data for both subjects
        brain_data_A = data_A{b}{i};
        brain_data_B = data_B{b}{i};
        
        % Consistency checks (you might want to adjust these as needed)
        if ~any(size(brain_data_A) == size(brain_data_B))
            error("Sizes do not match across neural datasets!")
        end
        if any(~any(mask_data == sub_B{i,4}))
            error("Mask segments do not correspond across datasets!")
        end
        if any(~any(spkrA_sgmts == sub_B{i,5}))
            error("Speaker A segments do not correspond across datasets!")
        end
        if any(~any(spkrB_sgmts == sub_B{i,6}))
            error("Speaker B segments do not correspond across datasets!")
        end
        
        % Split the data for each speaker-listener relationship
        brain_A_WhenSpeaks_A{b}{counter} = hyper_split_speakers(brain_data_A, spkrA_sgmts, mask_data);
        brain_A_WhenSpeaks_B{b}{counter} = hyper_split_speakers(brain_data_A, spkrB_sgmts, mask_data);
        brain_B_WhenSpeaks_A{b}{counter} = hyper_split_speakers(brain_data_B, spkrA_sgmts, mask_data);
        brain_B_WhenSpeaks_B{b}{counter} = hyper_split_speakers(brain_data_B, spkrB_sgmts, mask_data);
    end
    counter = counter + 1;
end

% Optionally, display a message once done.
disp('# -- Speaker/Listener splitting done for all frequency bands -- #')

%% Create information structure for each trial and frequency band

numBands = numel(bandNames);

% Initialize output cell arrays for Brain A and Brain B for each frequency band
out_brainA = cell(numBands, 1);
out_brainB = cell(numBands, 1);
for b = 1:numBands
    out_brainA{b} = {};  % Each row: {splitData, Speaker, Condition, Index, FrequencyBand}
    out_brainB{b} = {};
end

% Loop through each frequency band
for b = 1:numBands
    % Loop through each conversation trial (convIdx holds indices for HighInterest or LowInterest)
    for i = 1:length(convIdx)
        convIdx_i = convIdx(i);
        % Extract condition and index information from sub_A
        condition = sub_A{convIdx_i, 2};  
        index = sub_A{convIdx_i, 3};

        % For Brain A:
        % When Speaker A speaks
        out_brainA{b}(end+1, :) = {brain_A_WhenSpeaks_A{b}{i}, 'Speaker A', condition, index, bandNames{b}};
        % When Speaker B speaks
        out_brainA{b}(end+1, :) = {brain_A_WhenSpeaks_B{b}{i}, 'Speaker B', condition, index, bandNames{b}};
        
        % For Brain B:
        % When Speaker A speaks
        out_brainB{b}(end+1, :) = {brain_B_WhenSpeaks_A{b}{i}, 'Speaker A', condition, index, bandNames{b}};
        % When Speaker B speaks
        out_brainB{b}(end+1, :) = {brain_B_WhenSpeaks_B{b}{i}, 'Speaker B', condition, index, bandNames{b}};
    end
end

% Optionally, display a message when done
disp('# -- Output structure created for all frequency bands -- #')


%% Add non-conversation trials for each frequency band
nonConvIdx = find(~isConversation);

% Preallocate cell arrays for non-conversation outputs for each frequency band
nonconv_brainA = cell(numBands, 1);
nonconv_brainB = cell(numBands, 1);
for b = 1:numBands
    nonconv_brainA{b} = {};
    nonconv_brainB{b} = {};
end

% Loop through each non-conversation trial and each frequency band
for b = 1:numBands
    for i = 1:length(nonConvIdx)
        trialIdx = nonConvIdx(i);
        condition = sub_A{trialIdx, 2};  % "Listening" or "Silence"
        trialIndex = sub_A{trialIdx, 3};
        
        % Use the appropriate frequency band data from data_A and data_B
        nonconv_brainA{b}(end+1, :) = {data_A{b}{trialIdx}, 'none', condition, trialIndex, bandNames{b}};
        nonconv_brainB{b}(end+1, :) = {data_B{b}{trialIdx}, 'none', condition, trialIndex, bandNames{b}};
    end
end

%% Prepend non-conversation trials to the conversation outputs for each frequency band

% Here, out_brainA and out_brainB were built earlier for conversation trials
final_out_brainA = cell(numBands, 1);
final_out_brainB = cell(numBands, 1);
for b = 1:numBands
    % Concatenate non-conversation trials (which now have 5 columns: data, speaker label, condition, index, band)
    % with the conversation trials already stored in out_brainA{b} and out_brainB{b}
    final_out_brainA{b} = [nonconv_brainA{b}; out_brainA{b}];
    final_out_brainB{b} = [nonconv_brainB{b}; out_brainB{b}];
end

%% Add signal length to the right for later normalization
getSignalLength = @(data) size(data, 2);  % Helper to count columns (signal length)

% Now add a new column with the signal length.
% Because our data now have 5 columns, the new column will be the 6th.
for b = 1:numBands
    for i = 1:size(final_out_brainA{b}, 1)
        final_out_brainA{b}{i, 6} = getSignalLength(final_out_brainA{b}{i, 1});
    end
    
    for i = 1:size(final_out_brainB{b}, 1)
        final_out_brainB{b}{i, 6} = getSignalLength(final_out_brainB{b}{i, 1});
    end
end

disp('# -- Non-conversation trials added and final outputs constructed for all frequency bands -- #')

%% Compute Inter-Brain Coordination With Time Lags for Each Frequency Band

% Structure to store results for each frequency band
final_results = struct();

for b = 1:numBands
    % Extract the data column (the first column holds the EEG matrix)
    split_A = final_out_brainA{b}(:, 1);
    split_B = final_out_brainB{b}(:, 1);

    Nblock = numel(split_B);
    % Determine the number of channels 
    Nch = size(split_A{1}, 1)-1;  

    % Define time lags (in samples) and compute other lag parameters
    if b == 2
        lags = -100:1:100;
    else
        lags = -400:4:400;
    end

    Nlags = length(lags);
    lagtime = lags ./ eeg_fs;    % convert lags to seconds
    L_max = max(abs(lags));

    % Preallocate results matrix: dimensions: [trial, lag, channel_A, channel_B]
    results = nan(Nblock, Nlags, Nch, Nch);

    % tic;
    % Use parfor to loop over trials (each trial is independent)
    parfor bi = 1:Nblock
        % For each trial, extract the EEG matrix 
        brain_A = copnorm(split_A{bi}(1:33,:)')';  
        brain_B = copnorm(split_B{bi}(1:33,:)')';  

        for li = 1:Nlags
            l = lags(li);
            % Align signals based on lag
            if l == 0
                % Zero lag: trim a fixed number of samples (L_max)
                samples2trim = L_max;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = size(brain_A, 2) - floor(samples2trim/2);
                Alag = brain_A(:, idx_start:idx_end);
                Blag = brain_B(:, idx_start:idx_end);
            elseif l < 0
                % Negative lag: stimulus precedes response
                lag_abs = abs(l);
                samples2trim = L_max - lag_abs;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = size(brain_A, 2) - floor(samples2trim/2);
                A_segment = brain_A(:, idx_start:idx_end);
                B_segment = brain_B(:, idx_start:idx_end);
                Alag = A_segment(:, 1:end - lag_abs);
                Blag = B_segment(:, lag_abs + 1:end);
            else % l > 0
                % Positive lag: response precedes stimulus
                samples2trim = L_max - l;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = size(brain_A, 2) - floor(samples2trim/2);
                A_segment = brain_A(:, idx_start:idx_end);
                B_segment = brain_B(:, idx_start:idx_end);
                Alag = A_segment(:, l + 1:end);
                Blag = B_segment(:, 1:end - l);
            end

            % Loop over channels of subject A and B to compute MI
            for chi = 1:Nch
                chan_A = Alag(chi, :);
                for cha = 1:Nch
                    chan_B = Blag(cha, :);
                    if chi == 34 && cha == 34
                        sync = nan;
                    else
                        sync = mi_gg(chan_A', chan_B', true, true);  % compute mutual information
                    end
                    results(bi, li, chi, cha) = sync;
                end
            end
        end
        disp(['Frequency Band ' bandNames{b} ', Block ' num2str(bi) '/' num2str(Nblock) ' done!'])
    end
    elapsedTime = toc;
    fprintf('Frequency Band %s Elapsed time: %.4f hours\n', bandNames{b}, elapsedTime/3600);

    % Store raw results for the current frequency band
    final_results.(bandNames{b}).raw = results;

    %% Normalize results for current frequency band
    effective_lengths = zeros(Nblock, 1);
    for bi = 1:Nblock
        % For normalization, compute the effective signal length using the l=0 trimming
        samples2trim = L_max;
        idx_start = ceil(samples2trim/2) + 1;
        idx_end = size(split_A{bi}, 2) - floor(samples2trim/2);
        effective_lengths(bi) = idx_end - idx_start + 1;
    end

    % Choose a baseline length 
    baseline_length = median(effective_lengths);

    % Apply a correction factor to account for differences in effective trial lengths
    normalized_results = results;
    for bi = 1:Nblock
        correction_factor = effective_lengths(bi) / baseline_length;
        normalized_results(bi, :, :, :) = results(bi, :, :, :) * correction_factor;
    end

    final_results.(bandNames{b}).normalized = normalized_results;
end

disp('# -- Inter-Brain Coordination Computation Completed for All Frequency Bands -- #')


%% Try this to export data

% channels = [sub_B{1,7}; "Stimtrack"];
channels = [sub_B{1,7}];

tmp = final_out_brainA{1}(:,2:4);
trial_idx = {};
for i = 1:size(tmp, 1)
    trial_idx{i} = [char(tmp{i,2}) '_' char(tmp{i,1}) '_' num2str(tmp{i,3})];
end

for b = 1:numBands
    % Get the normalized MI results for the current band
    norm_results = final_results.(bandNames{b}).normalized;
    
    % Determine dimensions
    [Nblock, Nlags, Nch, ~] = size(norm_results);

    % Create grids of indices for trials, lags, channels
    [trialIdx, lagIdx, chanAIdx, chanBIdx] = ndgrid(trial_idx, 1:Nlags, channels, channels);
    
    % Flatten the indices and MI values
    trial_flat  = trialIdx(:);
    lag_flat    = lags(lagIdx(:))';  % Convert lag index to actual lag value (in samples or seconds)
    chanA_flat  = chanAIdx(:);
    chanB_flat  = chanBIdx(:);
    MI_flat     = norm_results(:);
    
    % Create a table with columns for Trial, Lag, Channel A, Channel B, MI, and Frequency Band
    T = table(trial_flat, lag_flat, chanA_flat, chanB_flat, MI_flat, ...
              'VariableNames', {'Trial', 'Lag', 'Channel_A', 'Channel_B', 'MI'});
    % Add frequency band information (as a string column)
    T.FrequencyBand = repmat({bandNames{b}}, height(T), 1);
    
    % Write the table to a CSV file
    subId = file_names{1}(15:22);
    filename = fullfile('output', [subId '_' bandNames{b} '_MI_results.csv']);
    writetable(T, filename);
    
    fprintf('Exported %s\n', filename);
end

%% Explore Results (GCMI) of Averaged Lags for Each Frequency Band

% Loop over each frequency band:
bandNames = {'alpha','beta','theta'};
numBands = numel(bandNames);

% misc info
listening_idx = find(string(final_out_brainB{1}(:,3)) == "Listening")';
silence_idx   = find(string(final_out_brainB{1}(:,3)) == "Silence")';
low_interest_idx = find(string(final_out_brainB{1}(:,3)) == "LowInterest")';
high_interest_idx = find(string(final_out_brainB{1}(:,3)) == "HighInterest")';

Aspeaks = find(string(final_out_brainB{1}(:,2)) == "Speaker A")';
Bspeaks = find(string(final_out_brainB{1}(:,2)) == "Speaker B")';

frontal_idx        = [1,2,3,4,30,31,32];            % Fp1, Fz, F3, F7, F4, F8, Fp2
left_temporal_idx  = [5,9,10];                       % FT9, T7, TP9
central_idx        = [8,6,7,28,29,24,25,33];          % C3, FC5, FC1, FC6, FC2, Cz, C4, FCz
right_temporal_idx = [27,26,21];                     % FT10, T8, TP10
parietal_idx       = [11,12,13,14,15,19,20,22,23];     % CP5, CP1, Pz, P3, P7, P4, P8, CP6, CP2
occipital_idx      = [16,17,18];                     % O1, Oz, O2

for b = 1:numBands
    % Extract normalized results for current band (dimensions: [Nblock, Nlags, Nch, Nch])
    normalized_results = final_results.(bandNames{b}).normalized;
    
    % Compute the main IBS value in each trial & channel pair as the maximum over the desired window.
    win_idx = 95:107;
    results_around_zero = squeeze(max(normalized_results(:,win_idx,:,:), [], 2));
    
    around_zero = (lags(win_idx)*2);  % (e.g., -8 to 8 ms)
    
    % --- Average results across trials for each condition ---
    avg_mat_listening     = squeeze(mean(results_around_zero(listening_idx,1:33,1:33),1));
    avg_mat_low_interestAspks  = squeeze(mean(results_around_zero(intersect(Aspeaks, low_interest_idx)',1:33,1:33),1));
    avg_mat_low_interestBspks  = squeeze(mean(results_around_zero(intersect(Bspeaks, low_interest_idx)',1:33,1:33),1));
    avg_mat_high_interestAspks = squeeze(mean(results_around_zero(intersect(Aspeaks, high_interest_idx)',1:33,1:33),1));
    avg_mat_high_interestBspks = squeeze(mean(results_around_zero(intersect(Bspeaks, high_interest_idx)',1:33,1:33),1));
    avg_mat_silence       = squeeze(mean(results_around_zero(silence_idx,1:33,1:33),1));
    
    % --- Order channels by MI density (highest to lowest) for each condition ---
    % (Assumes that the variable "channels" contains the channel names.)
    [~, order_listening] = sort(mean(avg_mat_listening,1), 'descend');
    avg_mat_listening = flipud(avg_mat_listening(order_listening, order_listening));
    channels_listening = channels(order_listening);
    
    [~, order_lowA] = sort(mean(avg_mat_low_interestAspks,1), 'descend');
    avg_mat_low_interestAspks = flipud(avg_mat_low_interestAspks(order_lowA, order_lowA));
    channels_lowA = channels(order_lowA);
    
    [~, order_lowB] = sort(mean(avg_mat_low_interestBspks,1), 'descend');
    avg_mat_low_interestBspks = flipud(avg_mat_low_interestBspks(order_lowB, order_lowB));
    channels_lowB = channels(order_lowB);
    
    [~, order_highA] = sort(mean(avg_mat_high_interestAspks,1), 'descend');
    avg_mat_high_interestAspks = flipud(avg_mat_high_interestAspks(order_highA, order_highA));
    channels_highA = channels(order_highA);
    
    [~, order_highB] = sort(mean(avg_mat_high_interestBspks,1), 'descend');
    avg_mat_high_interestBspks = flipud(avg_mat_high_interestBspks(order_highB, order_highB));
    channels_highB = channels(order_highB);
    
    [~, order_silence] = sort(mean(avg_mat_silence,1), 'descend');
    avg_mat_silence = flipud(avg_mat_silence(order_silence, order_silence));
    channels_silence = channels(order_silence);
    
    % --- Compute a uniform color scale across all conditions ---
    all_values = [avg_mat_listening(:); avg_mat_low_interestAspks(:); avg_mat_low_interestBspks(:); ...
                  avg_mat_high_interestAspks(:); avg_mat_high_interestBspks(:); avg_mat_silence(:)];
    cmin = quantile(all_values, 0.05);
    cmax = quantile(all_values, 0.95);
    
    % --- Create Figure with Subplots for Each Condition ---
    figure('Name', sprintf('Averaged MI - %s band', bandNames{b})), clf;
    
    % Listening condition
    subplot(2,3,1)
    imagesc(avg_mat_listening);
    caxis([cmin cmax]);       % Uniform color scale
    colorbar;
    title('Listening');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_listening), 'XTickLabel', channels_listening, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_listening), 'YTickLabel', flip(channels_listening));
    
    % Low Interest - A speaks
    subplot(2,3,2)
    imagesc(avg_mat_low_interestAspks);
    caxis([cmin cmax]);
    colorbar;
    title('Low Interest - A speaks');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_lowA), 'XTickLabel', channels_lowA, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_lowA), 'YTickLabel', flip(channels_lowA));
    
    % High Interest - A speaks
    subplot(2,3,3)
    imagesc(avg_mat_high_interestAspks);
    caxis([cmin cmax]);
    colorbar;
    title('High Interest - A speaks');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_highA), 'XTickLabel', channels_highA, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_highA), 'YTickLabel', flip(channels_highA));
    
    % Silence condition
    subplot(2,3,4)
    imagesc(avg_mat_silence);
    caxis([cmin cmax]);
    colorbar;
    title('Silence');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_silence), 'XTickLabel', channels_silence, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_silence), 'YTickLabel', flip(channels_silence));
    
    % Low Interest - B speaks
    subplot(2,3,5)
    imagesc(avg_mat_low_interestBspks);
    caxis([cmin cmax]);
    colorbar;
    title('Low Interest - B speaks');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_lowB), 'XTickLabel', channels_lowB, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_lowB), 'YTickLabel', flip(channels_lowB));
    
    % High Interest - B speaks
    subplot(2,3,6)
    imagesc(avg_mat_high_interestBspks);
    caxis([cmin cmax]);
    colorbar;
    title('High Interest - B speaks');
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
    set(gca, 'XTick', 1:length(channels_highB), 'XTickLabel', channels_highB, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_highB), 'YTickLabel', flip(channels_highB));
    
end

%% Define surrogate lag parameters per frequency band
surrogate_params.alpha.lags = -24:8:24;   % for Alpha surrogates
surrogate_params.beta.lags  = -8:2:8;      % for Beta surrogates
surrogate_params.theta.lags = -24:8:24;     % example for Theta surrogates

% Loop over frequency bands to perform surrogate analysis, normalization, p-value computation, 
% and plotting/condition combination.
for b = 1:numBands
    band = bandNames{b};
    % Get surrogate lag parameters for this band:
    current_lags = surrogate_params.(band).lags;
    Nlags = length(current_lags);
    lagtime = current_lags ./ eeg_fs;  % convert to seconds
    L_max = max(abs(current_lags));
    Nsurrogate = 200;  % number of surrogate iterations
    
    split_A = final_out_brainA{b}(:,1);
    split_B = final_out_brainB{b}(:,1);
    Nblock = numel(split_A);
    % Assume that the EEG matrices have one extra row (stimtrack) that we ignore:
    Nch = size(split_A{1},1)-1;
    
    % Preallocate the surrogate results array.
    % Dimensions: [trial, surrogate lag, channel_A, channel_B, surrogate iteration]
    results_sur = nan(Nblock, Nlags, Nch, Nch, Nsurrogate);
    
    % tic;
    % Process each trial in parallel
    parfor bi = 1:Nblock
        % Extract neural data from the current trial.
        % Use rows 1:33 (channels) from the matrix (assuming row 34 is stimtrack)
        brain_A = copnorm(split_A{bi}(1:33,:)')';
        brain_B = split_B{bi}(1:33,:);
        T = size(brain_B,2);  % number of timepoints
        
        for surr = 1:Nsurrogate
            % Random circular shift on subject B’s data (starting at ~8 seconds)
            shift_amount = randi([4000, T-4000]);
            pre_brain_B_shift = circshift(brain_B, [0, shift_amount]);
            brain_B_shift = copnorm(pre_brain_B_shift')';
            
            for li = 1:Nlags
                l = current_lags(li);
                if l == 0
                    samples2trim = L_max;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(brain_A,2) - floor(samples2trim/2);
                    Alag = brain_A(:, idx_start:idx_end);
                    Blag = brain_B_shift(:, idx_start:idx_end);
                elseif l < 0
                    lag_abs = abs(l);
                    samples2trim = L_max - lag_abs;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(brain_A,2) - floor(samples2trim/2);
                    A_segment = brain_A(:, idx_start:idx_end);
                    B_segment = brain_B_shift(:, idx_start:idx_end);
                    Alag = A_segment(:, 1:end - lag_abs);
                    Blag = B_segment(:, lag_abs + 1:end);
                else  % l > 0
                    samples2trim = L_max - l;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(brain_A,2) - floor(samples2trim/2);
                    A_segment = brain_A(:, idx_start:idx_end);
                    B_segment = brain_B_shift(:, idx_start:idx_end);
                    Alag = A_segment(:, l + 1:end);
                    Blag = B_segment(:, 1:end - l);
                end
                
                % Loop over channel pairs and compute mutual information (MI)
                for chi = 1:Nch
                    chan_A = Alag(chi,:);
                    for cha = 1:Nch
                        chan_B_shift = Blag(cha,:);
                        if chi == 34 && cha == 34
                            sync = nan;
                        else
                            sync_sur = mi_gg(chan_A', chan_B_shift', true, true);  % compute mutual information
                        end
                        results_sur(bi, li, chi, cha, surr) = sync_sur;
                    end
                end
            end
        end
        disp(['Frequency band ' band ', trial ' num2str(bi) '/' num2str(Nblock) ' done!']);
    end
    elapsedTime = toc;
    fprintf('Frequency band %s surrogate elapsed time: %.4f hours\n', band, elapsedTime/3600);
    
    % Save surrogate results for the current band
    save_filename = sprintf('pilot02_timeMaxxedGCMI3_%s_splitSpkrs_resultsAndSurrogates.mat', upper(band));
    save(save_filename, 'results_sur');
    
    %% Normalize surrogates
    effective_lengths = zeros(Nblock,1);
    for bi = 1:Nblock
        samples2trim = L_max;
        idx_start = ceil(samples2trim/2) + 1;
        idx_end = size(split_A{bi},2) - floor(samples2trim/2);
        effective_lengths(bi) = idx_end - idx_start + 1;
    end
    baseline_length = median(effective_lengths);
    normalized_surrs = results_sur;
    for bi = 1:Nblock
        correction_factor = effective_lengths(bi) / baseline_length;
        normalized_surrs(bi,:,:,:,:) = results_sur(bi,:,:,:,:) * correction_factor;
    end
    
    %% Compute p-values for each channel pair per trial
    % Here we use the observed MI from the main analysis.
    % The IBS value for each trial is taken as the maximum MI within a range.
     
    % if string(band) == "beta"
    %     win_idx = 95:107;
    % else
    %     win_idx = 95:107;
    % end
    win_idx = 95:107;
    
    results_around_zero = squeeze(max(final_results.(band).normalized(:, win_idx, :, :), [], 2));
        
    % For each trial, obtain the max MI across surrogate lags.
    lagged_surr_stats = squeeze(max(normalized_surrs, [], 2));  % dims: [Nblock, Nch, Nch, Nsurrogate]
    p_values = nan(Nblock, Nch, Nch);
    for bi = 1:Nblock
        for chi = 1:Nch
            for cha = 1:Nch
                observed = results_around_zero(bi, chi, cha);
                surrogate_distribution = squeeze(lagged_surr_stats(bi, chi, cha, :));
                p_values(bi, chi, cha) = sum(surrogate_distribution >= observed) / Nsurrogate;
            end
        end
    end
    
    
    %% Combine surrogate distributions across trials for each condition
    conditions = {'Listening', 'Low Interest A Speaks', 'Low Interest B Speaks', ...
                  'High Interest A Speaks', 'High Interest B Speaks', 'Silence'};
    cond_indices = {listening_idx, intersect(Aspeaks, low_interest_idx), intersect(Bspeaks, low_interest_idx), ...
                    intersect(Aspeaks, high_interest_idx), intersect(Bspeaks, high_interest_idx), silence_idx};
    p_combined_conditions = cell(1, numel(conditions));
    
    for c = 1:numel(conditions)
        idx = cond_indices{c};  % trial indices for the condition
        p_combined = nan(Nch, Nch);
        for i = 1:Nch
            for j = 1:Nch
                MI_obs_mean = mean(results_around_zero(idx, i, j));
                surrogate_values = squeeze(mean(lagged_surr_stats(idx, i, j, :), 1));
                p_combined(i, j) = sum(surrogate_values >= MI_obs_mean) / numel(surrogate_values);
            end
        end
        p_combined_conditions{c} = p_combined;
    end
    
    %% Plot combined p-values as heatmaps for each condition
    new_order = [frontal_idx, left_temporal_idx, central_idx, right_temporal_idx, parietal_idx, occipital_idx];
    channels_ordered      = channels(new_order);
    sig_threshold = 0.01;
    figure; clf;
    for c = 1:numel(conditions)
        subplot(2,3,c);
        p_mat = p_combined_conditions{c};
        p_sig = p_mat;
        p_sig(p_mat >= sig_threshold) = 1;
        imagesc(p_sig);
        colormap('jet');
        colorbar;
        caxis([0 sig_threshold+0.01]);
        set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
        set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', channels_ordered);
        title(sprintf('%s (p < %.2f) [%s]', conditions{c}, sig_threshold, band));
        xlabel('Subject B Channels');
        ylabel('Subject A Channels');
    end

    %% EXPORT SURROGATE DATA TO CSV (R-ready) (untested)
    % Flatten the 5D normalized surrogate results into a table.
    channels = [sub_B{1,7}];

    [trialIdx, lagIdx, chanAIdx, chanBIdx, surrIdx] = ndgrid(trial_idx', 1:Nlags, channels, channels, 1:Nsurrogate);
    trial_flat = trialIdx(:);
    lag_flat   = current_lags(lagIdx(:))';  % actual lag values (in samples; convert by dividing by eeg_fs if desired)
    chanA_flat = chanAIdx(:);
    chanB_flat = chanBIdx(:);
    surr_flat  = surrIdx(:);
    MI_flat    = normalized_surrs(:);
    
    T_surr = table(trial_flat, lag_flat, chanA_flat, chanB_flat, surr_flat, MI_flat, ...
                   'VariableNames', {'Trial','Lag','Channel_A','Channel_B','Surrogate','MI'});
    % Append frequency band info:
    T_surr.FrequencyBand = repmat({band}, height(T_surr), 1);
    
    csv_filename = fullfile('output', [subId '_' sprintf('%s_surrogate_MI_results_v3.csv', band)]);
    writetable(T_surr, csv_filename);
    fprintf('Exported surrogate CSV: %s\n', csv_filename);
    
end

%% Incorporate CCOR calculation

final_circ_results = struct();

for b = 1:numBands
    % Extract  data matrices 
    split_A = final_out_brainA{b}(:, 1);
    split_B = final_out_brainB{b}(:, 1);
    
    % info
    Nblock = numel(split_A);
    Nch = size(split_A{1}, 1)-1;
    
    if b == 2
        lags = -100:1:100;
    else
        lags = -400:4:400;
    end
    Nlags = length(lags);
    L_max = max(abs(lags));
    
    % Preallocate circular correlation results matrix
    % Dimensions: [trial, lag, channel_A, channel_B]
    circ_results = nan(Nblock, Nlags, Nch, Nch);
    
    % Begin processing
    parfor bi = 1:Nblock
        data_A = split_A{bi}(1:33,:);  % [channels x time]
        data_B = split_B{bi}(1:33,:);  % [channels x time]
        
        % --- STEP 1: Compute instantaneous phase for each channel ---
        % Note: hilbert expects the data in columns
        phase_A = angle(hilbert(data_A')');  % Result: [channels x time]
        phase_B = angle(hilbert(data_B')');
        
        % --- STEP 2: Wrap phases to [0, 2*pi] ---
        phase_A = wrapTo2Pi(phase_A);
        phase_B = wrapTo2Pi(phase_B);
        
        % Loop overlags
        for li = 1:Nlags
            l = lags(li);
            
            % Align phase signals between subjects A and B based on the current lag.
            if l == 0
                samples2trim = L_max;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end   = size(phase_A, 2) - floor(samples2trim/2);
                phaseA_lag = phase_A(:, idx_start:idx_end);
                phaseB_lag = phase_B(:, idx_start:idx_end);
            elseif l < 0
                lag_abs = abs(l);
                samples2trim = L_max - lag_abs;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end   = size(phase_A, 2) - floor(samples2trim/2);
                phaseA_seg = phase_A(:, idx_start:idx_end);
                phaseB_seg = phase_B(:, idx_start:idx_end);
                phaseA_lag = phaseA_seg(:, 1:end - lag_abs);
                phaseB_lag = phaseB_seg(:, lag_abs + 1:end);
            else  % l > 0
                samples2trim = L_max - l;
                idx_start = ceil(samples2trim/2) + 1;
                idx_end   = size(phase_A, 2) - floor(samples2trim/2);
                phaseA_seg = phase_A(:, idx_start:idx_end);
                phaseB_seg = phase_B(:, idx_start:idx_end);
                phaseA_lag = phaseA_seg(:, l + 1:end);
                phaseB_lag = phaseB_seg(:, 1:end - l);
            end
            
            % --- STEP 3: Compute circular correlation for each channel pair ---
            for chi = 1:Nch
                for cha = 1:Nch
                    % circ_corrcc computes the circular correlation coefficient between two vectors.
                    circ_results(bi, li, chi, cha) = circ_corrcc(phaseA_lag(chi, :)', phaseB_lag(cha, :)');
                end
            end
        end
        disp(['Circular correlation, Frequency Band ' bandNames{b} ', Block ' num2str(bi) '/' num2str(Nblock) ' done!']);
    end
    % Store results for current frequency band
    final_circ_results.(bandNames{b}) = circ_results;
end

disp('Inter-Brain Circular Correlation Computation Completed for All Frequency Bands');

%% Export: Create an R-ready CSV file

channels = [sub_B{1,7}];

tmp = final_out_brainA{1}(:,2:4);
trial_idx = {};
for i = 1:size(tmp, 1)
    trial_idx{i} = [char(tmp{i,2}) '_' char(tmp{i,1}) '_' num2str(tmp{i,3})];
end

for b = 1:numBands
    % Retrieve the circular correlation data for the current band.
    % Dimensions: [trial, lag, channel_A, channel_B]
    circ_data = final_circ_results.(bandNames{b});
    [Nblock, Nlags, Nch, ~] = size(circ_data);
    
    % Define the lags (same as used in the computation section)
    if b == 2
        lags = -100:1:100;
    else
        lags = -400:4:400;
    end
    
    % Create grids of indices for trials, lags, and channels.
    [trialIdx, lagIdx, chanAIdx, chanBIdx] = ndgrid(trial_idx, 1:Nlags, channels, channels);
    
    % Flatten the indices and circular correlation values.
    trial_flat = trialIdx(:);
    lag_flat   = lags(lagIdx(:))';
    chanA_flat = chanAIdx(:);
    chanB_flat = chanBIdx(:);
    circCorr_flat = circ_data(:);
    
    % Create a table for export.
    T = table(trial_flat, lag_flat, chanA_flat, chanB_flat, circCorr_flat, ...
        'VariableNames', {'Trial', 'Lag', 'Channel_A', 'Channel_B', 'CircCorr'});
    
    % Add a frequency band column.
    T.FrequencyBand = repmat({bandNames{b}}, height(T), 1);
    
    % filename
    subID = file_names{1}(15:22);
    csv_filename = fullfile('output', [subID '_' bandNames{b} '_circCorr_results.csv']);
    
    % Write the table to CSV.
    writetable(T, csv_filename);
    fprintf('Exported circular correlation CSV for %s Band: %s\n', bandNames{b}, csv_filename);
end

disp('Export of circular correlation results completed.');

%%
elapsedTime = toc;
fprintf('Main CCorr completed. Time elapsed: %.4f hours\n', elapsedTime/3600);

%% CCorr Surrogate Analysis
% surrogate_params.alpha.lags = -24:8:24;  
% surrogate_params.beta.lags  = -8:2:8;     
% surrogate_params.theta.lags = -24:8:24;    

for b = 1:numBands
    band = bandNames{b};
    % Get surrogate lag parameters for this band:
    current_lags = surrogate_params.(band).lags;
    Nlags = length(current_lags);
    lagtime = current_lags ./ eeg_fs;  
    L_max = max(abs(current_lags));
    Nsurrogate = 200;  % number of surrogate iterations
    
    % Get the preprocessed data for CCorr 
    split_A = final_out_brainA{b}(:,1);
    split_B = final_out_brainB{b}(:,1);
    Nblock = numel(split_A);
    % stimtrack we ignore (-1)
    Nch = size(split_A{1},1) - 1;
    
    % Preallocate the surrogate results array.
    % Dimensions: [trial, surrogate lag, channel_A, channel_B, surrogate iteration]
    results_sur_ccorr = nan(Nblock, Nlags, Nch, Nch, Nsurrogate);
    
    % start loop
    parfor bi = 1:Nblock
        % Extract data for Subject A and B (only EEG channels: rows 1:33)
        data_A = split_A{bi}(1:33, :);
        data_B = split_B{bi}(1:33, :);
        T = size(data_B,2);  
        
        % instantaneous phase for subject A (Hilbert transform)
        phase_A = wrapTo2Pi(angle(hilbert(data_A')'));
        
        for surr = 1:Nsurrogate
            % Apply a random circular shift to subject B data (starting at ~8 sec)
            shift_amount = randi([4000, T-4000]);
            data_B_shift = circshift(data_B, [0, shift_amount]);
            % instantaneous phase for shifted data
            phase_B = wrapTo2Pi(angle(hilbert(data_B_shift')'));
            
            for li = 1:Nlags
                l = current_lags(li);
                if l == 0
                    samples2trim = L_max;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(phase_A,2) - floor(samples2trim/2);
                    phaseA_lag = phase_A(:, idx_start:idx_end);
                    phaseB_lag = phase_B(:, idx_start:idx_end);
                elseif l < 0
                    lag_abs = abs(l);
                    samples2trim = L_max - lag_abs;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(phase_A,2) - floor(samples2trim/2);
                    A_segment = phase_A(:, idx_start:idx_end);
                    B_segment = phase_B(:, idx_start:idx_end);
                    phaseA_lag = A_segment(:, 1:end-lag_abs);
                    phaseB_lag = B_segment(:, lag_abs+1:end);
                else  % l > 0
                    samples2trim = L_max - l;
                    idx_start = ceil(samples2trim/2) + 1;
                    idx_end = size(phase_A,2) - floor(samples2trim/2);
                    A_segment = phase_A(:, idx_start:idx_end);
                    B_segment = phase_B(:, idx_start:idx_end);
                    phaseA_lag = A_segment(:, l+1:end);
                    phaseB_lag = B_segment(:, 1:end-l);
                end
                
                % Loop over channel pairs and compute circ_corrcc
                for chi = 1:Nch
                    for cha = 1:Nch
                        results_sur_ccorr(bi, li, chi, cha, surr) = circ_corrcc(phaseA_lag(chi, :)', phaseB_lag(cha, :)');
                    end
                end
            end
        end
        disp(['CCorr surrogate, Frequency band ' band ', trial ' num2str(bi) '/' num2str(Nblock) ' done!']);
    end
    
    % Extract the maximum observed CCorr over a fixed window of lags.
    win_idx = 95:107;  % around 0
    results_around_zero = squeeze(max(final_circ_results.(band)(:, win_idx, :, :), [], 2));  % [Nblock x Nch x Nch]
        
    % For each trial, get maximum surrogate CCorr across lags.
    lagged_surr_stats = squeeze(max(results_sur_ccorr, [], 2));  % [Nblock x Nch x Nch x Nsurrogate]
    p_values = nan(Nblock, Nch, Nch);
    for bi = 1:Nblock
        for chi = 1:Nch
            for cha = 1:Nch
                observed = results_around_zero(bi, chi, cha);
                surrogate_distribution = squeeze(lagged_surr_stats(bi, chi, cha, :));
                p_values(bi, chi, cha) = sum(surrogate_distribution >= observed) / Nsurrogate;
            end
        end
    end
    
    %% Combine surrogate distributions across trials for each condition
    conditions = {'Listening', 'Low Interest A Speaks', 'Low Interest B Speaks', ...
                  'High Interest A Speaks', 'High Interest B Speaks', 'Silence'};
    cond_indices = {listening_idx, intersect(Aspeaks, low_interest_idx), intersect(Bspeaks, low_interest_idx), ...
                    intersect(Aspeaks, high_interest_idx), intersect(Bspeaks, high_interest_idx), silence_idx};
    p_combined_conditions = cell(1, numel(conditions));
    
    for c = 1:numel(conditions)
        idx = cond_indices{c};  
        p_combined = nan(Nch, Nch);
        for i = 1:Nch
            for j = 1:Nch
                ccorr_obs_mean = mean(results_around_zero(idx, i, j));
                surrogate_values = squeeze(mean(lagged_surr_stats(idx, i, j, :), 1));
                p_combined(i, j) = sum(surrogate_values >= ccorr_obs_mean) / numel(surrogate_values);
            end
        end
        p_combined_conditions{c} = p_combined;
    end
    
    %% Plot combined p-values as heatmaps for each condition
    new_order = [frontal_idx, left_temporal_idx, central_idx, right_temporal_idx, parietal_idx, occipital_idx];
    channels_ordered = channels(new_order);
    sig_threshold = 0.01;
    figure; clf;
    for c = 1:numel(conditions)
        subplot(2,3,c);
        p_mat = p_combined_conditions{c};
        p_sig = p_mat;
        p_sig(p_mat >= sig_threshold) = 1;
        imagesc(p_sig);
        colormap('jet');
        colorbar;
        caxis([0 sig_threshold+0.01]);
        set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
        set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', channels_ordered);
        title(sprintf('%s (p < %.2f) [%s]', conditions{c}, sig_threshold, band));
        xlabel('Subject B Channels');
        ylabel('Subject A Channels');
    end

    %% EXPORT SURROGATE DATA TO CSV (R-ready)
    % Flatten the 5D surrogate CCorr results into a table.
    [trialIdx, lagIdx, chanAIdx, chanBIdx, surrIdx] = ndgrid(trial_idx', 1:Nlags, channels, channels, 1:Nsurrogate);
    trial_flat = trialIdx(:);
    lag_flat = current_lags(lagIdx(:))';  % actual lag values (in samples)
    chanA_flat = chanAIdx(:);
    chanB_flat = chanBIdx(:);
    surr_flat = surrIdx(:);
    CCorr_flat = results_sur_ccorr(:);
    
    T_surr = table(trial_flat, lag_flat, chanA_flat, chanB_flat, surr_flat, CCorr_flat, ...
                   'VariableNames', {'Trial','Lag','Channel_A','Channel_B','Surrogate','CCorr'});
    T_surr.FrequencyBand = repmat({band}, height(T_surr), 1);
    
    csv_filename = fullfile('output', [subId '_' sprintf('%s_surrogate_CCorr_results_.csv', band)]);
    writetable(T_surr, csv_filename);
    fprintf('Exported surrogate CCorr CSV: %s\n', csv_filename);
end

disp('Surrogate Circular Correlation Analysis Completed for All Frequency Bands');
elapsedTime = toc;
fprintf('Whole processing completed. Time elapsed: %.4f hours\n', elapsedTime/3600);