clear, clc, close all

% Select files from raw files to be processed
[file_names, file_paths] = uigetfile('06_MR\\*.set',  'set Files (*.set*)','MultiSelect','on');

% Get sub and cond str
dyadID = strrep(file_names, '_MR.set', '');

% Load file
EEG = pop_loadset(file_names, file_paths);

% Separate into two distinct EEG objects
% Which?
whichA = find(startsWith({EEG.chanlocs.labels}, 'A_'));
whichB = find(startsWith({EEG.chanlocs.labels}, 'B_'));

% Fetch.
EEG_A = pop_select(EEG, 'channel', whichA);
EEG_B = pop_select(EEG, 'channel', whichB);

% Remove prefix from EEG_A channel labels
for i = 1:length(EEG_A.chanlocs)
    EEG_A.chanlocs(i).labels = EEG_A.chanlocs(i).labels(3:end);
end

% Remove prefix from EEG_B channel labels
for i = 1:length(EEG_B.chanlocs)
    EEG_B.chanlocs(i).labels = EEG_B.chanlocs(i).labels(3:end);
end

% Save
EEG_A = pop_saveset(EEG_A, strcat(dyadID, '_MR_A.set'), '07_MR_Fix\\');
EEG_B = pop_saveset(EEG_B, strcat(dyadID, '_MR_B.set'), '07_MR_Fix\\');

%% Run ICA (AMICA?)
% Here Han uses a separate dataset for ICA and then adds the ica data to
% the original dataset. I don't think we need that for now.
clear, clc, close all
% FIRST DATASET
[file_names, file_paths] = uigetfile('07_MR_Fix\\*.set',  'set Files (*.set*)','MultiSelect','on');
EEG_A = pop_loadset(file_names, file_paths);
% EEG_A_amica = pop_runamica(EEG_A); % :(

% Get sub and cond str
dyadID = strrep(file_names, '_MR_A.set', '');

% Convert masks to sample indices:
rejIntervalsSamples = round(EEG_A.maskedIntervals);

% Create a new dataset with the masked intervals removed.
EEG_A_ica = eeg_eegrej(EEG_A, rejIntervalsSamples);

% Run ICA
EEG_A_ica = pop_runica(EEG_A_ica, 'icatype', 'runica', 'chanind', 1:32, 'extended', 1, 'rndreset','yes', 'interrupt','on');

EEG_A.icaweights = EEG_A_ica.icaweights;
EEG_A.icasphere  = EEG_A_ica.icasphere;
EEG_A.icawinv    = EEG_A_ica.icawinv;
EEG_A.icachansind = EEG_A_ica.icachansind;

EEG_A = pop_editset(EEG_A, 'setname', strcat(dyadID, '_ICA_A'));
EEG_A = pop_saveset(EEG_A, strcat(dyadID, 'ICA_A.set'), '08_ICA\\');

%% SECOND DATASET
clear, clc, close all

[file_names, file_paths] = uigetfile('07_MR_Fix\\*.set',  'set Files (*.set*)','MultiSelect','on');
EEG_B = pop_loadset(file_names, file_paths);

% Get sub and cond str
dyadID = strrep(file_names, '_MR_B.set', '');

% Convert masks to sample indices:
rejIntervalsSamples = round(EEG_B.maskedIntervals);

% Create a new dataset with the masked intervals removed.
EEG_B_ica = eeg_eegrej(EEG_B, rejIntervalsSamples);

% Run ICA
EEG_B_ica = pop_runica(EEG_B_ica, 'icatype', 'runica', 'chanind', 1:32, 'extended', 1, 'rndreset','yes', 'interrupt','on');

EEG_B.icaweights = EEG_B_ica.icaweights;
EEG_B.icasphere  = EEG_B_ica.icasphere;
EEG_B.icawinv    = EEG_B_ica.icawinv;
EEG_B.icachansind = EEG_B_ica.icachansind;

EEG_B = pop_editset(EEG_B, 'setname', strcat(dyadID, '_ICA_B'));
EEG_B = pop_saveset(EEG_B, strcat(dyadID, '_ICA_B.set'), '08_ICA\\');

% Edit: Is the removed channel to be restored here?