% Initiate eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Navigate to the root folder of the data
tmp = matlab.desktop.editor.getActive; % Get the current active file
current_dir = fileparts(tmp.Filename); % Get the directory of the current file
up_one_level = fullfile(current_dir, '..');
eeg_dir = fullfile(up_one_level);
cd(fullfile(eeg_dir)); 

% Select files from raw files to be processed
file_names = cellstr(uigetfile('00_Raw\\*.vhdr',  'vhdr Files (*.vhdr*)','MultiSelect','on'));
   
filename = char(file_names(2)); % Change from 1 to 2 and run all below again
 % Get sub and cond str
subID = strrep(filename, '.vhdr', '');

%% Steps 01-02 Processing raw data

EEG = pop_loadbv('00_Raw\\', filename, []);
  
% Convert to SET file and save
EEG = pop_saveset(EEG, strcat(subID, '_SET.set'), '01_SET\\');

%% Step 03 Add channel locations

% Select files from raw files to be processed, if SET already exists
% [file_name, file_path] = uigetfile('*.set', 'Select subject (*.set*)', 'MultiSelect', 'off');
% EEG = pop_loadset(file_name, file_path);
% subID = file_name(1:(strfind(file_name, '_SET') - 1));

% Lookup for channel locations
EEG = pop_chanedit( ...
    EEG, ...
    'lookup', 'C:\Program Files\eeglab2025.0.0\plugins\dipfit\standard_BEM\elec\standard_1005.elc');

% Plot channel spectra and maps (make sure channels are loaded properly and note for abnormal channels)
figure; pop_spectopo(EEG, 1, [EEG.xmin  EEG.xmax], 'EEG' , 'freq', [6 10 60], 'freqrange', [2 80], 'electrodes', 'off');

% Just make sure there are location details for each electrode, no changes
% needed usually
pop_chanedit(EEG);

% Wait to continue after check...
input('Checked channel locations for this subject? Press enter to continue')

% Save data with channel location checked
EEG = pop_editset(EEG, 'setname', strcat(subID, '_chan'));
EEG = pop_saveset(EEG, strcat(subID, '_chan.set'), '02_Chan\\');

%% Step 4a Clean Line
nc = EEG.nbchan;
EEG = pop_cleanline(EEG, 'Bandwidth',3,'ChanCompIndices', 1:nc, 'SignalType','Channels','ComputeSpectralPower',false,'LineFrequencies',[60 120] ,'NormalizeSpectrum',false,'LineAlpha',0.01,'PaddingFactor',2,'PlotFigures',false,'ScanForLines',true,'SmoothingFactor',300,'VerbosityLevel',1,'SlidingWinLength',2,'SlidingWinStep',1);

%% Step 4b Filtering

% Filtering (remove high- and low-frequency noise)
EEG = pop_basicfilter( ...
    EEG, ...
    1:32, ...
    'Boundary', 'boundary', ...
    'Cutoff', [1 40], ...
    'Design', 'butter', ...
    'Filter', 'bandpass', ...
    'Order', 4, ...
    'RemoveDC', 'on' ...
    );

% Save dataset
EEG = pop_editset(EEG, 'setname', strcat(subID, '_filt'));
EEG = pop_saveset(EEG, strcat(subID, '_filt.set'), '03_Filtered\\');

%% Step 05 Interpolate bad channels

completed = 0;

while completed == 0

    % Plot for detecting bad channels
    pop_eegplot(EEG, 1, 1, 1);

    % Ask for which channels to remove
    rmchannel = input("Which channels are you interpolating, if any?\n" + ...
        "Enter in this format, without quotes, '{'Fp1','Fz','F3','F7'}'.\n" + ...
        "If none, just press enter:\n");

    % Check if input is empty
    if isempty(rmchannel)

        % Ask for confirmation
        confirm = input('You did not enter any channels. Is this correct? (y/n)\n', 's');

        % Only proceed if confirmed
        if confirm == 'y'
            disp('No channels to remove.');
            completed = 1;
        else
            disp("Let's try again!");
        end

    else

        % Ask for confirmation
        confirm = input(strcat('You entered: ', strjoin(string(rmchannel), ', '), '. Is this correct? (y/n)\n'), 's');

        % Only proceed if confirmed
        if confirm == 'y'

            rm_idx = find(ismember({EEG.chanlocs.labels}, rmchannel));
            EEGtemp = pop_interp(EEG, rm_idx, 'spherical');
            
            % Ask for confirmation
            confirm = input('Does the interpolation look ok? (y/n)\n', "s");
            if confirm == 'y'
                EEG = EEGtemp; % Confirm the temporarily removed channels
                completed = 1;
            else
                disp("Let's try again!");
            end

        else
            disp("Let's try again!");
        end
    end
end

% Save dataset
EEG = pop_editset(EEG, 'setname', strcat(subID, '_rmChan'));
EEG = pop_saveset(EEG, strcat(subID, '_rmChan.set'), '04_rmChan\\');

%% Step 05 Remove bad channels
% 
% completed = 0;
% 
% while completed == 0
% 
%     % Plot for detecting bad channels
%     pop_eegplot(EEG, 1, 1, 1);
% 
%     % Ask for which channels to remove
%     rmchannel = input("Which channels are you removing, if any?\n" + ...
%         "Enter in this format, without quotes, '{'Fp1','Fz','F3','F7'}'.\n" + ...
%         "If none, just press enter:\n");
% 
%     % Check if input is empty
%     if isempty(rmchannel)
% 
%         % Ask for confirmation
%         confirm = input('You did not enter any channels. Is this correct? (y/n)\n', 's');
% 
%         % Only proceed if confirmed
%         if confirm == 'y'
%             disp('No channels to remove.');
%             completed = 1;
%         else
%             disp("Let's try again!");
%         end
% 
%     else
% 
%         % Ask for confirmation
%         confirm = input(strcat('You entered: ', strjoin(string(rmchannel), ', '), '. Is this correct? (y/n)\n'), 's');
% 
%         % Only proceed if confirmed
%         if confirm == 'y'
%             EEGtemp = pop_select(EEG, 'nochannel', rmchannel);
%             
%             % Ask for confirmation
%             confirm = input('Does the number of channels match what you input? (y/n)\n', "s");
%             if confirm == 'y'
%                 EEG = EEGtemp; % Confirm the temporarily removed channels
%                 completed = 1;
%             else
%                 disp("Let's try again!");
%             end
% 
%         else
%             disp("Let's try again!");
%         end
%     end
% end
% 
% % Save dataset
% EEG = pop_editset(EEG, 'setname', strcat(subID, '_rmChan'));
% EEG = pop_saveset(EEG, strcat(subID, '_rmChan.set'), '04_rmChan\\');
