clear, clc, close all

% Navigate to the root folder of the data
tmp = matlab.desktop.editor.getActive; % Get the current active file
current_dir = fileparts(tmp.Filename); % Get the directory of the current file
up_one_level = fullfile(current_dir, '..');
eeg_dir = fullfile(up_one_level);
cd(fullfile(eeg_dir)); 

% Select files from raw files to be processed
[file_names, file_paths] = uigetfile('05_Merged\\*.set',  'set Files (*.set*)','MultiSelect','on');

% Get sub and cond str
dyadID = strrep(file_names, '_merged.set', '');

EEG = pop_loadset(file_names, file_paths);

%% Add offline markers and replace StimTrack samples

% Part 1: Fetch conversation event latencies
selectedEvents = struct('type', {}, 'latency', {});
index = 1;

% Loop through all events
for i = 1:length(EEG.event)
    evType = EEG.event(i).type;
    
    % Check if the event type starts with 'S'
    if startsWith(evType, 'S')
        % Use a regular expression to extract the numeric part
        numStr = regexp(evType, '\d+', 'match');
        if ~isempty(numStr)
            numVal = str2double(numStr{1});
            % Check if the numeric value is less than 50
            if numVal < 50
                selectedEvents(index).type = numVal;
                selectedEvents(index).latency = EEG.event(i).latency;
                index = index + 1;
            end
        end
    end
end

% Part 2: Organize in a tidy format
tidyEvents = struct('type', {}, 'n', {}, 'code', {}, 'start', {}, 'end', {});
index = 1;

possible_types = ["low_interest", "high_interest"];

blocks = [1 1 2 2 3 3 4 4 5 5 6 6];

for i = 1:12
    block = blocks(i);
    curr_event = selectedEvents(index).type;
    type = (curr_event > 10 && curr_event < 20) + 1;
    tidyEvents(i).type = possible_types(type);
    tidyEvents(i).n = block;
    tidyEvents(i).code  = curr_event;
    tidyEvents(i).start = selectedEvents(index).latency;
    tidyEvents(i).end = selectedEvents(index+1).latency;
        
    if ~selectedEvents(index+1).type == curr_event + 10
        error('Start and end event numbers do not match during iteration %d. Stopping.', i);
    end

    index = index + 2;
end

% Part 3: fetch audio data, find point of highest cross correlation with
% each trial, and add offline markers based on csv

% eventData = readtable('00_Event_Data\pilot_02\low_interest_1_knitting.csv');
N = size(tidyEvents,2);
dnum = dyadID(end-1:end);
csvDir = ['00_Event_Data\dyad_' dnum '\'];
wavDir = ['00_Clean_Audio_Files\dyad' dnum '\'];

% is chan 33 the stimrtack?
if ~contains(EEG.chanlocs(33).labels, 'Aux')
    error('Could not find Stimtrack');
end

for i = 1:12 % should be N
    % Set partial filename
    partialFileName = [char(tidyEvents(i).type) '_' num2str(tidyEvents(i).n) '*'];

    % Load Audio
    audioFileName = dir(fullfile(wavDir, partialFileName)).name;
    [wav, fs] = audioread(fullfile(wavDir, audioFileName));
    
    % Resample
    wavt = wav';
    wav_resampled = resample(wavt, 500, 44100);               % Resample to 500 Hz
    wav_resampled = wav_resampled - mean(wav_resampled);      % Remove mean
    wav_resampled = wav_resampled / max(abs(wav_resampled));  % Normalize
    
    % Compute amplitude envelope
    wav_env = mTRFenvelope(wav, fs, 500)'; 

    % Fetch Stimtrack 
    start_lat = tidyEvents(i).start;
    end_lat = tidyEvents(i).end;

    stimtrack = EEG.data(33, start_lat:end_lat);
    stimtrack = stimtrack - mean(stimtrack);                   % Remove mean
    stimtrack = stimtrack / max(abs(stimtrack));               % Normalize

    % Adjust the length of wav if necessary
    % The envelope may be one sample shorter than the waveform
    if length(wav_resampled) > length(wav_env)
        l = length(wav_resampled);
        wav_resampled = wav_resampled(1:l-1);
    end

    % Compute cross-correlation between EEG audio channel and audio waveform
    [r, lags] = xcorr(stimtrack, wav_resampled);

    % Create figure
    figure;
    plot(lags, r);
    xlabel('Lags');
    ylabel('r');
    title(sprintf('Iteration %d: Accept?', i));

    % Add an "Accept" button
    btn = uicontrol('Style', 'pushbutton', 'String', 'Accept', ...
        'Position', [20 20 100 40], ...
        'Callback', 'uiresume(gcbf)');

    % Wait for user confirmation
    uiwait(gcf);

    % Close the figure after confirmation
    close(gcf);

    % Continue with the loop
    disp(['Iteration ' num2str(i) ' accepted.']);

    % Find the lag with the maximum absolute correlation
    [larger_r, idx] = max(abs(r));
    shift = lags(idx);
    
    % A positive shift means the WAV is delayed with respect to the
    % StimTrack data. Thus, my event markers should be at T + shift.
    
    % Replace StimTrack data with Amplitude Envelope
    st_length = length(stimtrack);
    wv_length = length(wav_env);
    if shift > 0
        if shift + wv_length > st_length
            envLim = wav_env(1:(st_length-shift+1)); % trim envelope
            stimtrack(shift:end) = envLim;           % replace data with envelope
        else
%             warning('This else statement has not been tested, at iteration %d.', i);
            stimtrack(shift:(shift+wv_length-1)) = wav_env; % (warning: untested)
        end
    else
        envLim = wav_env(abs(shift):end);
        if length(envLim) > st_length
            stimtrack(1:length(stimtrack)) = envLim(1:length(stimtrack)); 
        else
            warning('This else statement has not been tested, at iteration %d.', i);
            stimtrack(1:length(envLim)) = envLim; % (warning: untested)
        end
    end
    
    EEG.data(33, start_lat:end_lat) = stimtrack;
    
    % Load Praat TextGrid Data
    fileName = dir(fullfile(csvDir, partialFileName));

    if isempty(fileName)
        error('No files found matching the pattern: %s', partialFileName);
    else
        eventData = readtable(fullfile(csvDir, fileName.name));
    end

    % Transform text grid (keep relevant rows and convert latencies)
    % 1. initialize struct
    speakingLatencies = struct('speaker', {}, 'start', {}, 'end', {});
    
    % 2. filter to keep only rows of speaking turns
    keep = {'SpeakerA', 'SpeakerB'};
    matches = false(height(eventData), 1);
    for j = 1:length(keep)
         matches = matches | contains(eventData.tier, keep{j});
    end

    filteredEvents = eventData(matches, :);
    
%     if shift < 0
%         error("Shift is negative. Figure out what to do before continuing");
%     end

    % 3. Loop through rows and populate new struct
    for j = 1:height(filteredEvents)
        speakingLatencies(j).speaker = char(filteredEvents{j,"tier"});
        speakingLatencies(j).start   = filteredEvents{j,"tmin"} * 500 + shift; % Warning: assuming positive shift
        speakingLatencies(j).end     = filteredEvents{j,"tmax"} * 500 + shift;
    end
    
    startsb40 = find([speakingLatencies.start]<0);
    endsb40 = find([speakingLatencies.end]<0);

    if ~isempty(startsb40)
        if length(startsb40) > 1
            error('Multiple events start befoer 0. Figure out how to handle this before continuing.');
        else
            speakingLatencies(startsb40).start = 0;
        end
    end

    if ~isempty(endsb40)
        error('An event ends before 0. Figure out how to handle this before continuing.')
        % probably just delete that row.
    end

    % Add event markers to the EEG set
    for j = 1:size(speakingLatencies,2)
        spkr = speakingLatencies(j).speaker;

        % Start marker
        marker_latency_start = speakingLatencies(j).start; 
        if spkr == "SpeakerA"
            newMarker.type = 'S 91';   
        elseif spkr == "SpeakerB"
            newMarker.type = 'S 95';
        else
            error('Something went wrong.')
        end

        newMarker.latency = start_lat + marker_latency_start;
        newMarker.duration = 1; % is this ok?
        newMarker.channel = 0;
        newMarker.bvmknum = [];
        newMarker.visible = [];
        newMarker.code = 'Stimulus';
        newMarker.urevent = [];
        newMarker.bvtime = [];

        EEG.event(end+1) = newMarker;

        % End marker
        marker_latency_end = speakingLatencies(j).end; 
        if spkr == "SpeakerA"
            newMarker.type = 'S 92';   
        elseif spkr == "SpeakerB"
            newMarker.type = 'S 96';
        else
            error('Something went wrong.')
        end

        newMarker.latency = start_lat + marker_latency_end;
        newMarker.duration = 1; % is this ok?
        newMarker.channel = 0;
        newMarker.bvmknum = [];
        newMarker.visible = [];
        newMarker.code = 'Stimulus';
        newMarker.urevent = [];
        newMarker.bvtime = [];

        EEG.event(end+1) = newMarker;
    end
end

% Update the EEG structure (this ensures event consistency)
EEG = eeg_checkset(EEG, 'eventconsistency');

%% delete unevent data

oneSec = EEG.srate;

% Initialize matrix to store rejection intervals.
% Each row will be [startSample, endSample]
rejIntervals = [];

% number of events
numEvents = length(EEG.event);

for i = 1:numEvents
    % Extract the numeric part of the event type.
    % For example, if the event type is 'T 25' or 'S 41'
    evTypeStr = EEG.event(i).type;
    numStr = regexp(evTypeStr, '\d+', 'match');
    if isempty(numStr)
        continue; % skip if no number found
    end
    evVal = str2double(numStr{1});
    
    % Check if this event is marks the end of a trial
    if (evVal >= 20 && evVal <= 29) || ...
       (evVal >= 40 && evVal <= 49) || ...
       (evVal >= 60 && evVal <= 69) || ...
       (evVal >= 80 && evVal <= 89)

        % Define the start of the rejection interval:
        startRej = EEG.event(i).latency + oneSec;  % one second AFTER the trigger
        
        % Now look for the next event that marks the beginning of a trial:
        foundBoundary = false;
        nextIdx = i + 1;
        while nextIdx <= numEvents
            evTypeStr2 = EEG.event(nextIdx).type;
            numStr2 = regexp(evTypeStr2, '\d+', 'match');
            if ~isempty(numStr2)
                evVal2 = str2double(numStr2{1});
                if (evVal2 >= 10 && evVal2 <= 19) || ...
                   (evVal2 >= 30 && evVal2 <= 39) || ...
                   (evVal2 >= 50 && evVal2 <= 59) || ...
                   (evVal2 >= 70 && evVal2 <= 79)
                    % Found one.
                    endRej = EEG.event(nextIdx).latency - oneSec; % one second BEFORE this event
                    foundBoundary = true;
                    break;
                end
            end
            nextIdx = nextIdx + 1;
        end
        
        % If good, save the interval.
        if foundBoundary && endRej > startRej
            rejIntervals = [rejIntervals; [startRej, endRej]];
        end
    end
end

% Display the intervals to be rejected (in sample points)
disp('Rejection intervals (in samples):');
disp(rejIntervals);

% Now automatically reject those segments from the EEG data.
% eeg_eegrej takes a matrix with rows of [start end] sample indices.
EEG = eeg_eegrej(EEG, rejIntervals);

disp(' # - - - - - Unevent deletion done. - - - - - #')


%% Step 06 Manual Masking

ALLEEG = EEG;
CURRENTSET = 1;

% Change set name
EEG = pop_editset(EEG, 'setname', strcat(dyadID, '_MR'));

% Manual masking
eegplot(EEG.data([1:32 34:end],:), 'srate', EEG.srate, 'title', 'Manual masking (Click Accent And Close when Finished!)', 'events', EEG.event);
uiwait(gcf);

% Extract only the start and end times from TMPREJ
maskedIntervals = TMPREJ(:,1:2);

% Store mask in EEG structure 
EEG.maskedIntervals = maskedIntervals;

% Wait to continue after check...
input('Done manual masking for this subject? Press enter to save')

% Save dataset
EEG = pop_editset(EEG, 'setname', strcat(dyadID, '_merged'));
EEG = pop_saveset(EEG, strcat(dyadID, '_MR.set'), '06_MR\\');