function [epoched_data] = epoch_hyperscanning_data_byspeaker(set_name, onset_marker, offset_marker)
   
    % Load the EEG dataset
    datapath = 'D:\marcosOneDrive\OneDrive - McGill University\Study Materials\3_processing\ready_data_v0';
    EEG = pop_loadset('filename', set_name, 'filepath', datapath);
    
    % List of substrings to match for removal
    drop_substrings = {'T1', 'boundary'};
    
    % Get all codelabels from EEG.event
    codelabels = {EEG.event.codelabel};
    
    % Initialize logical array to mark events for removal
    remove_indices = false(1, length(codelabels));
    
    % Loop over each substring and find matching events
    for i = 1:length(drop_substrings)
        substring = drop_substrings{i};
        
        % For MATLAB R2016b and newer, use 'contains'
%         matches = contains(codelabels, substring);
        matches = strcmp(codelabels, substring);
        
        % Update the indices of events to remove
        remove_indices = remove_indices | matches;
    end
    
    % Indices of events to keep
    keep_indices = ~remove_indices;
    
    % Keep only the desired events in EEG.event
    EEG.event = EEG.event(keep_indices);
    
    % If urevent field exists, update it as well
    if ~isempty(EEG.urevent)
        EEG.urevent = EEG.urevent(keep_indices);
    end
    
    % Get the number of events
    n_events = length(EEG.event);

    % Save the mask
    mask = EEG.maskedIntervals;
    
    % Initialize the output cell array and counters
    epoched_data = {};
    w = 1;
    trials_added = 0;
    segments_invalid = 0;

    % Loop through events
    for i = 1:n_events-1
    
        % Current and next events
        currEvent = EEG.event(i);
        afterEvent = EEG.event(i+1);
        
        % Event types (labels)
        curr_type = currEvent.codelabel;
        next_type = afterEvent.codelabel;
        
        % Check if current and next events are matching
        isCurrentOk  = contains(curr_type, onset_marker);
        isNextOk     = contains(next_type, offset_marker);
        
        if (isCurrentOk + isNextOk) < 2
            segments_invalid = segments_invalid + 1;
            continue;
        end
        
        % Check if the current and previous events form a complete trial
       
            % Define the start and end of the epoch, adding a 200-sample buffer
            trial_start = currEvent.latency;
            trial_end   = afterEvent.latency;
            
            % Extract the epoch data from the EEG
            epoch = EEG.data(:, trial_start:trial_end);
            
            % Store the epoch data and trial type in the output cell array
            epoched_data{w, 1} = epoch;
            epoched_data{w, 2} = curr_type;
            epoched_data{w, 3} = string({EEG.chanlocs.labels})';

            % Increment the counter
            w = w + 1;
            trials_added = trials_added + 1;
    end

% Display the summary of trials processed
disp(['Total trials added: ', num2str(trials_added)]);
disp(['Total ignored events: ', num2str(segments_invalid)]);
end