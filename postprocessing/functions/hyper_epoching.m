function [epoched_data] = hyper_epoching(set_name)
   
    % Load the EEG dataset
    datapath = 'ready_data_v0';
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

    trimmed_labels = {EEG.event.codelabel};
    
    %% Part 1: Fetch listening trials
    
    tmp_counter = 0;

    for i = 1:6
        
        % Identify Listening events
        count_idx = contains(trimmed_labels, ['Listening_T' num2str(i)]);
        listening_idx = find(count_idx);
        
        % Check whether there is a beginning and end (i.e., two events of the type)
        if sum(count_idx) < 2
            warning(['Listening T' num2str(i) ' did not have 2 events.'])
            continue
        end
        
        % Fetch beginning and end latencies
        trial_start = EEG.event(listening_idx(1)).latency;
        trial_end = EEG.event(listening_idx(2)).latency;
        
        % Identify if any mask portion should be attached
        isMasked = mask >= trial_start & mask <= trial_end;
        
        % Check if any full mask segment falls in trial; store
        full_mask = sum(isMasked,2) == 2;
        full_mask_idx = full_mask;
        
        % Check if any mask segment partially falls in trial; store
        partial_mask = sum(isMasked,2) == 1;
        partial_idx  = find(partial_mask);
        
        % If there are partial segments, shift start/end of trial
        if sum(partial_mask) > 0
           for j = 1:numel(partial_idx)
                
               m  = mask(partial_idx(j),:);
               
               if m(1) < trial_start && m(2) < trial_end
                   
                   m(1) = trial_start+1;
               elseif m(2) > trial_end && m(1) > trial_start

                   m(2) = trial_end;
               end

               mask(partial_idx(j),:) = m; % modify mask to not exceed trial bounds

           end
        end

        % If there are full segments, scale and store them for later sue
        if sum(full_mask) > 0
            trial_mask = mask(full_mask_idx,:) - trial_start;
        else
            trial_mask = nan;
        end 

        % Extract the epoch data from the EEG
        epoch = EEG.data(:, trial_start:trial_end);
        
        % Store the data and trial type in the output cell array
        epoched_data{w, 1} = epoch;                   % epoch data
        epoched_data{w, 2} = "Listening";             % condition
        epoched_data{w, 3} = i;                       % counter
        epoched_data{w, 4} = trial_mask;              % mask info
        epoched_data{w, 5} = "none";                  % speaker A segments
        epoched_data{w, 6} = "none";                  % speaker B segments
        epoched_data{w, 7} = string({EEG.chanlocs.labels})'; % channels info
        
        % Increment the counters
        w = w + 1;
        tmp_counter = tmp_counter + 1;

    end

    disp([num2str(tmp_counter) ' Listening trials processed!'])

    %% Part 2: Fetch silence trials
    
    tmp_counter = 0;

    for i = 1:6
        
        % Identify Listening events
        count_idx = contains(trimmed_labels, ['Silence_T' num2str(i)]);
        silence_idx = find(count_idx);
        
        % Check whether there is a beginning and end (i.e., two events of the type)
        if sum(count_idx) < 2
            warning(['Listening T' num2str(i) ' did not have 2 events.'])
            continue
        end
        
        % Fetch beginning and end latencies
        trial_start = EEG.event(silence_idx(1)).latency;
        trial_end = EEG.event(silence_idx(2)).latency;
        
        % Identify if any mask portion should be attached
        isMasked = mask >= trial_start & mask <= trial_end;
        
        % Check if any full mask segment falls in trial; store
        full_mask = sum(isMasked,2) == 2;
        full_mask_idx = full_mask;
        
        % Check if any mask segment partially falls in trial; store
        partial_mask = sum(isMasked,2) == 1;
        partial_idx  = find(partial_mask);
        
        % If there are partial segments, shift start/end of trial
        if sum(partial_mask) > 0
           for j = 1:numel(partial_idx)
                
               m  = mask(partial_idx(j),:);
               
               if m(1) < trial_start && m(2) < trial_end
                   
                   m(1) = trial_start+1;
               elseif m(2) > trial_end && m(1) > trial_start

                   m(2) = trial_end;
               end

               mask(partial_idx(j),:) = m; % modify mask to not exceed trial bounds

           end
        end

        % If there are full segments, scale and store them for later sue
        if sum(full_mask) > 0
            trial_mask = mask(full_mask_idx,:) - trial_start;
        else
            trial_mask = nan;
        end 

        % Extract the epoch data from the EEG
        epoch = EEG.data(:, trial_start:trial_end);
        
        % Store the data and trial type in the output cell array
        epoched_data{w, 1} = epoch;
        epoched_data{w, 2} = "Silence";
        epoched_data{w, 3} = i;
        epoched_data{w, 4} = trial_mask;
        epoched_data{w, 5} = "none";
        epoched_data{w, 6} = "none";
        epoched_data{w, 7} = string({EEG.chanlocs.labels})';
        
        % Increment the counters
        w = w + 1;
        tmp_counter = tmp_counter + 1;

    end

    disp([num2str(tmp_counter) ' Silence trials processed!'])

    %% Part 3: Fetch High Interest Trials

    tmp_counter = 0;

    for i = 1:6
        
        % Identify Listening events
        count_idx = contains(trimmed_labels, ['HighInterest_T' num2str(i)]);
        highInt_idx = find(count_idx);
        
        % Check whether there is a beginning and end (i.e., two events of the type)
        if sum(count_idx) < 2
            warning(['Listening T' num2str(i) ' did not have 2 events.'])
            continue
        end
        
        % Fetch beginning and end latencies
        trial_start = EEG.event(highInt_idx(1)).latency;
        trial_end = EEG.event(highInt_idx(2)).latency;
        
        % Identify if any mask portion should be attached
        isMasked = mask >= trial_start & mask <= trial_end;
        
        % Check if any full mask segment falls in trial; store
        full_mask = sum(isMasked,2) == 2;
        full_mask_idx = full_mask;
        
        % Check if any mask segment partially falls in trial; store
        partial_mask = sum(isMasked,2) == 1;
        partial_idx  = find(partial_mask);
        
        % If there are partial segments, shift start/end of trial
        if sum(partial_mask) > 0
           for j = 1:numel(partial_idx)
                
               m  = mask(partial_idx(j),:);
               
               if m(1) < trial_start && m(2) < trial_end
                   
                   m(1) = trial_start+1;
               elseif m(2) > trial_end && m(1) > trial_start

                   m(2) = trial_end;
               end

               mask(partial_idx(j),:) = m; % modify mask to not exceed trial bounds

           end
        end

        % If there are full segments, scale and store them for later sue
        if sum(full_mask) > 0
            trial_mask = mask(full_mask_idx,:) - trial_start;
        else
            trial_mask = nan;
        end 

        % Extract the epoch data from the EEG
        epoch = EEG.data(:, trial_start:trial_end);

        % Identify the events within the trial
        eventsInTrial = EEG.event([EEG.event.latency] > trial_start & [EEG.event.latency] < trial_end); 
        codelabels = {eventsInTrial.codelabel}; % Convert to cell array

        % Identify Speaker A events
        isSpeakerA = contains(codelabels, "Speaker_A");
        Anumb = sum(isSpeakerA);
        A_Events = eventsInTrial(isSpeakerA);

        % Identify Speaker B events
        isSpeakerB = contains(codelabels, "Speaker_B");
        Bnumb = sum(isSpeakerB);
        B_Events = eventsInTrial(isSpeakerB);
        
        % Loop through speaker A turns
        whereisA = [];
        turn = 1;
        for k = 1:Anumb
            if k < Anumb
                curr = A_Events(k);
                foll = A_Events(k+1);

                if contains(curr.codelabel, "Start")
                    if contains(foll.codelabel, "End")
                        whereisA(turn,1) = curr.latency;
                        whereisA(turn,2) = foll.latency;
                        turn = turn + 1;
                    else
                        warning(['In Speaker A, a speaking event had no end.']);
                    end
                end
            else
                % If trial ends with speaker speaking, cut at end
                curr = A_Events(k);
                if contains(curr.codelabel, "Start")
                    whereisA(turn,1) = curr.latency;
                    whereisA(turn,2) = trial_end;
                end
            end
        end
        
        % Loop through speaker B turns
        whereisB = [];
        turn = 1;
        for k = 1:Bnumb
            if k < Bnumb
                curr = B_Events(k);
                foll = B_Events(k+1);

                if contains(curr.codelabel, "Start")
                    if contains(foll.codelabel, "End")
                        whereisB(turn,1) = curr.latency;
                        whereisB(turn,2) = foll.latency;
                        turn = turn + 1;
                    else
                        warning(['In Speaker B, a speaking event had no end.']);
                    end
                end
            else
                % If trial ends with speaker speaking, cut at end
                curr = B_Events(k);
                if contains(curr.codelabel, "Start")
                    whereisB(turn,1) = curr.latency; %#ok<*AGROW> 
                    whereisB(turn,2) = trial_end;
                end
            end
        end

%         (whereisA - trial_start) / 500
%         (whereisB - trial_start) / 500  % Good!
                

        % Store the data and trial type in the output cell array
        epoched_data{w, 1} = epoch;                   % epoch data
        epoched_data{w, 2} = "HighInterest";          % condition
        epoched_data{w, 3} = i;                       % counter
        epoched_data{w, 4} = trial_mask;              % mask info
        epoched_data{w, 5} = whereisA - trial_start;  % speaker A segments
        epoched_data{w, 6} = whereisB - trial_start;  % speaker B segments
        epoched_data{w, 7} = string({EEG.chanlocs.labels})'; % channels info
        
        % Increment the counters
        w = w + 1;
        tmp_counter = tmp_counter + 1;

    end

    disp([num2str(tmp_counter) ' High Interest trials processed!'])


    %% Part 4: Fetch Low Interest Trials

    tmp_counter = 0;

    for i = 1:6
        
        % Identify Listening events
        count_idx = contains(trimmed_labels, ['LowInterest_T' num2str(i)]);
        lowInt_idx = find(count_idx);
        
        % Check whether there is a beginning and end (i.e., two events of the type)
        if sum(count_idx) < 2
            warning(['Listening T' num2str(i) ' did not have 2 events.'])
            continue
        end
        
        % Fetch beginning and end latencies
        trial_start = EEG.event(lowInt_idx(1)).latency;
        trial_end = EEG.event(lowInt_idx(2)).latency;
        
        % Identify if any mask portion should be attached
        isMasked = mask >= trial_start & mask <= trial_end;
        
        % Check if any full mask segment falls in trial; store
        full_mask = sum(isMasked,2) == 2;
        full_mask_idx = full_mask;
        
        % Check if any mask segment partially falls in trial; store
        partial_mask = sum(isMasked,2) == 1;
        partial_idx  = find(partial_mask);
        
        % If there are partial segments, shift start/end of trial
        if sum(partial_mask) > 0
           for j = 1:numel(partial_idx)
                
               m  = mask(partial_idx(j),:);
               
               if m(1) < trial_start && m(2) < trial_end
                   
                   m(1) = trial_start+1;
               elseif m(2) > trial_end && m(1) > trial_start

                   m(2) = trial_end;
               end

               mask(partial_idx(j),:) = m; % modify mask to not exceed trial bounds

           end
        end

        % If there are full segments, scale and store them for later sue
        if sum(full_mask) > 0
            trial_mask = mask(full_mask_idx,:) - trial_start;
        else
            trial_mask = nan;
        end 

        % Extract the epoch data from the EEG
        epoch = EEG.data(:, trial_start:trial_end);

        % Identify the events within the trial
        eventsInTrial = EEG.event([EEG.event.latency] > trial_start & [EEG.event.latency] < trial_end); 
        codelabels = {eventsInTrial.codelabel}; % Convert to cell array

        % Identify Speaker A events
        isSpeakerA = contains(codelabels, "Speaker_A");
        Anumb = sum(isSpeakerA);
        A_Events = eventsInTrial(isSpeakerA);

        % Identify Speaker B events
        isSpeakerB = contains(codelabels, "Speaker_B");
        Bnumb = sum(isSpeakerB);
        B_Events = eventsInTrial(isSpeakerB);
        
        % Loop through speaker A turns
        whereisA = [];
        turn = 1;
        for k = 1:Anumb
            if k < Anumb
                curr = A_Events(k);
                foll = A_Events(k+1);

                if contains(curr.codelabel, "Start")
                    if contains(foll.codelabel, "End")
                        whereisA(turn,1) = curr.latency;
                        whereisA(turn,2) = foll.latency;
                        turn = turn + 1;
                    else
                        warning(['In Speaker A, a speaking event had no end.']);
                    end
                end
            else
                % If trial ends with speaker speaking, cut at end
                curr = A_Events(k);
                if contains(curr.codelabel, "Start")
                    whereisA(turn,1) = curr.latency;
                    whereisA(turn,2) = trial_end;
                end
            end
        end
        
        % Loop through speaker B turns
        whereisB = [];
        turn = 1;
        for k = 1:Bnumb
            if k < Bnumb
                curr = B_Events(k);
                foll = B_Events(k+1);

                if contains(curr.codelabel, "Start")
                    if contains(foll.codelabel, "End")
                        whereisB(turn,1) = curr.latency;
                        whereisB(turn,2) = foll.latency;
                        turn = turn + 1;
                    else
                        warning(['In Speaker B, a speaking event had no end.']);
                    end
                end
            else
                % If trial ends with speaker speaking, cut at end
                curr = B_Events(k);
                if contains(curr.codelabel, "Start")
                    whereisB(turn,1) = curr.latency; %#ok<*AGROW> 
                    whereisB(turn,2) = trial_end;
                end
            end
        end

%         (whereisA - trial_start) / 500
%         (whereisB - trial_start) / 500  % Good!
                

        % Store the data and trial type in the output cell array
        epoched_data{w, 1} = epoch;                   % epoch data
        epoched_data{w, 2} = "LowInterest";          % condition
        epoched_data{w, 3} = i;                       % counter
        epoched_data{w, 4} = trial_mask;              % mask info
        epoched_data{w, 5} = whereisA - trial_start;  % speaker A segments
        epoched_data{w, 6} = whereisB - trial_start;  % speaker B segments
        epoched_data{w, 7} = string({EEG.chanlocs.labels})'; % channels info
        
        % Increment the counters
        w = w + 1;
        tmp_counter = tmp_counter + 1;

    end

    disp([num2str(tmp_counter) ' Low Interest trials processed!'])

end
