function Profiles = load_hardware_profile()
    % LOAD_HARDWARE_PROFILE
    % Loads hardware profiles from CSV files in ../Data/
    
    % Determine path to Data folder
    current_dir = fileparts(mfilename('fullpath'));
    data_dir = fullfile(current_dir, '..', 'Data');
    
    Profiles = struct();
    
    % --- VGG16 ---
    vgg_file = fullfile(data_dir, 'VGG16_Profile.csv');
    if exist(vgg_file, 'file')
        opts = detectImportOptions(vgg_file);
        opts.VariableNamingRule = 'preserve';
        T = readtable(vgg_file, opts);
        
        % Check column names: VGG16 has Output_Data_MB, others have Input_MB
        % Note: Variable names might be normalized by readtable if preserve is not used, 
        % but we used preserve.
        % Actually, let's check standard names too.
        
        vars = T.Properties.VariableNames;
        
        if ismember('Output_Data_MB', vars)
            Profiles.vgg16.data = T.Output_Data_MB';
        elseif ismember('Input_MB', vars)
            Profiles.vgg16.data = T.Input_MB';
        else
            % Fallback: Assume column 3 is data (Input_MB/Output_Data_MB)
             Profiles.vgg16.data = table2array(T(:, 3))';
        end
        
        if ismember('Comp_GFLOPs', vars)
            Profiles.vgg16.comp = T.Comp_GFLOPs';
        else
             Profiles.vgg16.comp = table2array(T(:, 4))'; % Adjust index if needed
        end
        
        % Ensure row vector
        if size(Profiles.vgg16.data, 1) > 1, Profiles.vgg16.data = Profiles.vgg16.data'; end
        if size(Profiles.vgg16.comp, 1) > 1, Profiles.vgg16.comp = Profiles.vgg16.comp'; end
        
    else
        warning('VGG16_Profile.csv not found at %s', vgg_file);
    end
    
    % --- ResNet ---
    resnet_file = fullfile(data_dir, 'ResNet_Profile.csv');
    if exist(resnet_file, 'file')
        opts = detectImportOptions(resnet_file);
        opts.VariableNamingRule = 'preserve';
        T = readtable(resnet_file, opts);
        
        vars = T.Properties.VariableNames;
        if ismember('Input_MB', vars)
            Profiles.resnet.data = T.Input_MB';
        else
            Profiles.resnet.data = table2array(T(:, 2))'; % Check index
        end
        
        if ismember('Comp_GFLOPs', vars)
            Profiles.resnet.comp = T.Comp_GFLOPs';
        else
             Profiles.resnet.comp = table2array(T(:, 3))';
        end
        
         % Ensure row vector
        if size(Profiles.resnet.data, 1) > 1, Profiles.resnet.data = Profiles.resnet.data'; end
        if size(Profiles.resnet.comp, 1) > 1, Profiles.resnet.comp = Profiles.resnet.comp'; end

    else
        warning('ResNet_Profile.csv not found');
    end
    
    % --- MobileNet ---
    mobilenet_file = fullfile(data_dir, 'MobileNet_Profile.csv');
    if exist(mobilenet_file, 'file')
        opts = detectImportOptions(mobilenet_file);
        opts.VariableNamingRule = 'preserve';
        T = readtable(mobilenet_file, opts);
        
        vars = T.Properties.VariableNames;
        if ismember('Input_MB', vars)
            Profiles.mobilenet.data = T.Input_MB';
        else
             Profiles.mobilenet.data = table2array(T(:, 2))';
        end
        
        if ismember('Comp_GFLOPs', vars)
            Profiles.mobilenet.comp = T.Comp_GFLOPs';
        else
             Profiles.mobilenet.comp = table2array(T(:, 3))';
        end
        
         % Ensure row vector
        if size(Profiles.mobilenet.data, 1) > 1, Profiles.mobilenet.data = Profiles.mobilenet.data'; end
        if size(Profiles.mobilenet.comp, 1) > 1, Profiles.mobilenet.comp = Profiles.mobilenet.comp'; end

    else
        warning('MobileNet_Profile.csv not found');
    end

end
