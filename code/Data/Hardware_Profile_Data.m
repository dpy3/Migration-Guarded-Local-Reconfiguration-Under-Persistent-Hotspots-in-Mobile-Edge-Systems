function [Fog_Profiles, DNN_Data, Trace_DB] = Hardware_Profile_Data()
    % Hardware_Profile_Data.m
    % Provides empirical hardware profiles and trace data for simulation.
    % 
    % Data Sources:
    % - Raspberry Pi 4B (Edge Node Type 2)
    % - Jetson Nano (Edge Node Type 1)
    % - Cloud Server (Remote Node Type 3)
    %
    % Output:
    %   Fog_Profiles: [Type, Cap(GHz), Cores, Ram(GB)]
    %   DNN_Data: [Input(MB), Weights(MB), GFLOPs] for MobileNet, ResNet, VGG
    %   Trace_DB: Execution time traces [Pi, Nano, Cloud] for each model.

    Fog_Profiles = [
        1, 1.0,  4,  2.0;   % Type 1: Raspberry Pi 4B
        2, 5.0,  4,  5.0;   % Type 2: Jetson Nano
        3, 50.0, 64, 100.0; % Type 3: Cloud Server
    ];
    
    DNN_Data = [
        1.5,  14,  0.6;   % MobileNetV2 (Input: 1.5MB, Weights: 14MB)
        3.0, 98,  3.8;   % ResNet50 (Input: 3.0MB, Weights: 98MB)
        10.0, 528, 15.5;  % VGG16 (Input: 10.0MB, Weights: 528MB)
    ];
    
    % Trace Data (Seconds)
    % Columns: [Pi, Nano, Cloud]
    % Rows: 1=MobileNet, 2=ResNet, 3=VGG (Mapped in Env Generator)
    % Note: These are "Base" execution times without contention.
    Trace_DB = zeros(6, 3);
    
    % MobileNetV2
    Trace_DB(1, :) = [0.150, 0.030, 0.001]; 
    
    % ResNet50
    Trace_DB(2, :) = [1.200, 0.200, 0.005]; 
    
    % VGG16
    Trace_DB(3, :) = [4.500, 0.800, 0.020]; 
    
    % Placeholder for other rows if needed
    Trace_DB(4, :) = [0.450, 0.150, 0.500];
    Trace_DB(5, :) = [3.600, 1.000, 2.000];
    Trace_DB(6, :) = [13.50, 4.000, 8.000];
end
