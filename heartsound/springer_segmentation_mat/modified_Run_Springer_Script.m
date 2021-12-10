

%% Example Springer script
% A script to demonstrate the use of the Springer segmentation algorithm

%% Copyright (C) 2016  David Springer
% dave.springer@gmail.com
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%%
close all;
clear all;

%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding:
springer_options = default_Springer_HSMM_options;

% Load the trained parameter matrices for Springer's HSMM model.
% The parameters were trained using 409 heart sounds from MIT heart
% sound database, i.e., recordings a0001-a0409.
%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding:
springer_options = default_Springer_HSMM_options;

%% Load the audio data and the annotations:
% These are 6 example PCG recordings, downsampled to 1000 Hz, with
% annotations of the R-peak and end-T-wave positions.
load('example_data.mat');





train_recordings = example_data.example_audio_data;
train_annotations = example_data.example_annotations;


%% Train the HMM:
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

save('example_B_matrix.mat', 'B_matrix');

% Specify the folder where the files live.
myFolder = 'YOUR/SOURCE/FOLDER';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    %% Load data and resample data
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    fprintf(baseFileName);
    [folder, baseFileNameNoExt, extension] = fileparts(fullFileName);
    [PCG, Fs1]         = audioread([fullFileName]);  % load data
    PCG_resampled      = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)
    %% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
    [assigned_states] = modified_Run_Springer_Segmentation_Algorithm(PCG_resampled, baseFileNameNoExt, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, true); % obtain the locations for S1, systole, s2 and diastole
    if isempty(assigned_states)
        continue;
    end
    %% Running extractFeaturesFromHsIntervals.m to obtain the features for normal/abnormal heart sound classificaiton
    %%features  = extractFeaturesFromHsIntervals(assigned_states,PCG_resampled);

    %% Running classifyFromHsIntervals.m to obtain the final classification result for the current recording
    %%classifyResult = classifyFromHsIntervals(features);

%%alternative way to get baseFileName
%%


end

