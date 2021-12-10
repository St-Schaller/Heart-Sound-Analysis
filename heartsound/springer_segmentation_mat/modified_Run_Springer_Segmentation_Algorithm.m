% function assigned_states = runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix, pi_vector, total_observation_distribution, figures)
%
% A function to assign states to a PCG recording using a duration dependant
% logisitic regression-based HMM, using the trained B_matrix and pi_vector
% trained in "trainSpringerSegmentationAlgorithm.m". Developed for use in
% the paper:
% D. Springer et al., "Logistic Regression-HSMM-based Heart Sound
% Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.
%
%% INPUTS:
% audio_data: The audio data from the PCG recording
% Fs: the sampling frequency of the audio recording
% B_matrix: the observation matrix for the HMM, trained in the
% "trainSpringerSegmentationAlgorithm.m" function
% pi_vector: the initial state distribution, also trained in the
% "trainSpringerSegmentationAlgorithm.m" function
% total_observation_distribution, the observation probabilities of all the
% data, again, trained in trainSpringerSegmentationAlgorithm.
% figures: (optional) boolean variable for displaying figures
%
%% OUTPUTS:
% assigned_states: the array of state values assigned to the original
% audio_data (in the original sampling frequency).
%
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

function assigned_states = modified_Run_Springer_Segmentation_Algorithm(audio_data,filename, Fs, B_matrix, pi_vector, total_observation_distribution, figures)

%% Preliminary
if(nargin < 7)
    figures = false;
end

%% Get PCG Features:

[PCG_Features, featuresFs] = getSpringerPCGFeatures(audio_data, Fs);

%% Get PCG heart rate

[heartRate, systolicTimeInterval] = getHeartRateSchmidt(audio_data, Fs);

if heartRate == 0 || systolicTimeInterval == 0
    assigned_states = [];
    return;
end

[~, ~, qt] = viterbiDecodePCG_Springer(PCG_Features, pi_vector, B_matrix, total_observation_distribution, heartRate, systolicTimeInterval, featuresFs);

assigned_states = expand_qt(qt, featuresFs, Fs, length(audio_data));

disp(class(assigned_states))

writematrix(assigned_states,'test.csv')

indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
    switch assigned_states(1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=4;
    end
else
    switch assigned_states(indx(1)+1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=0;
    end
    K=K+1;
end

indx2                = indx(K:end);
rem                  = mod(length(indx2),4);
indx2(end-rem+1:end) = [];
A                    = reshape(indx2,4,length(indx2)/4).';

folder = 'YOUR/SEGMENTATION/FOLDER'
ext = '.csv'
path = [folder filename ext]
writematrix(A, path)

figures = false
if(figures)
   figure('Name','Derived state sequence', 'visible', 'off');
   t1 = (1:length(audio_data))./Fs;
   plot(t1,normalise_signal(audio_data),'k');
   hold on;
   plot(t1,assigned_states,'r--');
   xlabel('Time (s)');
   legend('Audio data', 'Derived states');
   saveas(gcf,fullfile('YOUR/IMAGE/FOLDER', filename), 'png');
end








