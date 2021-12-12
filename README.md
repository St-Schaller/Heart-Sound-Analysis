# Heart-Sound-Analysis

Heart-Sound-Analysis is the repository for the associated thesis "Audio Based Detection of Heart Abnormalities" at the MISIT chair of the University Augsburg. The thesis compares different methods of automatic heart sounds analysis including audio feature extraction tools and manually extracted features.

(c) 2021 Stefanie Schaller under GNU, see the LICENSE.md file for details.

## Data processing
 In the data package all files are for data preprocessing. File preprocessing.py includes upsampling and creating labels for test and trainin set. The two different audio feature extractors [DeepSpectrum](https://github.com/DeepSpectrum/DeepSpectrum) and [OpenL3](https://github.com/marl/openl3) have different dependencies, so installing both in different virtual environments for feature extraction is advised. In create_feature_extractors.py example code for extracting features with OpenL3 can be found. The code for the manual extracted features can be found in feature_extraction.py.

## Segmentation
### Springer Algorithm
The folder springer_segmentation_mat includes the code for the [Springer](https://github.com/davidspringer/Springer-Segmentation-Code) segmentation algorithm used in this thesis for segmentation. To run the code a valid matlab version must be installed. The segmentation.py file runs the *modified* Springer script and Springer algorithm.


### Modified Empirical Wavelet Transform
This methods is an implementation based on the method proposed by Narváez, et. al [[1]](#1). A the authors did not provide the source code, this method uses the (modified) ewtpy python package [[2]](#2) for extracting the modified empirical wavelet transform. While the NASE could be extracted, the differentiation of S1 and S2 sounds did not succeed.

## Evaluation
In the evaluation package the file grid_search.py was used for extensive grid search and the train_model.py was used to train the model with the best hyperparameters found during grid search.




## References
<a id="1">[1]</a> 
Narváez, Pedro and Gutierrez, Steven and Percybrooks, Winston (2020). 
Automatic Segmentation and Classification of Heart Sounds Using Modified Empirical Wavelet Transform and Power Features
Applied Sciences, 10(07), 4791.

<a id="2">[2]</a> 
Vinícius R. Carvalho, Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes (2020)
Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification
Biomedical Signal Processing and Control, Volume 62, , 102073
ISSN 1746-8094
https://doi.org/10.1016/j.bspc.2020.102073.




