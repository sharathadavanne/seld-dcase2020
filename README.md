
# DCASE 2020: Sound event localization and detection (SELD) task
[Please visit the official webpage of the DCASE 2020 Challenge for details missing in this repo](http://dcase.community/challenge2020/task-sound-event-localization-and-detection). 
   
As the baseline method for the SELD task, we use the SELDnet method studied in the following papers. If you are using this baseline method or the datasets in any format, then please consider citing the following two papers

> Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)

> Sharath Adavanne, Archontis Politis and Tuomas Virtanen, "Localization, Detection and Tracking of Multiple Moving Sound Sources with a Convolutional Recurrent Neural Network" in the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2019)

## BASELINE METHOD

In comparison to the SELDnet studied in the papers above, we have changed the following to improve its performance and evaluate the performance better.
 * **Features**: The original SELDnet employed naive phase and magnitude components of the spectrogram as the input feature for all input formats of audio. In this baseline method, we use separate features for first-order Ambisonic (FOA) and microphone array (MIC) datasets. As the interaural level difference feature, we employ the 64-band mel energies extracted from each channel of the input audio for both FOA and MIC. To encode the interaural time difference features, we employ intensity vector features for FOA, and generalized cross correlation features for MIC. 
 * **Loss/Objective**: The original SELDnet employed mean square error (MSE) for the DOA loss estimation, and this was computed irrespecitve of the presence or absence of the sound event. In the current baseline, we used a masked-MSE, which computes MSE only when the sound event is active in the reference.
 * **Evaluation metrics**: The performance of the original SELDnet was evaluated with stand-alone metrics for detection, and localization. Mainly because there was no suitable metric which could jointly evaluate the performance of localization and detection. Since then, we have proposed a new metric that can jointly evaluate the performance (more about it is described in the metrics section below), and we employ this new metric for evaluation here.   
 
The final SELDnet architecture is as shown below. The input is the multichannel audio, from which the different acoustic features are extracted based on the input format of the audio. Based on the chosen dataset (FOA or MIC), the baseline method takes a sequence of consecutive feature-frames and predicts all the active sound event classes for each of the input frame along with their respective spatial location, producing the temporal activity and DOA trajectory for each sound event class. In particular, a convolutional recurrent neural network (CRNN) is used to map the frame sequence to the two outputs in parallel. At the first output, SED is performed as a multi-label multi-class classification task, allowing the network to simultaneously estimate the presence of multiple sound events for each frame. At the second output, DOA estimates in the continuous 3D space are obtained as a multi-output regression task, where each sound event class is associated with three regressors that estimate the Cartesian coordinates x, y and z axes of the DOA on a unit sphere around the microphone.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2020/blob/master/images/CRNN_SELDT_DCASE2020.png" width="400" title="SELDnet Architecture">
</p>

The SED output of the network is in the continuous range of [0 1] for each sound event in the dataset, and this value is thresholded to obtain a binary decision for the respective sound event activity. Finally, the respective DOA estimates for these active sound event classes provide their spatial locations.

The figure below visualizes the SELDnet input and outputs for one of the recordings in the dataset. The horizontal-axis of all sub-plots for a given dataset represents the same time frames, the vertical-axis for spectrogram sub-plot represents the frequency bins, vertical-axis for SED reference and prediction sub-plots represents the unique sound event class identifier, and for the DOA reference and prediction sub-plots, it represents the distances along the Cartesian axes. The figures represents each sound event class and its associated DOA outputs with a unique color. Similar plot can be visualized on your results using the [provided script](visualize_SELD_output.py).

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2020/blob/master/images/SELDnet_output.jpg" width="300" title="SELDnet input and output visualization">
</p>

## DATASETS

The participants can choose either of the two or both the following datasets,

 * **TAU-NIGENS Spatial Sound Events 2020 - Ambisonic**
 * **TAU-NIGENS Spatial Sound Events 2020 - Microphone Array**

These datasets contain recordings from an identical scene, with **TAU-NIGENS Spatial Sound Events 2020 - Ambisonic** providing four-channel First-Order Ambisonic (FOA) recordings while  **TAU-NIGENS Spatial Sound Events 2020 - Microphone Array** provides four-channel directional microphone recordings from a tetrahedral array configuration. Both formats are extracted from the same microphone array, and additional information on the spatial characteristics of each format can be found below. The participants can choose one of the two, or both the datasets based on the audio format they prefer. Both the datasets, consists of a development and evaluation set. The development set consists of 600, one minute long recordings sampled at 24000 Hz. All participants are expected to use the fixed splits provided in the baseline method for reporting the development scores. We use 400 recordings for training split (fold 3 to 6), 100 for validation (fold 2) and 100 for testing (fold 1). The evaluation set consists of 200, one-minute recordings, and will be released at a later point. 

More details on the recording procedure and dataset can be read on the [DCASE 2020 task webpage](http://dcase.community/challenge2020/task-sound-event-localization-and-detection).

The two development datasets can be downloaded from the link - [**TAU-NIGENS Spatial Sound Events 2020 - Ambisonic and Microphone Array**, Development dataset](https://doi.org/10.5281/zenodo.3740236) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3740236.svg)](https://doi.org/10.5281/zenodo.3740236) 


## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `keras_model.py` script implements the SELDnet architecture.
* The `evaluation_metrics.py` script implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and the DOA metrics explained in the paper. These were used in the DCASE 2019 SELD task. We use this here to just for legacy comparison
* The `SELD_evaluation_metrics.py` script implements the metrics for joint evaluation of detection and localization.
* The `seld.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.

Additionally, we also provide supporting scripts that help analyse the results.
 * `visualize_SELD_output.py` script to visualize the SELDnet output
 

### Prerequisites

The provided codebase has been tested on python 3.6.9/3.7.3 and Keras 2.2.4/2.3.1


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset path in `parameter.py` script. For the above example, you will change `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped. 

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder.

```
python3 batch_feature_extraction.py
```

You can now train the SELDnet using default parameters using
```
python3 seld.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following
```
python3 seld.py <task-id> <job-id>
```
Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.

In order to get baseline results on the development set for Microphone array recordings, you can run the following command
```
python3 seld.py 2
```
Similarly, for Ambisonic format baseline results, run the following command
```
python3 seld.py 4
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.

* The code also plots training curves, intermediate results and saves models in the `model_dir` path provided by the user in `parameter.py` file.

* In order to visualize the output of SELDnet and for submission of results, set `dcase_output=True` and provide `dcase_dir` directory. This will dump file-wise results in the directory, which can be individually visualized using `visualize_SELD_output.py` script.

## Results on development dataset

As the evaluation metrics we use two different approaches as discussed in our recent paper below

> Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen. Joint measurement of localization and detection of sound events. In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). New Paltz, NY, Oct 2019.

The first metric is more focused on the detection part, also referred as the location-aware detection, which gives us the error rate (ER) and F-score (F) in one-second non-overlapping segments. We consider the prediction to be correct if the prediction and reference class are the same, and the distance between them is below 20&deg;.
The second metric is more focused on the localization part, also referred as the class-aware localization, which gives us the DOA error (DE), and F-score (DE_F) in one-second non-overlapping segments. Unlike the location-aware detection, we do not use any distance threshold, but estimate the distance between the correct prediction and reference.

The evaluation metric scores for the test split of the development dataset is given below    

| Dataset | ER | F | DE | DE_F |
| ----| --- | --- | --- | --- |
| Ambisonic (FOA) | 0.84 | 23.3 % | 28.0&deg; | 56.4 % |
| Microphone Array (MIC) |0.82 | 24.3 % | 28.4&deg; | 61.2 % |

**Note:** The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.

## Submission

* Before submission, make sure your SELD results look good by visualizing the results using `visualize_SELD_output.py` script
* Make sure the file-wise output you are submitting is produced at 100 ms hop length. At this hop length a 60 s audio file has 600 frames.

For more information on the submission file formats [check the website](http://dcase.community/challenge2020/task-sound-event-localization-and-detection#submission)

## License

Except for the contents in the `metrics` folder that have [MIT License](metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](LICENSE.md).

## Acknowledgments

The research leading to these results has received funding from the European Research Council under the European Unions H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND.
