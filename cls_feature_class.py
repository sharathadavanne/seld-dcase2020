# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.externals import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import math


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param dataset: string, dataset name, supported: foa - ambisonic or mic- microphone format
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        # Sound event classes dictionary
        self._unique_classes = params['unique_classes']
        self._audio_max_len_samples = params['max_audio_len_s'] * self._fs  # TODO: Fix the audio synthesis code to always generate 60s of
        # audio. Currently it generates audio till the last active sound event, which is not always 60s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        # For regression task only
        self._default_azi = 180
        self._default_ele = 90

        self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))
        self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.random.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1])*self._eps
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
        return spectra

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.reshape((linear_spectra.shape[0], self._nb_mel_bins * linear_spectra.shape[-1]))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        I1 = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        I2 = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])
        I3 = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])

        normal = np.sqrt(I1**2 + I2**2 + I3**2) + self._eps
        I1 = np.dot(I1 / normal, self._mel_wts)
        I2 = np.dot(I2 / normal, self._mel_wts)
        I3 = np.dot(I3 / normal, self._mel_wts)

        # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        foa_iv = np.dstack((I1, I2, I3))
        foa_iv = foa_iv.reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.reshape((linear_spectra.shape[0], self._nb_mel_bins*gcc_channels))

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        return audio_spec

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file):
        """
        Reads description csv file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: csv file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 2*nb_classes], nb_classes each for azimuth and elevation angles,
        if active, the DOA values will be in degrees, else, it will contain default doa values given by
        self._default_ele and self._default_azi
        """

        se_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        azi_label = self._default_azi*np.ones((self._max_label_frames, len(self._unique_classes)))
        ele_label = self._default_ele*np.ones((self._max_label_frames, len(self._unique_classes)))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]-1] = 1
                    azi_label[frame_ind, active_event[0]-1] = active_event[1]
                    ele_label[frame_ind, active_event[0]-1] = active_event[2]

        label_mat = np.concatenate((se_label, azi_label, ele_label), axis=1)
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            spect = self._get_spectrogram_for_file(wav_filename)

            #extract mel
            mel_spect = self._get_mel_spectrogram(spect)

            feat = None
            if self._dataset is 'foa':
                # extract intensity vectors
                foa_iv = self._get_foa_intensity_vectors(spect)
                feat = np.concatenate((mel_spect, foa_iv), axis=-1)
            elif self._dataset is 'mic':
                # extract gcc
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
            else:
                print('ERROR: Unknown dataset format {}'.format(self._dataset))
                exit()

            # plot.figure()
            # plot.subplot(211), plot.imshow(mel_spect.T)
            # plot.subplot(212), plot.imshow(foa_iv.T)
            # plot.show()

            if feat is not None:
                print('{}: {}, {}'.format(file_cnt, file_name, feat.shape ))
                np.save(os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0])), feat)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self.load_output_format_file(os.path.join(self._desc_dir, file_name))
            label_mat = self.get_labels_for_file(desc_file)
            print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), int(_words[3])])
            else:
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[2]), float(_words[3]), float(_words[4])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                if len(_value)==3:
                    _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
                else:
                    _fid.write('{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), float(_value[1]), float(_value[2]), float(_value[3])))
        _fid.close()

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_default_azi_ele_regr(self):
        return self._default_azi, self._default_ele

    def get_nb_channels(self):
        return self._nb_channels

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_azi_ele_list(self):
        return self._azi_list, self._ele_list

    def get_nb_frames(self):
        return self._max_label_frames

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)
