# Script for visualising the SELD output.
#
# NOTE: Make sure to use the appropriate backend for the matplotlib based on your OS

import os
import numpy as np
import librosa.display
import cls_feature_class
import parameter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plot
plot.switch_backend('agg')
plot.rcParams.update({'font.size': 22})


def collect_classwise_data(_in_dict):
    _out_dict = {}
    for _key in _in_dict.keys():
        for _seld in _in_dict[_key]:
            if _seld[0] not in _out_dict:
                _out_dict[_seld[0]] = []
            _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
    return _out_dict


def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False, plot_y_ax=False):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    for class_ind in plot_data.keys():
        time_ax = np.array(plot_data[class_ind])[:, 0] *hop_len_s
        y_ax = np.array(plot_data[class_ind])[:, ind]
        plot.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
    plot.grid()
    plot.xlim([0, 60])
    if not plot_x_ax:
        plot.gca().axes.set_xticklabels([])

    if not plot_y_ax:
        plot.gca().axes.set_yticklabels([])
# --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------
params = parameter.get_params()

# output format file to visualize
pred = os.path.join(params['dcase_dir'], '2_mic_dev/fold1_room1_mix006_ov1.csv')

# path of reference audio directory for visualizing the spectrogram and description directory for
# visualizing the reference
# Note: The code finds out the audio filename from the predicted filename automatically
ref_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
aud_dir = os.path.join(params['dataset_dir'], 'mic_dev')

# load the predicted output format
feat_cls = cls_feature_class.FeatureClass(params)
pred_dict = feat_cls.load_output_format_file(pred)
pred_dict_polar = feat_cls.convert_output_format_cartesian_to_polar(pred_dict)

# load the reference output format
ref_filename = os.path.basename(pred)
ref_dict_polar = feat_cls.load_output_format_file(os.path.join(ref_dir, ref_filename))

pred_data = collect_classwise_data(pred_dict_polar)
ref_data = collect_classwise_data(ref_dict_polar)

nb_classes = len(feat_cls.get_classes())

# load the audio and extract spectrogram
ref_filename = os.path.basename(pred).replace('.csv', '.wav')
audio, fs = feat_cls._load_audio(os.path.join(aud_dir, ref_filename))
stft = np.abs(np.squeeze(feat_cls._spectrogram(audio[:, :1])))
stft = librosa.amplitude_to_db(stft, ref=np.max)

plot.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 4)
ax0 = plot.subplot(gs[0, 1:3]), librosa.display.specshow(stft.T, sr=fs, x_axis='s', y_axis='linear'), plot.xlim([0, 60]), plot.xticks([]), plot.xlabel(''), plot.title('Spectrogram')
ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=1, plot_y_ax=True), plot.ylim([-1, nb_classes + 1]), plot.title('SED reference')
ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED predicted')
ax3 = plot.subplot(gs[2, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=2, plot_y_ax=True), plot.ylim([-180, 180]), plot.title('Azimuth reference')
ax4 = plot.subplot(gs[2, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=2), plot.ylim([-180, 180]), plot.title('Azimuth predicted')
ax5 = plot.subplot(gs[3, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=3, plot_y_ax=True), plot.ylim([-90, 90]), plot.title('Elevation reference')
ax6 = plot.subplot(gs[3, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=3), plot.ylim([-90, 90]), plot.title('Elevation predicted')
ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
plot.savefig(os.path.join(params['dcase_dir'] , ref_filename.replace('.wav', '.jpg')), dpi=300, bbox_inches = "tight")


