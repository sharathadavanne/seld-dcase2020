# Script for visualising the SELD output.
#
# NOTE: Make sure to use the appropriate backend for the matplotlib based on your OS

import os
import numpy as np
import librosa.display
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
from metrics import evaluation_metrics
import cls_feature_class
import parameter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plot
#plot.switch_backend('Qt4Agg')
plot.switch_backend('agg')

from IPython import embed

def collect_classwise_data(_in_dict):
    _out_dict = {}
    for _key in _in_dict.keys():
        for _seld in _in_dict[_key]:
            if _seld[0] not in _out_dict:
                _out_dict[_seld[0]] = []
            if len(_seld) == 3:
                ele_rad = _seld[2]*np.pi/180.
                azi_rad = _seld[1]*np.pi/180
                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                _out_dict[_seld[0]].append([_key, _seld[0], x, y, z])
            else:
                _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2], _seld[3]])
    return _out_dict


def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    for class_ind in plot_data.keys():
        time_ax = np.array(plot_data[class_ind])[:, 0] *hop_len_s
        y_ax = np.array(plot_data[class_ind])[:, ind]
        plot.plot(time_ax, y_ax, marker='.', color=cmap[class_ind], linestyle='None', markersize=4)
    plot.grid()
    plot.xlim([0, 60])
    if not plot_x_ax:
        plot.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off


# --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------
# output format file to visualize
pred = '/users/sadavann/seld-dcase2020/results/4_foa_dev/fold1_room1_mix008_ov1.csv'

# path of reference audio directory for visualizing the spectrogram and description directory for
# visualizing the reference
# Note: The code finds out the audio filename from the predicted filename automatically
ref_dir = '/scratch/asignal/sharath/DCASE2020_SELD_dataset/metadata_dev/'
aud_dir = '/scratch/asignal/sharath/DCASE2020_SELD_dataset/foa_dev/'

# load the predicted output format
params = parameter.get_params()
feat_cls = cls_feature_class.FeatureClass(params)
pred_dict = feat_cls.load_output_format_file(pred)

# load the reference output format
ref_filename = os.path.basename(pred)
ref_dict = feat_cls.load_output_format_file(os.path.join(ref_dir, ref_filename))


pred_data = collect_classwise_data(pred_dict)
ref_data = collect_classwise_data(ref_dict)

nb_classes = len(feat_cls.get_classes())

# load the audio and extract spectrogram
ref_filename = os.path.basename(pred).replace('.csv', '.wav')
audio, fs = feat_cls._load_audio(os.path.join(aud_dir, ref_filename))
stft = np.abs(np.squeeze(feat_cls._spectrogram(audio[:, :1])))
stft = librosa.amplitude_to_db(stft, ref=np.max)

plot.figure()
gs = gridspec.GridSpec(5, 4)
ax0 = plot.subplot(gs[0, 1:3]), librosa.display.specshow(stft.T, sr=fs, x_axis='time', y_axis='linear'), plot.title('Spectrogram')
ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED reference')
ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED predicted')
ax3 = plot.subplot(gs[2, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=2), plot.ylim([-1, 1]), plot.title('x-axis DOA reference')
ax4 = plot.subplot(gs[2, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=2), plot.ylim([-1, 1]), plot.title('x-axis DOA predicted')
ax5 = plot.subplot(gs[3, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=3), plot.ylim([-1, 1]), plot.title('y-axis DOA reference')
ax6 = plot.subplot(gs[3, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=3), plot.ylim([-1, 1]), plot.title('y-axis DOA predicted')
ax7 = plot.subplot(gs[4, :2]), plot_func(ref_data, params['label_hop_len_s'], ind=4, plot_x_ax=True), plot.ylim([-1, 1]), plot.title('z-axis DOA reference')
ax8 = plot.subplot(gs[4, 2:]), plot_func(pred_data, params['label_hop_len_s'], ind=4, plot_x_ax=True), plot.ylim([-1, 1]), plot.title('z-axis DOA predicted')
ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
#plot.show()
plot.savefig(os.path.join('../images/', ref_filename.replace('.wav', '.jpg')), dpi=300)

