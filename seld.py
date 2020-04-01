#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
from metrics import evaluation_metrics, SELD_evaluation_metrics
import keras_model
import parameter
import time
plot.switch_backend('agg')
from IPython import embed


def segment_labels(_pred_dict, _max_frames, _frames_seg):
    '''
        Collects class-wise sound event location information in segments of length _frames_seg from reference dataset
    :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :param _frames_seg: Number of frames in one segment
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    nb_blocks = int(np.ceil(_max_frames/float(_frames_seg)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, _frames_seg):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // _frames_seg
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+_frames_seg):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


def collect_test_labels(_data_gen_test, _data_out, _nb_classes, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        if _data_gen_test.get_data_gen_mode():
            doa_label = tmp_label[1]
        else:
            doa_label = tmp_label[1][:, :, _nb_classes:]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa_label
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def plot_functions(fig_name, _tr_loss, _sed_loss, _doa_loss, _epoch_metric_loss, _new_metric, _new_seld_metric):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(411)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(412)
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='sed er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='sed f1')
    plot.plot(range(nb_epoch), _doa_loss[:, 0]/180., label='doa er / 180')
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='doa fr')
    plot.plot(range(nb_epoch), _epoch_metric_loss, label='seld')
    plot.legend()
    plot.grid(True)

    plot.subplot(413)
    plot.plot(range(nb_epoch), _new_metric[:, 0], label='seld er')
    plot.plot(range(nb_epoch), _new_metric[:, 1], label='seld f1')
    plot.plot(range(nb_epoch), _new_metric[:, 2]/180., label='doa er / 180')
    plot.plot(range(nb_epoch), _new_metric[:, 3], label='doa fr')
    plot.plot(range(nb_epoch), _new_seld_metric, label='seld')

    plot.legend()
    plot.grid(True)

    plot.subplot(414)
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_pks')
    plot.plot(range(nb_epoch), _doa_loss[:, 3], label='good_pks')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        # test_splits = [1, 2, 3, 4]
        # val_splits = [2, 3, 4, 1]
        # train_splits = [[3, 4], [4, 1], [1, 2], [2, 3]]

        # SUGGESTION: Considering the long training time, major tuning of the method can be done on the first split.
        # Once you finlaize the method you can evaluate its performance on the complete cross-validation splits
        test_splits = [2]
        val_splits = [2]
        train_splits = [[1, 2]]

    elif params['mode'] == 'eval':
        test_splits = [0]
        val_splits = [1]
        train_splits = [[2, 3, 4]]

    avg_scores_val = []
    avg_scores_test = []
    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False
        )

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        nb_classes = data_gen_train.get_nb_classes()
        gt = collect_test_labels(data_gen_val, data_out, nb_classes, params['quick_test'])
        sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
        doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n\tdoa_objective: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size'], params['doa_objective']))

        print('Using loss weights : {}'.format(params['loss_weights']))
        model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'])
        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        seld_metric = np.zeros(nb_epoch)
        new_seld_metric = np.zeros(nb_epoch)
        tr_loss = np.zeros(nb_epoch)
        doa_metric = np.zeros((nb_epoch, 6))
        sed_metric = np.zeros((nb_epoch, 2))
        new_metric = np.zeros((nb_epoch, 4))

        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # train once per epoch
            hist = model.fit_generator(
                generator=data_gen_train.generate(),
                steps_per_epoch=2 if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
                epochs=params['epochs_per_fit'],
                verbose=2,
            )
            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]

            # predict once per peoch
            pred = model.predict_generator(
                generator=data_gen_val.generate(),
                steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
                verbose=2
            )

            # Calculate the metrics
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1] if params['doa_objective'] is 'mse' else pred[1][:, :, nb_classes:])

            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_val.nb_frames_1s())
            doa_metric[epoch_cnt, :] = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt, sed_pred, sed_gt)
            seld_metric[epoch_cnt] = evaluation_metrics.compute_seld_metric(sed_metric[epoch_cnt, :], doa_metric[epoch_cnt, :])

            # new SELD scores
            cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=data_gen_val.get_nb_classes(), doa_threshold=20)
            pred_dict = evaluation_metrics.regression_label_format_to_output_format(
                data_gen_val, sed_pred, doa_pred
            )
            gt_dict = evaluation_metrics.regression_label_format_to_output_format(
                data_gen_val, sed_gt, doa_gt
            )

            pred_blocks_dict = segment_labels(pred_dict, sed_pred.shape[0], data_gen_val.nb_frames_1s())
            gt_blocks_dict = segment_labels(gt_dict, sed_gt.shape[0], data_gen_val.nb_frames_1s())

            cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
            new_metric[epoch_cnt, :] = cls_new_metric.compute_seld_scores()
            new_seld_metric[epoch_cnt] = evaluation_metrics.compute_seld_metric(new_metric[epoch_cnt, :2], new_metric[epoch_cnt, 2:])

            # Visualize the metrics with respect to epochs
            plot_functions(unique_name, tr_loss, sed_metric, doa_metric, seld_metric, new_metric, new_seld_metric)

            patience_cnt += 1
            if new_seld_metric[epoch_cnt] < best_seld_metric:
                best_seld_metric = new_seld_metric[epoch_cnt]
                best_epoch = epoch_cnt
                model.save(model_name)
                patience_cnt = 0

            print(
                'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
                '\n\t\t OLD SCORES: ER: {:0.2f}, F: {:0.2f}, DE: {:0.2f}, DE_F:{:0.2f}, seld_score: {:0.2f}, ' 
                '\n\t\t NEW SCORES: ER: {:0.2f}, F: {:0.2f}, DE: {:0.2f}, DE_F:{:0.2f}, seld_score: {:0.2f}, '
                'best_seld_score: {:0.2f}, best_epoch : {}\n'.format(
                    epoch_cnt, time.time() - start, tr_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1],
                    doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1], seld_metric[epoch_cnt],
                    new_metric[epoch_cnt, 0], new_metric[epoch_cnt, 1],
                    new_metric[epoch_cnt, 2], new_metric[epoch_cnt, 3],
                    new_seld_metric[epoch_cnt], best_seld_metric, best_epoch
                )
            )
            if patience_cnt > params['patience']:
                break

        avg_scores_val.append([new_metric[best_epoch, 0], new_metric[best_epoch, 1], new_metric[best_epoch, 2],
                               new_metric[best_epoch, 3], best_seld_metric])
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSELD_score: {}'.format(best_seld_metric))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(doa_metric[best_epoch, 0],
                                                                      doa_metric[best_epoch, 1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(sed_metric[best_epoch, 0],
                                                                       sed_metric[best_epoch, 1]))

        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('Loading testing dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=split, shuffle=False
        )

        print('\nLoading the best model and predicting results on the testing split')
        model = keras_model.load_seld_model('{}_model.h5'.format(unique_name), params['doa_objective'])
        pred_test = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )

        test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
        test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1] if params['doa_objective'] is 'mse' else pred_test[1][:, :, nb_classes:])

        if params['dcase_output']:
            # Dump results in DCASE output format for calculating final scores
            dcase_dump_folder = os.path.join(params['dcase_dir'], '{}_{}_{}'.format(task_id, params['dataset'], params['mode']))
            cls_feature_class.create_folder(dcase_dump_folder)
            print('Dumping recording-wise results in: {}'.format(dcase_dump_folder))

            test_filelist = data_gen_test.get_filelist()
            # Number of frames for a 60 second audio with 20ms hop length = 3000 frames
            max_frames_with_content = data_gen_test.get_nb_frames()

            # Number of frames in one batch (batch_size* sequence_length) consists of all the 3000 frames above with
            # zero padding in the remaining frames
            frames_per_file = data_gen_test.get_frame_per_file()

            for file_cnt in range(test_sed_pred.shape[0]//frames_per_file):
                output_file = os.path.join(dcase_dump_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
                dc = file_cnt * frames_per_file
                output_dict = evaluation_metrics.regression_label_format_to_output_format(
                    data_gen_test,
                    test_sed_pred[dc:dc + max_frames_with_content, :],
                    test_doa_pred[dc:dc + max_frames_with_content, :]
                )
                data_gen_test.write_output_format_file(output_file, output_dict)

        if params['mode'] is 'dev':
            test_data_in, test_data_out = data_gen_test.get_data_sizes()
            test_gt = collect_test_labels(data_gen_test, test_data_out, nb_classes, params['quick_test'])
            test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
            test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])

            test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_sed_gt, data_gen_test.nb_frames_1s())
            test_doa_loss = evaluation_metrics.compute_doa_scores_regr_xyz(test_doa_pred, test_doa_gt, test_sed_pred, test_sed_gt)
            test_metric_loss = evaluation_metrics.compute_seld_metric(test_sed_loss, test_doa_loss)

            # new SELD scores
            cls_new_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=data_gen_test.get_nb_classes(), doa_threshold=20)
            test_pred_dict = evaluation_metrics.regression_label_format_to_output_format(
                data_gen_test, test_sed_pred, test_doa_pred
            )
            test_gt_dict = evaluation_metrics.regression_label_format_to_output_format(
                data_gen_test, test_sed_gt, test_doa_gt
            )

            test_pred_blocks_dict = segment_labels(test_pred_dict, test_sed_pred.shape[0], data_gen_test.nb_frames_1s())
            test_gt_blocks_dict = segment_labels(test_gt_dict, test_sed_gt.shape[0], data_gen_test.nb_frames_1s())

            cls_new_metric.update_seld_scores_xyz(test_pred_blocks_dict, test_gt_blocks_dict)
            test_new_metric = cls_new_metric.compute_seld_scores()
            test_new_seld_metric = evaluation_metrics.compute_seld_metric(test_new_metric[:2], test_new_metric[2:])

            avg_scores_test.append([test_new_metric[0], test_new_metric[1], test_new_metric[2], test_new_metric[3], test_new_seld_metric])
            print('Results on test split:')
            print('\tOld Metrics: ER: {:0.2f}, F: {:0.2f}, DE: {:0.2f}, DE_FR: {:0.2f}, SELD: {:0.2f}'.format(test_sed_loss[0], test_sed_loss[1], test_doa_loss[0], test_doa_loss[1], test_metric_loss))
            print('\tNew Metrics: ER: {:0.2f}, F: {:0.2f}, DE: {:0.2f}, DE_FR: {:0.2f}, SELD: {:0.2f}'.format(test_new_metric[0], test_new_metric[1], test_new_metric[2], test_new_metric[3], test_new_seld_metric))

    print('\n\nValidation split scores per fold:')
    for cnt in range(len(val_splits)):
        print('\tSplit {} - SED ER: {:0.2f} F1: {:0.2f}; DOA error: {:0.2f} frame recall: {:0.2f}; SELD score: {:0.2f}'.format(cnt, avg_scores_val[cnt][0], avg_scores_val[cnt][1], avg_scores_val[cnt][2], avg_scores_val[cnt][3], avg_scores_val[cnt][4]))

    if params['mode'] is 'dev':
        print('\n\nTesting split scores per fold:')
        for cnt in range(len(val_splits)):
            print('\tSplit {} - SED ER: {:0.2f} F1: {:0.2f}; DOA error: {:0.2f} frame recall: {:0.2f}; SELD score: {:0.2f}'.format(cnt, avg_scores_test[cnt][0], avg_scores_test[cnt][1], avg_scores_test[cnt][2], avg_scores_test[cnt][3], avg_scores_test[cnt][4]))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
