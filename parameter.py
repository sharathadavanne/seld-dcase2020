# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='/home/sharath/Downloads/SELD_2019_dataset',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        feat_label_dir='/home/sharath/Downloads/SELD_2019_dataset/feat_label/',  # Directory to dump extracted features and labels
        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',       # 'foa' - ambisonic or 'mic' - microphone signals

        # DNN MODEL PARAMETERS
        sequence_length=128,        # Feature sequence length
        batch_size=16,              # Batch size
        dropout_rate=0,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        pool_size=[4, 4, 4],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=50,               # Train for maximum epochs
        epochs_per_fit=1,           # Number of epochs per fit
        doa_objective='masked_mse',     # supports: mse, masked_mse. mse- original seld approach; masked_mse - dcase 2020 approach
        start_masked_epoch=5        # if doa_objective==masked, the masking is applied only after 'start_masked_epoch' epochs
    )
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience is reached

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '3':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '4':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '5':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    # Quick test
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
