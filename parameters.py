# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
# 用于特征提取、神经网络模型和训练SELDnet的参数可以在这里更改。
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
# 理想情况下，不要更改默认参数的值。如下面的代码（if-else循环）所示，使用唯一的<task-id>创建单独的案例并使用它们。通过这种方式，您可以在以后的时间轻松地复制配置。

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    # ########### 默认参数 ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH 输入路径
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders 包含foa/mic和元数据文件夹的基本文件夹，数据集位置
        dataset_dir='/scratch/asignal/partha/DCASE2023/DCASE2023_SELD_dataset',

        # OUTPUT PATHS 输出路径
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels 要转储提取的特征和标签的目录，特征标签位置
        feat_label_dir='/scratch/asignal/partha/DCASE2023/DCASE2023_SELD_dataset/seld_feat_label',
 
        model_dir='models/',            # Dumps the trained models and training curves in this folder 将经过训练的模型和训练曲线转储到此文件夹中，模型存储位置
        dcase_output_dir='results/',    # recording-wise results are dumped in this path. 记录方面的结果被转储在此路径中。输出结果存储位置

        # DATASET LOADING PARAMETERS
        # 数据集加载参数
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset 评估数据集
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals 数据集格式

        #FEATURE PARAMS
        # 特征参数
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features 仅用于MIC数据集。如果为true，则使用salsalite功能，否则使用GCC功能
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        # 模型类别
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA 是否为多ACCDOA格式
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.仅适用于Multi-ACCDOA。以度为单位的推理的统一阈值。

        # DNN MODEL PARAMETERS
        # DNN模型参数
        label_sequence_length=50,    # Feature sequence length 特征序列长度
        batch_size=128,              # Batch size 批量大小
        dropout_rate=0.05,           # Dropout rate, constant for all layers 脱落率，每层不变
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer CNN节点数，每层不变
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer CNN池化频率，列表长度=CNN层数，列表值=每层池

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes FNN内容，列表长度=层数，列表值=节点数

        nb_epochs=100,              # Train for maximum epochs 最大迭代训练次数
        lr=1e-3,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average 支持“micro”：样本平均值和“macro”：类平均值
        lad_doa_thresh=20
    )

    # ########### User defined parameters ##############
    # ########### 用户定义参数 ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n") # 使用默认参数

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
