#!/bin/bash
# Joint training 
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1


####### Shared configs
USE_INTER_LOSS=0
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="exp_CT" # Name of current experiment
DATASET='SABS' ## Abdominal-MRI
NWORKER=0
ALL_EV=( 0 1 2 3 ) # 4-fold cross validation (0, 1, 2, 3) 
SEED='1234'
### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[]' 


for EVAL_FOLD in "${ALL_EV[@]}"
do
    echo ===================================
    
    echo "Start training for fold : ${EVAL_FOLD}"

    PREFIX="train_sup_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/${CPT}_${LABEL_SETS}/"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi
    # Reset dataset before starting training
    cp ./data/SABS/normalized_labels/* ./data/SABS/sabs_CT_normalized
    ###### 1st optimization configs ######
    NSTEP=10100
    DECAY=0.95
    MAX_ITERS_PER_LOAD=1000 # defines the size of an epoch
    RELOAD_PATH=''
    SNAPSHOT_INTERVAL=4000 # interval for checking validation dice and saving best model

    python3 tr_ssl.py with \
    'modelname=densecl_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITERS_PER_LOAD \
    min_fg_data=100 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    reload_model_path=$RELOAD_PATH
    
    ### 2nd Optimization configs ###
    # Now start joint unsupervised training
    # ls <path> -tp | head -1 : Will give most recent run directory
    LATEST_RUN=$(ls "${LOGDIR}${PREFIX}" -tp | head -1)    
    RELOAD_PATH="${LOGDIR}${PREFIX}/${LATEST_RUN}snapshots/ft_final.pth"
    PREFIX="train_unsup_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}"
    MAX_ITERS_PER_LOAD=5000
    TOTAL_ITERS=3
    
    python3 tr_contrastive.py with \
    'modelname=densecl_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITERS_PER_LOAD \
    min_fg_data=100 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    reload_model_path=$RELOAD_PATH \
    total_iters=$TOTAL_ITERS

    echo "Finished training for fold : ${EVAL_FOLD}"
done
