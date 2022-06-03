#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1


####### Shared configs
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="exp_lbl1"
DATASET='CHAOST2' ## Abdominal-MRI
NWORKER=0
ALL_EV=( 0 1 2 3) # 4-fold cross validation (0, 1, 2, 3) 
SEED='1234'
### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[]' 

###### 1st optimization configs ######
NSTEP=15100
DECAY=0.95
MAX_ITERS_PER_LOAD=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=2000 # interval for checking validation dice


# Use true GT Masks for evaluation
cp ./data/CHAOST2/normalized_labels/* ./data/CHAOST2/chaos_MR_T2_normalized

for EVAL_FOLD in "${ALL_EV[@]}"
do
    
    echo ===================================
    

    echo "Start validation for fold : ${EVAL_FOLD}"

    RELOAD_PATH='./exps/exp_lbl1_0/train_unsup_CHAOST2_lbgroup0_vfold1/1/snapshots/seg_best.pth' # Feed the reload path for current model

    PREFIX="valid_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/${CPT}_${LABEL_SETS}/"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    python3 validation.py with \
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

    echo "Finished validation for fold : ${EVAL_FOLD}"

done
