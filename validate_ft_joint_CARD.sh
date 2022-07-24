#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=2
export CUDA_VISIBLE_DEVICES=$GPUID1


####### Shared configs
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="exp_CARD"
DATASET='C0' ## Abdominal-MRI
NWORKER=0
ALL_EV=( 0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4) 
SEED='1234'
MIN_FG=1
### Use LV_BP as testing class
LABEL_SETS=0 
EXCLU='[]' 

# LABEL_SETS=1 
# EXCLU='[]' 

# LABEL_SETS=2
# EXCLU='[]' 

###### 1st optimization configs ######
NSTEP=15100
DECAY=0.95
MAX_ITERS_PER_LOAD=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=2000 # interval for checking validation dice


# # Use true GT Masks for evaluation
cp ./data/C0/normalized_labels/* ./data/C0/C0_normalized
    
for EVAL_FOLD in "${ALL_EV[@]}"
    do
        
        echo ===================================
        

        echo "Start validation for fold : ${EVAL_FOLD}"
        LOGDIR="./exps/${CPT}_${LABEL_SETS}/"
        TRAIN_PREFIX="train_unsup_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}"
        LATEST_RUN=$(ls "${LOGDIR}${TRAIN_PREFIX}" -tp | head -1)
        RELOAD_PATH="./exps/${CPT}_${LABEL_SETS}/train_unsup_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}/${LATEST_RUN}/snapshots/seg_best.pth" 
        PREFIX="valid_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}"
        echo $PREFIX
        
        
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
        min_fg_data=$MIN_FG seed=$SEED \
        save_snapshot_every=$SNAPSHOT_INTERVAL \
        superpix_scale=$SUPERPIX_SCALE \
        lr_step_gamma=$DECAY \
        path.log_dir=$LOGDIR \
        reload_model_path=$RELOAD_PATH

        echo "Finished validation for fold : ${EVAL_FOLD}"

    done

echo "Calculating CV scores"

python3 exp_evaluate.py $DATASET \
$LABEL_SETS \
${#ALL_EV[@]} \
"./exps/${CPT}_${LABEL_SETS}/valid_${DATASET}_lbgroup${LABEL_SETS}_vfold" \
"${CPT}_${LABEL_SETS}"

echo "CV scores for ${CPT}_${LABEL_SETS} saved!"