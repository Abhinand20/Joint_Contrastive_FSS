# Joint_Contrastive_FSS

### Steps to run training on MRI-abd dataset
###### SETUP
1. Clone this repository and change the working directory to the root directory of this project `Joint_Contrastive_FSS/`
2. Copy the `data` folder from `/local/scratch/v_abhinand_jha/` directory on Lab's server and paste it to the root directory of this project
3. Copy the denseCL backbone weights from `/local/scratch/v_abhinand_jha/densecl_r101_imagenet_200ep.pth` on Lab's server and paste it to `./models/backbone` in the project directory

###### TRAINING
4. Run `bash train_ft_joint.sh` to start the cross validation training, configs can be updated in the bash script

##### VALIDATION
5. Open `validate_ft_joint.sh` and configure the trained model path in `RELOAD_PATH` variable eg. `RELOAD_PATH='./exps/exp_lbl1_0/train_unsup_CHAOST2_lbgroup0_vfold1/1/snapshots/seg_best.pth'`
6. Run `bash validate_ft_joint.sh` to get validation results for the trained model on all folds
7. The cross-validation DICE scores will be stored as a CSV file in `./results/` directory
