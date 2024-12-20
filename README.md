# Joint_Contrastive_FSS

### Steps to run training on MRI-abd dataset
###### SETUP
1. Clone this repository and change the working directory to the root directory of this project `Joint_Contrastive_FSS/`
2. Copy the `data` folder from `/local/scratch/v_abhinand_jha/` directory on Lab's server and paste it to the root directory of this project
3. Copy the denseCL backbone weights from `/local/scratch/v_abhinand_jha/densecl_r101_imagenet_200ep.pth` on Lab's server and paste it to `./models/backbone` in the project directory

###### TRAINING
4. Run `bash train_ft_joint.sh` to start the cross validation training, configs can be updated in the bash script

##### VALIDATION
5. Run `bash validate_ft_joint.sh` to get validation results for the trained model on all folds
6. The cross-validation DICE scores will be stored as a CSV file in `./results/` directory

##### Related work
\# | Paper | Tags | Code (if any) 
--- | --- | --- | ---
1 | [Semi-supervised few-shot learning for medical image segmentation](https://arxiv.org/pdf/2003.08462.pdf) | |
2 | [PoissonSeg: Semi-Supervised Few-Shot Medical Image Segmentation via Poisson Learning](https://arxiv.org/pdf/2108.11694.pdf) | |
3 | [Uncertainty-Aware Semi-Supervised Few Shot Segmentation](https://arxiv.org/pdf/2110.08954.pdf) | |
4 | [Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels](https://openreview.net/pdf?id=8Uui49rOfc)
