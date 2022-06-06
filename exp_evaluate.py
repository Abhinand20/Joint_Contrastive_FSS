import os.path as osp
import os
import pandas as pd
import argparse
from collections import defaultdict
import json
from dataloaders.dataset_utils import DATASET_INFO
import csv

# logdir + prefix + fold_no for each valid dir
# Open metrics.json , 
# for every eval_fold:
#   visit all dirs, get the curr eval_fold metrics and store them

def save_cv_results(args):
    baseset_name = args.baseset_name
    label_sets = args.label_sets
    eval_folds = args.eval_folds
    run_path = args.run_path
    exp_name = args.exp_name
    dict_res = {}
    lbls = DATASET_INFO[baseset_name]['REAL_LABEL_NAME']
    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][label_sets]
    output_path = osp.join('results','{}_results.csv'.format(exp_name))

    if not osp.exists('./results'):
        os.mkdir('./results')

    if osp.exists(output_path):
        os.remove(output_path)

    
    for val_fold in range(eval_folds):    
        eval_dict = {}
        curr_path = osp.join(run_path + str(val_fold),"1","metrics.json")
        with open(curr_path,'r') as f:
            data = json.load(f)
            mean_dice = data['mar_val_batches_meanDice']['values'][0]
            class_dice = data['mar_val_batches_classDice']['values'][0]
            for i,tst_lbl in enumerate(test_labels):
                eval_dict[lbls[tst_lbl]] = class_dice[i]
            
            eval_dict["Mean"] = mean_dice

        dict_res[val_fold] = eval_dict

    
    df = pd.DataFrame(dict_res).T
    avgs = {}
    for col in df.columns:
        avgs[col] = df[col].mean()

    dict_res["Average"] = avgs

    pd.DataFrame(dict_res).T.to_csv(output_path,mode='w')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("baseset_name", type=str,
                        help="path of data files")
    parser.add_argument("label_sets", type=int, default=0,
                        help="training label set")
    parser.add_argument("eval_folds", type=int,
                        help="training label set")
    parser.add_argument("run_path", type=str,
                        help="dir of stored val results")
    parser.add_argument("exp_name", type = str, help = "to store output file")
    args = parser.parse_args()

    save_cv_results(args)