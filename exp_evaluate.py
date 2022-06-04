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

    for train_fold in range(1,eval_folds + 1):
        train_dict = {}
        # {class1 : score, class2 : score, mean: score} #mar_val_batches_classDice, mar_val_batches_meanDice
        for val_fold in range(eval_folds):    
            eval_dict = {}
            curr_path = osp.join(run_path + str(val_fold),str(train_fold),"metrics.json")
            with open(curr_path,'r') as f:
                data = json.load(f)
                mean_dice = data['mar_val_batches_meanDice']['values'][0]
                class_dice = data['mar_val_batches_classDice']['values'][0]
                for i,tst_lbl in enumerate(test_labels):
                    eval_dict[lbls[tst_lbl]] = class_dice[i]
                
                eval_dict["Mean"] = mean_dice

            train_dict[val_fold] = eval_dict

        dict_res[train_fold-1] = train_dict


    # dict_res -> {train_fold0 : {eval_fold0 : {scores}, eval_fold1 : {scores} ... } , train_fold1 : ...}
    for train_fold in range(eval_folds):
        format_str = [""]*(len(test_labels) + 1)
        with open(output_path,'a') as f:
            write_str = ["Trained on Fold {}".format(train_fold + 1)]
            write_str.extend(format_str)
            writer = csv.writer(f)
            writer.writerow(write_str)   
        df = pd.DataFrame(dict_res[train_fold]).T
        avgs = {}
        for col in df.columns:
            avgs[col] = df[col].mean()

        dict_res[train_fold]["Average"] = avgs

        pd.DataFrame(dict_res[train_fold]).T.to_csv(output_path,mode='a')

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