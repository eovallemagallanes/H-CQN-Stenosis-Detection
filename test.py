import json
#import pandas as pd
#import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# PyTorch
import torch

# TODO: 
# ONLY WORK FOR BATCH OF SIZE=1
# FIX TO WORK FOR ANY BATCH SIZE
def test_model(device, model, test_loader, idx2class=None):
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]
    
    return (y_true_list, y_pred_list) 

    

def eval_preds(y_true_list,y_pred_list,PATH_RESULTS, idx2class=None):

    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list).ravel()
    print('tn=%d, fp=%d, fn=%d, tp=%d' % (tn, fp, fn, tp))
    accuracy = accuracy_score(y_true_list, y_pred_list)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true_list, y_pred_list)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true_list, y_pred_list)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true_list, y_pred_list)
    print('F1 score: %f' % f1)
    specificity = tn / (tn + fp)
    print('Specificity: %f ' % specificity)
    
    results = {'tp': float(tp), 'tn': float(tn), 'fn': float(fn), 'fp': float(fp),
        'accuracy': accuracy, 'precision': precision, 'recal': recall, 
        'f1-score': f1, 'specificity': specificity
    }
    
    # Serializing json    
    with open(PATH_RESULTS, "w") as outfile:  
        json.dump(results, outfile)  
        

# get statistics from a folder of test_runs
def get_statistics(checkpoint, num_runs, test_stage):
    
    # training time statistics
    all_times = []
    for i in range(num_runs):
        file = '%s/model_history_time_%02d.json' %(checkpoint, i)
        with open(file) as json_file:
            data = json.load(json_file)
            all_times.append(data['time'])
            
    all_times =  np.asarray(all_times)
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)
    print('TRAINING TIME: {:.0f}m {:.0f}s (+-/ {:.0f}m {:.0f}s)'.format(mean_time // 60, mean_time % 60, std_time // 60, std_time % 60 ))
    
    
    if test_stage:
        # test metrics statistics 
        all_acc = []
        all_prec = []
        all_rec = []
        all_f1 = []
        all_spec = []
        
        
        for i in range(num_runs):
            file = '%s/model_results_%02d.json' %(checkpoint, i)
            with open(file) as json_file:
                data = json.load(json_file)
                all_acc.append(data['accuracy']*100.0)
                all_prec.append(data['precision']*100.0)
                all_rec.append(data['recal']*100.0)
                all_f1.append(data['f1-score']*100.0)
                all_spec.append(data['specificity']*100.0)
        
        all_acc = np.asarray(all_acc)
        all_prec = np.asarray(all_prec)
        all_rec = np.asarray(all_rec)
        all_f1 = np.asarray(all_f1)
        all_spec = np.asarray(all_spec)
        
        print("ACC: %.2f%% (+/- %.2f%%)" % (np.mean(all_acc), np.std(all_acc)))
        print("PC: %.2f%% (+/- %.2f%%)" % (np.mean(all_prec), np.std(all_prec)))
        print("SN: %.2f%% (+/- %.2f%%)" % (np.mean(all_rec), np.std(all_rec)))
        print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(all_f1), np.std(all_f1)))
        print("SP: %.2f%% (+/- %.2f%%)" % (np.mean(all_spec), np.std(all_spec)))
    

    
    