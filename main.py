from __future__ import print_function
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np
import argparse
import torch
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def seed_torch(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

seed_torch(40)
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

parser.add_argument('dataset_name', type=str,
                    choices=['rcc', 'nsclc', 'ubc_ocean', 'camelyon', 'camelyon17'],
                    help='dataset name: rcc, nsclc, ubc_ocean, camelyon, camelyon17')
parser.add_argument('shots', type=int, choices=[16, 8, 4],
                    help='number of shots per class: 16, 8 or 4')
parser.add_argument('--data_root', type=str, required=True,
                    help='root folder that holds <subdir>/feature/conch_2048 (5x) and '
                         '<subdir>/feature/conch (20x) for every dataset')
parser.add_argument('--results_root', type=str, default='./results/tmp',
                    help='folder that receives the per-run results (default: ./results/tmp)')
parser.add_argument('--conch_ckpt', type=str, required=True,
                    help='path to the CONCH checkpoint (conch.bin)')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, last fold)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd', 'lookahead_radam'], default='adam')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--mode', type=str, default='transformer', choices=['transformer'],
                    help="pipeline branch; only 'transformer' is supported by the current two-scale code")
parser.add_argument('--camelyon_patch', action='store_true', default=False,
                    help='also report patch-level metrics on C16/C17 (requires --patch_label_dir)')
parser.add_argument('--patch_label_dir', type=str, default=None,
                    help='folder that holds {slide_id}_label.pt, the patch-level ground truth')
parser.add_argument('--patch_result_csv', type=str, default=None,
                    help='csv file to append the patch-level metrics to (optional)')

args = parser.parse_args()

dataset_name = args.dataset_name
shots = args.shots

# Per-dataset conventions: sub-folder under --data_root, name of the 20x feature folder,
# prompt csv and task name. The 5x folder is always <subdir>/feature/conch_2048.
DATASETS = {
    'rcc': dict(subdir='rcc', feat_l='conch', prompt='text_prompt/TCGA_RCC_two_scale_text_prompt.csv',
                task='task_tcga_rcc_subtyping'),
    'camelyon': dict(subdir='c16', feat_l='conch', prompt='text_prompt/CAMELYON_two_scale_text_prompt.csv',
                     task='task_camelyon_subtyping'),
    'camelyon17': dict(subdir='c17', feat_l='conch', prompt='text_prompt/CAMELYON_two_scale_text_prompt.csv',
                       task='task_camelyon17_subtyping'),
    'nsclc': dict(subdir='nsclc', feat_l='conch', prompt='text_prompt/TCGA_Lung_two_scale_text_prompt.csv',
                  task='task_tcga_lung_subtyping'),
    'ubc_ocean': dict(subdir='ubc_ocean', feat_l='conch_512',
                      prompt='text_prompt/UBC-OCEAN_two_scale_text_prompt.csv',
                      task='task_ubc_ocean_subtyping'),
}
spec = DATASETS[dataset_name]          # argparse `choices` already rejects unknown names
args.task = spec['task']
args.text_prompt_path = spec['prompt']
feature_root = os.path.join(args.data_root, spec['subdir'], 'feature')
args.data_folder_s = os.path.join(feature_root, 'conch_2048')
args.data_folder_l = os.path.join(feature_root, spec['feat_l'])

args.split_dir = f"{dataset_name}_{shots}shots_5folds"
args.exp_code = f"{dataset_name}_{shots}shots"
args.results_dir = os.path.join(args.results_root, dataset_name, str(shots))
args.results_testall = args.results_root
args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'reg': args.reg,
            'mode': args.mode,
            'opt': args.opt,
            'camelyon_patch': args.camelyon_patch,
            'data_root': args.data_root,
            'conch_ckpt': args.conch_ckpt}

print('\nLoad Dataset')

if args.task == 'task_tcga_rcc_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/rcc.csv',
                                  mode=args.mode,
                                  data_dir_s=args.data_folder_s,
                                  data_dir_l=args.data_folder_l,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'KICH': 0, 'KIRC': 1, 'KIRP': 2},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/nsclc.csv',
                                  mode=args.mode,
                                  data_dir_s=args.data_folder_s,
                                  data_dir_l=args.data_folder_l,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'LUAD': 0, 'LUSC': 1},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'task_ubc_ocean_subtyping':
    args.n_classes = 5
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/ubc_ocean.csv',
                                  mode=args.mode,
                                  data_dir_s=args.data_folder_s,
                                  data_dir_l=args.data_folder_l,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'CC': 0, 'HGSC': 1, 'LGSC': 2, 'EC': 3, 'MC': 4},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'task_camelyon_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/c16.csv',
                                  mode=args.mode,
                                  data_dir_s=args.data_folder_s,
                                  data_dir_l=args.data_folder_l,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'normal': 0, 'tumor': 1},
                                  patient_strat=False,
                                  ignore=[])
elif args.task == 'task_camelyon17_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/c17.csv',
                                  mode=args.mode,
                                  data_dir_s=args.data_folder_s,
                                  data_dir_l=args.data_folder_l,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'Solid Tissue Normal': 0, 'Primary Tumor': 1},
                                  patient_strat=False,
                                  ignore=[])
else:
    raise NotImplementedError

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

if args.split_dir is None:
    raise ValueError('split_dir must be set')
args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir, 'results_dir': args.results_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_f1 = []
    folds = np.arange(start, end)
    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_f1.append(test_f1)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame(
        {'folds': folds, 'test_auc': all_test_auc, 'test_acc': all_test_acc, 'test_f1': all_test_f1})
    result_df = pd.DataFrame({f'metric_{dataset_name}_{shots}': ['mean', 'var'],
                              'test_auc': [round(np.mean(all_test_auc), 4), round(np.std(all_test_auc), 4)],
                              'test_acc': [round(np.mean(all_test_acc), 4), round(np.std(all_test_acc), 4)],
                              'test_f1':  [round(np.mean(all_test_f1), 4),  round(np.std(all_test_f1), 4)],
                              })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        result_name = 'result_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        result_name = 'result.csv'

    result_df.to_csv(os.path.join(args.results_testall, result_name), mode='a', index=False)
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
