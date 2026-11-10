import os

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch_optimizer import RAdam, Lookahead
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import torch.optim as optim
import math
from itertools import islice
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_tranformer(batch):
    img_s = torch.cat([item[0] for item in batch], dim=0)
    # coord_s = torch.cat([item[1] for item in batch], dim = 0)
    img_l = torch.cat([item[1] for item in batch], dim=0)
    # coord_l = torch.cat([item[3] for item in batch], dim = 0)
    label = torch.LongTensor([item[2] for item in batch])
    # return [img_s, coord_s, img_l, coord_l, label]
    return [img_s, img_l, label]



# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
#
# g = torch.Generator()
# g.manual_seed(40)
def get_split_loader(split_dataset, training=False):
    if training:
        loader = DataLoader(split_dataset, batch_size=1, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate_tranformer)
    else:
        loader = DataLoader(split_dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=collate_tranformer)

    return loader

def save_results(save_dir, model_type, split_id, all_labels, all_probs):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_type}.npz")

    # 如果文件已存在，读取已有内容
    if split_id == 0 and os.path.exists(save_path):
        results = {}
    elif os.path.exists(save_path):
        old_data = np.load(save_path, allow_pickle=True)
        results = dict(old_data)
    else:
        results = {}

    # 保存当前 split 的结果
    results[f"labels_{split_id}"] = np.array(all_labels)
    results[f"probs_{split_id}"] = np.array(all_probs)

    # 更新整体拼接的 all_labels 和 all_probs
    if "all_labels" in results:
        results["all_labels"] = np.concatenate([results["all_labels"], all_labels])
        results["all_probs"] = np.concatenate([results["all_probs"], all_probs])
    else:
        results["all_labels"] = np.array(all_labels)
        results["all_probs"] = np.array(all_probs)

    # 重新写回文件（覆盖，但保留所有折的数据）
    np.savez(save_path, **results)
    print(f"Saved split {split_id} results to {save_path}")

def save_roc_curve(all_labels, all_probs, save_path, positive_class=1):
    """
    绘制并保存 ROC 曲线
    all_labels: numpy array 或 list，真实标签
    all_probs: numpy array 或 torch tensor，预测概率或 logits，shape=(N, num_classes)
    save_path: 保存路径，例如 './roc_curve.png'
    positive_class: 正类索引，默认1
    """
    # 如果是 torch tensor
    # if isinstance(all_probs, torch.Tensor):
    #     all_probs = F.softmax(all_probs, dim=1).cpu().numpy()

    # 正类概率
    y_score = all_probs[:, positive_class]

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(all_labels, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘图
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    elif args.opt == 'lookahead_radam':
        radam_params = {
            'lr': args.lr,  # 学习率
            'betas': (0.9, 0.999),  # 如果为 null，可以使用默认值
            'eps': 1e-8,  # 如果为 null，使用默认
            'weight_decay': 1e-5
        }
        radam = RAdam(model.parameters(), **radam_params)
        lookahead_params = {
            'k': 6,  # 每 k 步同步慢权重
            'alpha': 0.5  # 慢权重更新步长
        }
        optimizer = Lookahead(radam, **lookahead_params)

    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  #all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  #indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)
            # all_test_ids.extend(val_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
