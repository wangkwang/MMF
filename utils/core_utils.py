import csv
from collections import Counter
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.metrics import auc as calc_auc
import ml_collections
from models.model import MMF


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:

    def __init__(self, patience=20, stop_epoch=60, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        # Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit Model...', end=' ')

    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.text_prompt = args.text_prompt
    config.conch_checkpoint_path = args.conch_ckpt
    model_dict = {'config': config, 'num_classes': args.n_classes}
    model = MMF(**model_dict)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    print_network(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")

    for name, param in model.named_parameters():
        print(f"{name:60} | requires_grad = {param.requires_grad}")

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=40, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:03d} | LR: {lr:.6f}")
        stop = validate(cur, epoch, model, val_loader, args.n_classes,
                        early_stopping, writer, args.results_dir)
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(
        args.mode, model, test_loader, args.n_classes,
        camelyon_patch=args.camelyon_patch,
        patch_label_dir=args.patch_label_dir,
        patch_result_csv=args.patch_result_csv)

    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, each_class_acc, test_f1


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    print('\n')
    for batch_idx, (data_s, data_l, label) in enumerate(loader):
        data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
        _, Y_hat, loss = model(data_s, data_l, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, results_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_idx, (data_s, data_l, label) in enumerate(loader):
            data_s, data_l, label = data_s.to(device, non_blocking=True), data_l.to(device, non_blocking=True), label.to(device, non_blocking=True)
            Y_prob, Y_hat, loss = model(data_s, data_l, label)

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            all_pred.append(Y_hat.item())
            all_label.append(label.item())

    val_error /= len(loader)
    val_loss /= len(loader)
    val_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc,
                                                                                            val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(mode, model, loader, n_classes, camelyon_patch=False,
            patch_label_dir=None, patch_result_csv=None):
    """Run inference over `loader` and return per-slide results and metrics.

    camelyon_patch=True additionally scores patch-level predictions against the
    per-slide ground truth stored in `patch_label_dir` ({slide_id}_label.pt) and
    appends the metrics to `patch_result_csv` (no file is written when it is None).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.
    test_f1 = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    all_preds_p = []
    all_gts_p = []
    all_probs_p = []

    patch_slide_ids = []
    if camelyon_patch:
        if not patch_label_dir or not os.path.isdir(patch_label_dir):
            raise ValueError('camelyon_patch=True requires --patch_label_dir pointing to the '
                             'folder that holds {slide_id}_label.pt')
        # 'xxx_label.pt' -> 'xxx'
        patch_slide_ids = [os.path.splitext(f)[0][:-6]
                           for f in os.listdir(patch_label_dir) if f.endswith('.pt')]

    if (mode == 'transformer'):
        for batch_idx, (data_s, data_l, label) in enumerate(loader):
            data_s, data_l, label = data_s.to(device), data_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                if camelyon_patch:
                    Y_prob, Y_hat, loss, patch_prob, patch_hat = model(data_s, data_l, label, camelyon_patch=camelyon_patch)
                else:
                    Y_prob, Y_hat, loss = model(data_s, data_l, label)

            if camelyon_patch and slide_id in patch_slide_ids:
                label_path = os.path.join(patch_label_dir, f"{slide_id}_label.pt")
                if not os.path.exists(label_path):
                    print(f"[Warning] Label file not found for slide {slide_id}, patch GT skipped")
                else:
                    gt_labels = torch.load(label_path, map_location='cpu')
                    all_gts_p.append(gt_labels)
                    all_probs_p.append(patch_prob.cpu())
                    all_preds_p.append(patch_hat.squeeze(dim=-1).cpu())

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

            all_pred.append(Y_hat.item())
            all_label.append(label.item())

        if camelyon_patch and len(all_gts_p) > 0:
            all_preds_p = torch.cat(all_preds_p).numpy()
            all_gts_p = torch.cat(all_gts_p).numpy()
            all_probs_p = torch.cat(all_probs_p).numpy()
            acc_p = accuracy_score(all_gts_p, all_preds_p)
            f1_p = f1_score(all_gts_p, all_preds_p, average='macro')
            auc_p = roc_auc_score(all_gts_p, all_probs_p[:, 1])

            print("Ground Truth distribution:", Counter(all_gts_p))
            print("Preds distribution:", Counter(all_preds_p))
            print(f"patch_acc:{acc_p:.4f}, patch_f1:{f1_p:.4f}, patch_auc:{auc_p:.4f}")

            if patch_result_csv:
                result_dir = os.path.dirname(os.path.abspath(patch_result_csv))
                os.makedirs(result_dir, exist_ok=True)
                row = [auc_p, acc_p, f1_p]
                is_new = not os.path.exists(patch_result_csv)
                with open(patch_result_csv, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    if is_new:
                        csv_writer.writerow(['AUC', 'ACC', 'F1'])
                    csv_writer.writerow(row)
                print(f'patch metrics appended to {patch_result_csv}')


        test_error /= len(loader)
        test_f1 = f1_score(all_label, all_pred, average='macro')

        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])

        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

        return patient_results, test_error, auc, acc_logger, test_f1
