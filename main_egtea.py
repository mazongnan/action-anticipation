"""Main training/test program for RULSTM"""
import csv
import heapq
import os
from argparse import ArgumentParser

import cv2

from dataset import SequenceDataset
from os.path import join
from egtea_model_s import Network_s
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utils import topk_accuracy, ValueMeter, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax,  topk_recall_multiple_timesteps, tta, predictions_to_json, MeanTopKRecallMeter
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
pd.options.display.float_format = '{:05.2f}'.format

parser = ArgumentParser(description="Training program for RULSTM")

parser.add_argument('--device_ids', nargs='+', default=[3], type=int)
parser.add_argument('--gpu_ids', default='0,1,2,3', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')

parser.add_argument('--mode', type=str, choices=['train', 'validate', 'test', 'validate_json'], default='train',
                    help="Whether to perform training, validation or test.\
                            If test is selected, --json_directory must be used to provide\
                            a directory in which to save the generated jsons.")
parser.add_argument('--path_to_data', type=str,default= '/home/zhangfuchun/mazongnan/RULSTM/data/egtea/',
                    help="Path to the data folder, \
                            containing all LMDB datasets")
parser.add_argument('--path_to_models', type=str,default= '/home/zhangfuchun/mazongnan/RULSTM/models_anticipation/egtea/',
                    help="Path to the directory where to save all models")
parser.add_argument('--alpha', type=float, default=0.25,
                    help="Distance between time-steps in seconds")
parser.add_argument('--S_enc', type=int, default=6,
                    help="Number of encoding steps. \
                            If early recognition is performed, \
                            this value is discarded.")
parser.add_argument('--S_ant', type=int, default=8,
                    help="Number of anticipation steps. \
                            If early recognition is performed, \
                            this is the number of frames sampled for each action.")
parser.add_argument('--task', type=str, default='anticipation', choices=[
                    'anticipation', 'early_recognition'], help='Task to tackle: \
                            anticipation or early recognition')
parser.add_argument('--img_tmpl', type=str,
                    default='frame_{:010d}.jpg', help='Template to use to load the representation of a given frame')
parser.add_argument('--modality', type=str, default='fusion',
                    choices=['rgb', 'flow', 'fusion'], help = "Modality. Rgb/flow/obj represent single branches, whereas fusion indicates the whole model with modality attention.")
parser.add_argument('--sequence_completion', action='store_true',
                    help='A flag to selec sequence completion pretraining rather than standard training.\
                            If not selected, a valid checkpoint for sequence completion pretraining\
                            should be available unless --ignore_checkpoints is specified')
parser.add_argument('--mt5r', action='store_true')

parser.add_argument('--num_class', type=int, default=106, help='Number of classes')
parser.add_argument('--verb_class', type=int, default=19, help='')
parser.add_argument('--noun_class', type=int, default=51, help='')
parser.add_argument('--hidden', type=int, default=1024,
                    help='Number of hidden units')
parser.add_argument('--feat_in', type=int, default=1024,
                    help='Input size. If fusion, it is discarded (see --feats_in)')
parser.add_argument('--feats_in', type=int, nargs='+', default=[1024, 1024],
                    help='Input sizes when the fusion modality is selected.')
parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate")

parser.add_argument('--schedule_on', type=int, default=1, help='')
parser.add_argument('--schedule_epoch', type=int, default=3, help='')
parser.add_argument('--schedule_rate', type=int, default=0.3, help='')

parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")  #128
parser.add_argument('--num_workers', type=int, default=16,
                    help="Number of parallel thread to fetch the data")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")

parser.add_argument('--display_every', type=int, default=100,
                    help="Display every n iterations")
parser.add_argument('--epochs', type=int, default=60, help="Training epochs")

parser.add_argument('--ignore_checkpoints', action='store_true',default='False',
                    help='If specified, avoid loading existing models (no pre-training)')
parser.add_argument('--resume', action='store_true',default='False',
                    help='Whether to resume suspended training')

parser.add_argument('--ek100', action='store_true',
                    help="Whether to use EPIC-KITCHENS-100")

parser.add_argument('--json_directory', type=str, default = None, help = 'Directory in which to save the generated jsons.')

args = parser.parse_args()

if args.mode == 'test' or args.mode=='validate_json':
    assert args.json_directory is not None

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[1])

if args.task == 'anticipation':
    exp_name = f"{args.task}_{args.alpha}_{args.S_enc}_{args.S_ant}_{args.modality}"
else:
    exp_name = f"{args.task}_{args.alpha}_{args.S_ant}_{args.modality}"

if args.mt5r:
    exp_name += '_mt5r'

if args.sequence_completion:
    exp_name += '_sequence_completion'

def get_many_shot():
    """Get many shot verbs, nouns and actions for class-aware metrics (Mean Top-5 Recall)"""
    # read the list of many shot verbs
    many_shot_verbs = pd.read_csv(join(args.path_to_data, 'EGTEA_shot_verbs.csv'))['verb_class'].values
    # read the list of many shot nouns
    many_shot_nouns = pd.read_csv(
        join(args.path_to_data, 'EGTEA_shot_nouns.csv'))['noun_class'].values

    # read the list of actions
    actions = pd.read_csv(join(args.path_to_data, 'actions_re.csv'))
    # map actions to (verb, noun) pairs
    a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
               for a in actions.iterrows()}

    # create the list of many shot actions
    # an action is "many shot" if at least one
    # between the related verb and noun are many shot
    many_shot_actions = []
    for a, (v, n) in a_to_vn.items():
        if v in many_shot_verbs or n in many_shot_nouns:
            many_shot_actions.append(a)

    return many_shot_verbs, many_shot_nouns, many_shot_actions

def get_loader(mode, override_modality = None):
    if override_modality:
        path_to_lmdb = join(args.path_to_data, override_modality)
    else:
        path_to_lmdb = join(args.path_to_data, args.modality) if args.modality != 'fusion' else [join(args.path_to_data, m) for m in ['rgb', 'flow']]

    kargs = {
        'path_to_lmdb': path_to_lmdb,
        'path_to_csv': join(args.path_to_data, f"{mode}.csv"),
        'time_step': args.alpha,
        'img_tmpl': args.img_tmpl,
        'action_samples': args.S_ant,
        'past_features': args.task == 'anticipation',
        'sequence_length': args.S_enc + args.S_ant,
        'label_type': ['verb', 'noun', 'action'] if args.mode != 'train' else 'action',
        'challenge': 'test' in mode
    }

    _set = SequenceDataset(**kargs)

    return DataLoader(_set, batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, shuffle=mode == 'training')

def get_model(): # single branch
    model = Network_s(args)

    # load checkpoint only if not in sequence completion mode
    # and inf the flag --ignore_checkpoints has not been specified
    if args.mode == 'train' and not args.ignore_checkpoints and not args.sequence_completion:
        checkpoint = torch.load(join(
            args.path_to_models, exp_name + '_sequence_completion_best.pth.tar'))['state_dict']
        model.load_state_dict(checkpoint)

    return model


def load_checkpoint(model, best=False):
    if best:
        chk = torch.load(join(args.path_to_models, exp_name + '_best.pth.tar'))
    else:
        chk = torch.load(join(args.path_to_models, exp_name + '.pth.tar'))

    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf


def save_model(model, epoch, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'perf': perf, 'best_perf': best_perf}, join(args.path_to_models, exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, join(
            args.path_to_models, exp_name + '_best.pth.tar'))


def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')


def get_scores(model, loader, challenge=False, include_discarded = False):
    model.eval()
    predictions = []
    labels = []
    ids = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(loader, 'Evaluating...', len(loader)):
            x = batch['past_features' if args.task ==
                                         'anticipation' else 'action_features']
            x_f = batch['action_features']
            if type(x) == list:
                x = [xx.cuda() for xx in x]
            else:
                x = x.cuda()

            if type(x_f) == list:
                x_f = [xxx.cuda() for xxx in x_f]
            else:
                x_f = x_f.cuda()

            y = batch['label'].numpy()

            ids.append(batch['id'])

            preds = model(x[0], x[1], x_f[0], x_f[1]).cpu().numpy()[:, -args.S_ant:, :]

            predictions.append(preds)
            labels.append(y)

    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    actions = pd.read_csv(
        join(args.path_to_data, 'actions_re.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))

    verb_scores = marginalize(action_probs, vi).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)
    noun_scores = marginalize(action_probs, ni).reshape(
        action_scores.shape[0], action_scores.shape[1], -1)

    if include_discarded:
        dlab = np.array(loader.dataset.discarded_labels)
        dislab = np.array(loader.dataset.discarded_ids)
        ids = np.concatenate([ids, dislab])
        num_disc = len(dlab)
        labels = np.concatenate([labels, dlab])
        verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
        noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
        action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

    if labels.max()>0 and not challenge:
        return verb_scores, noun_scores, action_scores, labels[:, 0], labels[:, 1], labels[:, 2], ids
    else:
        return verb_scores, noun_scores, action_scores, ids


def trainval(model, loaders, optimizer, epochs, start_epoch, start_best_perf,schedule_on):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        if args.mt5r:
            accuracy_meter = {'training': MeanTopKRecallMeter(args.num_class), 'validation': MeanTopKRecallMeter(args.num_class)}
        else:
            accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['past_features' if args.task ==
                                                 'anticipation' else 'action_features']
                    x_f = batch['action_features']
                    if type(x) == list:
                        x = [xx.cuda() for xx in x]
                    else:
                        x = x.cuda()

                    if type(x_f) == list:
                        x_f = [xxx.cuda() for xxx in x_f]
                    else:
                        x_f = x_f.cuda()

                    #
                    # x = batch['past_features' if args.task ==
                    #           'anticipation' else 'action_features']
                    #
                    # if type(x) == list:
                    #     x = [xx.cuda() for xx in x]
                    # else:
                    #     x = x.cuda()

                    y = batch['label'].cuda()

                    bs = y.shape[0]  # batch size

                    # preds,preds_v,preds_n = model(x[0],x[1],x[2])
                    preds= model(x[0], x[1], x_f[0], x_f[1])


                    # take only last S_ant predictions
                    preds = preds[:, -args.S_ant:, :].contiguous()

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])
                    # replicate the labels across timesteps and linearize
                    linear_labels = y.view(-1, 1).expand(-1,
                                                         preds.shape[1]).contiguous().view(-1)

                    loss = F.cross_entropy(linear_preds, linear_labels)
                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    idx = -4 if args.task == 'anticipation' else -1
                    # use top-5 for anticipation and top-1 for early recognition
                    k = 5 if args.task == 'anticipation' else 1
                    acc = topk_accuracy(
                        preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0]*100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    if args.mt5r:
                        accuracy_meter[mode].add(preds[:, idx, :].detach().cpu().numpy(),
                                                 y.detach().cpu().numpy())
                    else:
                        accuracy_meter[mode].add(acc, bs)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])

                # log at the end of each epoch
                log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode],
                    max(accuracy_meter[mode].value(), best_perf) if mode == 'validation'
                    else None, green=True)
        schedule_on.step()
        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch+1, accuracy_meter['validation'].value(), best_perf,
                   is_best=is_best)

def get_validation_ids():
    unseen_participants_ids = pd.read_csv(join(args.path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
    tail_verbs_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
    tail_nouns_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
    tail_actions_ids = pd.read_csv(join(args.path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

    return unseen_participants_ids, tail_verbs_ids, tail_nouns_ids, tail_actions_ids


def main():
    model = get_model()

    model.cuda()

    if args.mode == 'train':
        loaders = {m: get_loader(m) for m in ['training', 'validation']}

        start_epoch = 0
        start_best_perf = 0

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum)
        schedule_on = lr_scheduler.StepLR(optimizer, args.schedule_epoch, gamma=args.schedule_rate, last_epoch=-1)
        trainval(model, loaders, optimizer, args.epochs,
                 start_epoch, start_best_perf,schedule_on)

    elif args.mode == 'validate':
        epoch, perf, _ = load_checkpoint(model, best=True)
        print(
            f"Loaded checkpoint for model {type(model)}. Epoch: {epoch}. Perf: {perf:0.2f}.")

        loader = get_loader('validation')

        verb_scores, noun_scores, action_scores, verb_labels, noun_labels, action_labels, ids = get_scores(model, loader, include_discarded=args.ek100)

        verb_accuracies = topk_accuracy_multiple_timesteps(
            verb_scores, verb_labels)
        noun_accuracies = topk_accuracy_multiple_timesteps(
            noun_scores, noun_labels)
        action_accuracies = topk_accuracy_multiple_timesteps(
            action_scores, action_labels)

        many_shot_verbs, many_shot_nouns, many_shot_actions = get_many_shot()

        noun_recalls = topk_recall_multiple_timesteps(
            noun_scores, noun_labels, k=5, classes=many_shot_nouns)
        verb_recalls = topk_recall_multiple_timesteps(
            verb_scores, verb_labels, k=5, classes=many_shot_verbs)
        action_recalls = topk_recall_multiple_timesteps(
            action_scores, action_labels, k=5, classes=many_shot_actions)

        all_accuracies = np.concatenate(
            [verb_accuracies, noun_accuracies, action_accuracies, verb_recalls, noun_recalls, action_recalls])
        all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
        indices = [
            ('Verb', 'Top-1 Accuracy'),
            ('Verb', 'Top-5 Accuracy'),
            ('Verb', 'Mean Top-5 Recall'),
            ('Noun', 'Top-1 Accuracy'),
            ('Noun', 'Top-5 Accuracy'),
            ('Noun', 'Mean Top-5 Recall'),
            ('Action', 'Top-1 Accuracy'),
            ('Action', 'Top-5 Accuracy'),
            ('Action', 'Mean Top-5 Recall'),
        ]
        cc = np.linspace(args.alpha*args.S_ant, args.alpha, args.S_ant, dtype=str)
        scores = pd.DataFrame(all_accuracies*100, columns=cc, index=pd.MultiIndex.from_tuples(indices))
        print(scores)

        if args.task == 'anticipation':
            tta_verb = tta(verb_scores, verb_labels)
            tta_noun = tta(noun_scores, noun_labels)
            tta_action = tta(action_scores, action_labels)

            print(
                f"\nMean TtA(5): VERB: {tta_verb:0.2f} NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}")


if __name__ == '__main__':
    main()
