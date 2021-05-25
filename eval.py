import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys

import robustbench as rb
import data
#from autopgd_train import apgd_train
import utils
from model_zoo.fast_models import PreActResNet18
import other_utils
import autoattack
from autopgd_train import apgd_train

eps_dict = {'cifar10': {'Linf': 8. / 255., 'L2': .5, 'L1': 12.},
    'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 255.}}


def eval_single_norm(model, x, y, norm='Linf', eps=8. / 255., bs=1000,
    log_path=None, verbose=True):
    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
        log_path=log_path)
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    #adversary.apgd.n_restarts = 1
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    #if verbose
    acc = rb.utils.clean_accuracy(model, x_adv, y, device='cuda')
    other_utils.check_imgs(x_adv, x, norm)
    print('robust accuracy: {:.1%}'.format(acc))
    return x_adv


def eval_norms(model, x, y, l_norms, l_epss, bs=1000, log_path=None, n_cls=10):
    l_x_adv = []
    acc_dets = []
    logger = other_utils.Logger(log_path)
    for norm, eps in zip(l_norms, l_epss):
        x_adv_curr = eval_single_norm(model, x, y, norm=norm, eps=eps, bs=bs,
            log_path=log_path, verbose=False)
        l_x_adv.append(x_adv_curr.cpu())
    acc, output = utils.get_accuracy_and_logits(model, x, y, batch_size=bs,
        n_classes=n_cls)
    pred = output.to(y.device).max(1)[1] == y
    logger.log('')
    logger.log('clean accuracy: {:.1%}'.format(pred.float().mean()))
    print('clean accuracy: {:.1%}'.format(acc))
    acc_dets.append(('clean', acc + 0.))
    for norm, eps, x_adv in zip(l_norms, l_epss, l_x_adv):
        acc, output = utils.get_accuracy_and_logits(model, x_adv, y,
            batch_size=bs, n_classes=n_cls)
        other_utils.check_imgs(x_adv, x.cpu(), norm)
        pred_curr = output.to(y.device).max(1)[1] == y
        logger.log('robust accuracy {}: {:.1%}'.format(norm, pred_curr.float().mean()))
        print('robust accuracy: {:.1%}'.format(acc))
        pred *= pred_curr
        acc_dets.append((norm, acc + 0.))
    logger.log('robust accuracy {}: {:.1%}'.format('+'.join(l_norms),
        pred.float().mean()))
    acc_dets.append(('union', pred.float().mean()))
    return l_x_adv, acc_dets


def eval_norms_fast(model, x, y, l_norms, l_epss, n_iter=100, n_cls=10):
    acc_dict = {}
    assert not model.training
    bs = x.shape[0]
    acc, output = utils.get_accuracy_and_logits(model, x, y, batch_size=bs,
        n_classes=n_cls)
    pred = output.to(y.device).max(1)[1] == y
    acc_dict['clean'] = acc + 0.
    for norm, eps in zip(l_norms, l_epss):
        _, _, _, x_adv = apgd_train(model, x, y, norm=norm, eps=eps,
            n_iter=n_iter, is_train=False)
        acc_dict[norm], output = utils.get_accuracy_and_logits(model,
            x_adv, y, batch_size=bs, n_classes=n_cls)
        pred *= output.to(y.device).max(1)[1] == y
    acc_dict['union'] = pred.float().mean()
    return acc_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Wong2020Fast')
    parser.add_argument('--n_ex', type=int, default=100, help='number of examples to evaluate on')
    parser.add_argument('--batch_size_eval', type=int, default=100, help='batch size for evaluation')
    #parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--data_dir', type=str, default='/home/scratch/datasets/CIFAR10', help='where to store downloaded datasets')
    parser.add_argument('--model_dir', type=str, default='./models', help='where to store downloaded models')
    parser.add_argument('--save_dir', type=str, default='./trained_models')
    parser.add_argument('--l_norms', type=str, default='Linf L2 L1')
    parser.add_argument('--l_eps', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--only_clean', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    x_test, y_test = data.load_cifar10(1000, data_dir='/home/scratch/datasets/CIFAR10')

    model = utils.load_pretrained_models('pretr_L2') #args.model_name
    ckpt = torch.load('./trained_models/model_2021-04-21 19:57:33.710832 cifar10 lr=0.05000 piecewise-ft ep=3 attack=apgd fts=pretr_L1 seed=0 at=Linf L1 eps=default iter=10/ep_3.pth')
    model.load_state_dict(ckpt)
    model.eval()
    
    #eval_single_norm(model, x_test, y_test, norm='L2', eps=.5, bs=256)
    eval_norms(model, x_test, y_test, l_norms=['L2', 'L1'], l_epss=[.5, 12.], bs=256)
    '''

    args = parse_args()

    # load data
    if args.dataset == 'cifar10':
        x_test, y_test = data.load_cifar10(args.n_ex, data_dir=args.data_dir,
            device='cpu')
        #x_test, y_test = x_test.cpu(), y_test.cpu()
    
    if os.path.isfile(args.model_name):
        pretr_model = args.model_name.split('fts=')[1].split(' ')[0]
        args.save_dir, ckpt_name = os.path.split(args.model_name) #os.path.join(args.model_name.split('/')[:-1])
        ckpt = torch.load(args.model_name)
    else:
        pretr_model = args.model_name
        args.save_dir = '{}/{}'.format(args.save_dir, args.model_name)
        other_utils.makedir(args.save_dir)
        ckpt_name = 'pretrained'
    not_pretr = os.path.isfile(args.model_name)
    log_path = '{}/log_eval_{}.txt'.format(args.save_dir, ckpt_name)
    
    # load model
    if pretr_model == 'rand':
        model = PreActResNet18(10, activation=args.act).cuda()
        #model.eval()
    elif pretr_model.startswith('RB'):
        model = rb.utils.load_model(pretr_model.split('_')[1], model_dir=args.model_dir,
            dataset=args.dataset, threat_model=pretr_model.split('_')[2])
        model.cuda()
        #model.eval()
        print('{} ({}) loaded'.format(*pretr_model.split('_')[1:]))
    elif pretr_model.startswith('pretr'):
        model = utils.load_pretrained_models(pretr_model)
        print('pretrained model loaded')
    if not_pretr:
        model.load_state_dict(ckpt)
    model.eval()

    # clean acc
    acc = rb.utils.clean_accuracy(model, x_test, y_test,
        device='cuda')
    print('clean accuracy {:.1%}'.format(acc))
    if args.only_clean:
        sys.exit()
    
    
    # set norms and eps
    args.l_norms = args.l_norms.split(' ')
    if args.l_eps is None:
        args.l_eps = [eps_dict[args.dataset][c] for c in args.l_norms]
    else:
        args.l_eps = [float(c) for c in args.l_eps.split(' ')]

    # run attacks
    l_x_adv, _ = eval_norms(model, x_test, y_test, l_norms=args.l_norms,
        l_epss=args.l_eps, bs=args.batch_size_eval, log_path=log_path)

    # saving
    for norm, eps, v in zip(args.l_norms, args.l_eps, l_x_adv):
        torch.save(v,  '{}/eval_{}_{}_1_{}_eps_{:.5f}.pth'.format(
            args.save_dir, ckpt_name, norm, args.n_ex, eps))




