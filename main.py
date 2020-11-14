import torch
import torch.nn.functional as F

import math
import os
import logging
import numpy as np
import copy
from pdb import set_trace

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from dataloaders import init_dataloaders

from MAML.model import ModelConvSynbols, ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid
from MAML.metalearners import ModelAgnosticMetaLearning, ModularMAML, ProtoMAML
from MAML.utils import ToTensor1D, set_seed, is_connected

from Utils.bgd_lib.bgd_optimizer import create_BGD_optimizer


def boilerplate(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set_seed(args, args.seed)
    if args.wandb is not None:
        if not is_connected():
            print('no internet connection. Going in dry')
            os.environ['WANDB_MODE'] = 'dryrun'
        import wandb

        if args.wandb_key is not None:
            wandb.login(key=args.wandb_key)
        if args.name is None:
            wandb.init(project=args.wandb)
        else:
            wandb.init(project=args.wandb, name=args.name)
        wandb.config.update(args)
    else:
        wandb = None

    return args, wandb


def model_for_dataset(args):
    """Instantiates and returns model based on args.dataset."""
    if args.dataset == 'omniglot':
        return ModelConvOmniglot(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
    elif args.dataset == 'tiered-imagenet':
        return ModelConvMiniImagenet(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
    elif args.dataset == 'synbols':
        return ModelConvSynbols(args.num_ways, hidden_size=args.hidden_size, deeper=args.deeper)
    elif args.dataset == "harmonics":
        return ModelMLPSinusoid(hidden_sizes=[40, 40])
    else:
        raise RuntimeError(f'Unknown dataset: {args.dataset}')


def loss_fn_for_dataset(args):
    """Instantiates and returns loss function based on args.dataset."""
    if args.dataset in ['omniglot', 'tiered-imagenet', 'synbols']:
        return F.cross_entropy
    elif args.dataset == "harmonics":
        return F.mse_loss
    else:
        raise RuntimeError(f'Unknown dataset: {args.dataset}')


def init_models(args, wandb, metalearner=None):
    """Returns meta{learner, optimizer, optimizer_cl}."""
    if metalearner is not None:
        model = metalearner.model
    else:
        model = model_for_dataset(args)
    loss_function = loss_fn_for_dataset(args)
    if args.bgd_optimizer:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
        meta_optimizer_cl = create_BGD_optimizer(
            model.to(args.device),
            mean_eta=args.mean_eta,
            std_init=args.std_init,
            mc_iters=args.train_mc_iters)
    else:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
        meta_optimizer_cl = meta_optimizer

    if metalearner is None:
        if args.method == 'MAML':
            metalearner = ModelAgnosticMetaLearning(model, meta_optimizer, loss_function, args)
        elif args.method == 'ProtoMAML':
            metalearner = ProtoMAML(model, meta_optimizer, loss_function, args)
        elif args.method == 'ModularMAML':
            metalearner = ModularMAML(model, meta_optimizer, loss_function, args, wandb=wandb)
        else:
            raise RuntimeError()
    return metalearner, meta_optimizer, meta_optimizer_cl


def pretraining(args, wandb, metalearner, meta_train_dataloader, meta_val_dataloader):
    """
    TODO: answer the following:
    - where are we sampling from P(C_pre)? I'm assuming the answer is that P(C_pre) for omniglot
      is the usual random task sampling some set of characters/labels.
    """
    if args.pretrain_model is None:
        best_metalearner = metalearner
        if args.num_epochs == 0:
            pass
        else:
            best_val = 0.
            epochs_overfitting = 0
            epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
            print(f'\npretraining for {args.num_epochs} epochs...\n')
            for epoch in range(args.num_epochs):
                metalearner.train(
                    meta_train_dataloader,
                    max_batches=args.num_batches,
                    verbose=args.verbose,
                    desc='Training',
                    leave=False)
                results = metalearner.evaluate(
                    meta_val_dataloader,
                    max_batches=args.num_batches,
                    verbose=args.verbose,
                    epoch=epoch,
                    desc=epoch_desc.format(epoch + 1))
                result_val = results['accuracies_after']
                # early stopping:
                if (best_val is None) or (result_val > best_val):
                    epochs_overfitting = 0
                    best_val = result_val
                    best_metalearner = copy.deepcopy(metalearner)
                    if args.output_folder is not None:
                        with open(args.model_path, 'wb') as f:
                            torch.save(model.state_dict(), f)
                else:
                    epochs_overfitting += 1
                    if epochs_overfitting > args.patience:
                        break
            print('\npretraining done!\n')
            if wandb is not None:
                wandb.log({'pretrain/best_acc': best_val, 'pretrain/best_epoch': epoch})# , step=epoch)
    else:
        best_metalearner = copy.deepcopy(metalearner)

    cl_model_init = copy.deepcopy(best_metalearner)
    del metalearner, best_metalearner
    return cl_model_init


def continual_learning(args, wandb, cl_model_init, meta_optimizer_cl, cl_dataloader):
    """
    TODO: answer the following
        - easy way to indicate when task switches occurred in plots? Is this what tbd tells us?
    """
    # TODO: why aren't most of these set within cl_model.__init__?
    cl_model_init.optimizer_cl = meta_optimizer_cl
    cl_model_init.cl_strategy = args.cl_strategy
    cl_model_init.cl_strategy_thres = args.cl_strategy_thres
    cl_model_init.gamma = args.gamma
    # TODO: where is this assigned/defined? Can't find in args.py or dataloaders.py.
    if args.no_cl_meta_learning:
        cl_model_init.no_meta_learning = True

    modes = ['train', 'test', 'ood']
    is_classification_task = args.is_classification_task

    accuracies = np.zeros([args.n_runs, args.timesteps])
    mses = np.zeros([args.n_runs, args.timesteps])

    tbds = np.zeros([args.n_runs, args.timesteps])
    avg_accuracies_mode = dict(zip(modes, [[], [], []]))
    avg_mses_mode = dict(zip(modes, [[], [], []]))
    log_interval = 100
    print(f'\n Continual learning for {args.n_runs} iterations...')
    for run in range(args.n_runs):
        print('=================', f'STARTING RUN {run}', '===============')
        accuracies_mode = dict(zip(modes, [[], [], []]))
        mses_mode = dict(zip(modes, [[], [], []]))

        cl_model = copy.deepcopy(cl_model_init)
        _, _, meta_optimizer_cl = init_models(args, wandb, metalearner=cl_model)
        cl_model.optimizer_cl = meta_optimizer_cl

        for i, batch in enumerate(cl_dataloader):
            # step = i + run * args.timesteps

            data, labels, task_switch, mode, _, _ = batch
            if args.algo3:
                results = cl_model.observe2(batch)
            else:
                results = cl_model.observe(batch)


            # Reporting:
            if is_classification_task:
                accuracy_after = results["accuracy_after"]
                accuracies[run, i] = accuracy_after
                accuracies_mode[mode[0]].append(accuracy_after)
            else:
                mse_after = results["mse_after"]
                mses[run, i] = mse_after
                mses_mode[mode[0]].append(mse_after)

            tbds[run, i] = float(results['tbd'])

            if wandb is not None and run == 0 and i % log_interval == 0:
                if is_classification_task:
                    accuracy_after = results["accuracy_after"]
                    wandb.log({
                        'accuracy_after': accuracy_after,
                        'timestep_i': i,
                    }) #, step=step)
                else:
                    mse_after = results["mse_after"]
                    wandb.log({'temp_cl_mse': mse_after,
                               'timestep1': i})
                    wandb.log({f'temp_cl_mse_{mode[0]}': mse_after,
                               'timestep2': i})

            if (args.verbose and i % log_interval == 0) or i == args.timesteps - 1:
                if 'g_lambda' in results:
                    wandb.log({'g_lambda': results['g_lambda']}) # step=step)
                if is_classification_task:
                    acc = np.mean(accuracies[run, :i])
                    acc_mode = []
                    for mode in modes:
                        acc_mode.append(np.mean(accuracies_mode[mode]))
                    acc_mode_str = [f'{m}_acc={a:.2f}' for m, a in zip(modes, acc_mode)]
                    print(f'total Acc: {acc:.2f},', f'mode accs: {acc_mode_str}', end='\t')
                    wandb.log({'console acc': acc})# , step=step)
                else:
                    mse = np.mean(mses[run, :i])
                    print(f'mean MSE: {mse:.5f} MSE: {mses[run, i]:.3f}', end='\t')

                # Note: tbd==task boundary detection
                tbd = np.mean(tbds[run, :i])
                print(f'Total tbd: {tbd:.2f}', f'it: {i}', sep='\t')

            if i == args.timesteps - 1:
                for mode in modes:
                    avg_accuracies_mode[mode].append(np.mean(accuracies_mode[mode]))
                if wandb is not None:
                    wandb.log({'cl_acc_by_runs': np.mean(accuracies[run, :])})
                if run == 0 and is_classification_task:
                    if acc < 1. / float(args.num_ways) + 0.2:
                        wandb.log({'fail': 1})
                        return
                break

    if wandb is not None:
        # avg accuracy per time steps:
        for i in range(args.timesteps):
            if i % log_interval == 0:
                wandb.log({
                    'run_agg/cl_acc_avg': np.mean(accuracies[:, i]),
                    'run_agg/cl_acc_std': np.std(accuracies[:, i]),
                    'run_agg/timestep_i': i
            }) #, step=i)

        # final avgs:
        final_accs = np.mean(accuracies, axis=1)
        final_begin_accs = np.mean(accuracies[:, :args.timesteps], axis=1)
        final_tbds = np.mean(tbds, axis=1)
        wandb.log({
            'final/acc': np.mean(final_accs),
            'final/acc_std': np.std(final_accs),
            'final/begin_acc': np.mean(final_begin_accs),
            'final/begin_acc_std': np.std(final_begin_accs),
            'final/tbd': np.mean(final_tbds),
            'final/tbd_std': np.std(final_tbds)})
        for mode in modes:
            wandb.log({
                f'final/{mode}': np.mean(avg_accuracies_mode[mode]),
                f'final/{mode}_std': np.std(avg_accuracies_mode[mode])})


def main(args):
    args, wandb = boilerplate(args)

    print('Initializing dataloaders...')
    # Notes:
    # - the train/val meta dls yield batches of {'train', 'test', 'ways', 'shots_tr', 'shots_te'}
    # - the cl meta dl yields list of 6 tensors (TODO?)
    meta_train_dataloader, meta_val_dataloader, cl_dataloader = init_dataloaders(args)

    print('Initializing models...')
    metalearner, meta_optimizer, meta_optimizer_cl = init_models(args, wandb)

    print('Executing pretraining...')
    cl_model_init = pretraining(
        args,
        wandb,
        metalearner,
        meta_train_dataloader,
        meta_val_dataloader)

    print('Executing continual learning...')
    continual_learning(
        args,
        wandb,
        cl_model_init,
        meta_optimizer_cl,
        cl_dataloader)


if __name__ == "__main__":
    from args import parse_args
    main(parse_args())
