import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import kl_divergence
from torch.nn import Sigmoid
from torch.nn.functional import relu

import numpy as np
from tqdm import tqdm
from pdb import set_trace
from Utils.bgd_lib.bgd_optimizer import BGD

from collections import OrderedDict
from MAML.utils import update_parameters, tensors_to_device, compute_accuracy

__all__ = ['ModelAgnosticMetaLearning', 'MAML', 'ProtoMAML', 'FOMAML', 'ModularMAML']


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer, loss_function, args):
        self.device = args.device
        self.model = model.to(device=self.device)  
        self.optimizer = optimizer
        self.optimizer_cl = None
        self.step_size = args.step_size
        self.first_order = args.first_order
        self.num_adaptation_steps = args.num_steps
        self.scheduler = None
        self.loss_function = loss_function
        self.is_classification_task = args.is_classification_task

        self.current_model = None
        self.cl_strategy = None
        self.freeze_visual_features = args.freeze_visual_features
        self.no_meta_learning = False

        if args.per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(args.step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=args.learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(args.step_size, dtype=torch.float32,
                device=self.device, requires_grad=args.learn_step_size)

        if (self.optimizer is not None) and args.learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if args.per_param_step_size else [self.step_size]})
            if self.scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])
        self.num_ways = args.num_ways
        self.um_power = args.um_power

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })

        mean_outer_loss = torch.tensor(0., device=self.device)

        # inner loop:
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets, batch['ways'][0], batch['shots_tr'][0])

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                   test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    # used both in pretraining and CL time
    # a few gradient descent steps starting from \phi (slow weights, accumulated knowledge)
    def adapt(self, inputs, targets, ways=None, shots=None, params=None):
        results = {'inner_losses': np.zeros(
            (self.num_adaptation_steps,), dtype=np.float32)}

        params_local = None
        if params is not None:
            #TODO: need this, otherwise backprop too many steps, OOM!!
            params_local = OrderedDict({k:v.clone().detach().requires_grad_(True) for k,v in params.items()})

        for step in range(self.num_adaptation_steps):
            logits = self.model(inputs, params=params_local)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()
            if step == 0:
                if self.is_classification_task:
                    accuracy_before = compute_accuracy(logits, targets)
                    results["accuracy_before"] = accuracy_before
                else:
                    mse_before = inner_loss
                    results["mse_before"] = mse_before

            self.model.zero_grad()
            params_local = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=params_local,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features=self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        return params_local, results

    # Called during pretraining.
    def train(self, dataloader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'outer_loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                if 'inner_losses' in results:
                    postfix['inner_loss'] = '{0:.4f}'.format(
                        np.mean(results['inner_losses']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        ''' one meta-update '''
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            '''
            for batch in dataloader:
                batch = {'train', 'test'}
                batch['train'][0] = batch-size x num_shots*num_ways x input_dim
                batch['train'][1] = batch-size x num_shots*num_ways x output_dim
                batch['test'][0]  = batch-size x num_shots-test*num_ways x input_dim
                batch['test'][1]  = batch-size x num_shots-test*num_ways x output_dim
            '''
            for batch in dataloader:
            #for i, batch in enumerate(dataloader):

                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, epoch=0, **kwargs):
        mean_outer_loss, mean_inner_loss, mean_accuracy, count = 0., 0., 0, 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'inner_losses' in results:
                    mean_inner_loss += (np.mean(results['inner_losses'])
                        - mean_inner_loss) / count
                    postfix['inner_loss'] = '{0:.4f}'.format(mean_inner_loss)
                pbar.set_postfix(**postfix)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'accuracies_after': mean_accuracy,
            'mean_inner_loss': mean_inner_loss,
        }

        return results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1

    def get_outer_loss_bgd(self,inputs,targets,num_of_mc_iters):
        self.model.zero_grad()
        self.optimizer_cl.zero_grad()
        self.optimizer_cl._init_accumulators()
        outer_loss = []
        acc = 0
        mse = 0
        for mc_iter in range(num_of_mc_iters):
            self.optimizer_cl.randomize_weights()
            self.model.zero_grad()
            self.optimizer_cl.zero_grad()
            if isinstance(self, ModularMAML):
                logits = self.model(inputs, params=self.reset_masks())
            else:
                logits = self.model(inputs, params=self.current_model)
            loss = self.loss_function(logits, targets)
            outer_loss.append(loss)
            self.model.zero_grad()
            self.optimizer_cl.zero_grad()
            loss.backward(retain_graph=not self.first_order)
            self.optimizer_cl.aggregate_grads(self.batch_size)
            # self.optimizer.step()
            if self.is_classification_task:
                acc += compute_accuracy(logits, targets)
            else:
                mse += loss
        return acc, mse, outer_loss

    # update the slow weights \phi at CL time
    def outer_update(self, outer_loss):
        if isinstance(self.optimizer_cl, BGD):
            self.optimizer_cl.step()
        else:
            self.optimizer.zero_grad()
            outer_loss.backward()  
            self.optimizer.step()

    @property
    def metric_name(self):
        if self.is_classification_task:
            return 'accuracy'
        else:
            return 'mse'

    def observe(self, batch):
        if self.cl_strategy == 'never_retrain':
            self.model.eval()
        else:
            self.model.train()

        #inputs, targets, _ , _ = batch
        # inputs [1, ways*shots, 1, 28, 28], targets [1, ways*shots]
        inputs, targets, task_switch, mode, ways, shots = batch

        assert inputs.shape[0] == 1
        assert self.optimizer_cl is not None, 'Set optimizer_cl'

        # mc sampling for bgd optimizer
        self.batch_size = inputs.shape[1]
        num_of_mc_iters = 1
        if hasattr(self.optimizer_cl, "get_mc_iters"):
            num_of_mc_iters = self.optimizer_cl.get_mc_iters()
        inputs, targets = inputs[0], targets[0]

        results = {
            'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32),
            'outer_loss': 0.,
            'tbd': 0.,
            f'{self.metric_name}_before': 0.,
            f'{self.metric_name}_after': 0.}

        # There's no \theta_{t-1}
        if self.current_model is None: 
            self.current_model, _ = self.adapt(inputs, targets, ways[0], shots[0])
            self.last_mode = mode[0]
            return results

        same_nways = 1 if self.num_ways == ways[0] else 0
        if same_nways :
            ## try the prev model on the incoming data:
            with torch.set_grad_enabled(self.model.training):
                if isinstance(self.optimizer_cl, BGD):
                    ## using BGD:
                    acc, mse, outer_loss  = self.get_outer_loss_bgd(inputs, targets, num_of_mc_iters)
                    if self.is_classification_task:
                        results['accuracy_after'] = acc / num_of_mc_iters
                    else:
                        results["mse_after"]  = mse / num_of_mc_iters
                    results['outer_loss'] = torch.mean(torch.tensor(outer_loss)).item()
                else:
                    ## using SGD
                    # line 17, outer_loss is loss of previous fast weights \theta_{t-1}
                    logits = self.model(inputs, params=self.current_model)
                    outer_loss = self.loss_function(logits, targets) 
                    results['outer_loss'] = outer_loss.item()
                    if self.is_classification_task:
                        results['accuracy_after'] = compute_accuracy(logits, targets)
                    else:
                        results["mse_after"] = F.mse_loss(logits, targets)
        #else:
        #    print('---- CL ways changed from ', self.num_ways, ' to ', ways[0])

        ## prediction is done and you can now use the labels
        # line 18, update fast_weight, generated from slow weights \phi, but \phi not changed
        self.current_model, _ = self.adapt(inputs, targets, ways[0], shots[0]) 
        with torch.no_grad():
            # line 19, current_outer_loss is second term, loss of line 18
            logits = self.model(inputs, params=self.current_model)
            current_outer_loss = self.loss_function(logits, targets).item()
        current_acc = compute_accuracy(logits, targets)
        # commented this out because when n_ways change for Proto-MAML, the current
        # task is already used to initialize the last FC layer to get current_model,
        # it's kind of cheating to test acc on the same data again. The evalutation
        # should happen in the next task when it's still the same kind of task, but
        # different data.
#        if not same_nways:
#            results['outer_loss'] = current_outer_loss
#            results['accuracy_after'] = current_acc

        #----------------- CL strategies ------------------#
        # if num_ways changed, then must be a task switch, skip the rest
        tbd = 0
        if not same_nways:
            tbd = 1
        else: # same_nways
            if self.cl_tbd_thres != -1: # gamma, !=-1, so need to check if task shifted
                ## if task switched, than inner and outer loop have a missmatch!
                if self.cl_strategy=='acc':
                    if current_acc >= results['accuracy_after'] + self.cl_tbd_thres:
                        tbd = 1
                elif self.cl_strategy=='loss':
                    if current_outer_loss + self.cl_tbd_thres <= results['outer_loss']:
                        tbd = 1  # task shifted

            if self.um_power == 0.0: 
                # No Update Modulation (UM)
                ood = 1
                if self.cl_strategy in ['loss', 'acc']:

                    if self.cl_strategy=='acc':
                        if results['accuracy_after'] >= self.cl_strategy_thres: 
                            ood = 0

                    elif self.cl_strategy=='loss':
                        if results['outer_loss'] <= self.cl_strategy_thres: # lambda
                            ood = 0

                # no task shifting and it's an ood task, update the slow weights \phi
                if self.cl_strategy != 'never_retrain' and not tbd and ood:
                    self.outer_update(outer_loss) #  line 22
                #else:
                #    ol = results['outer_loss']
                #    print(f'--- no UM, task shifted? {tbd}, ood? {ood} no slow weight update. {ol}, {current_outer_loss}')
            else:
                # With Update Modulation (UM)
                ood = 1.0
                if self.cl_strategy in ['loss', 'acc']:
                    if self.cl_strategy=='acc':
                        ood = min(1.0, (results['accuracy_after']/self.cl_strategy_thres)**self.um_power)

                    elif self.cl_strategy=='loss':
                        ood = min(1.0, (results['outer_loss']/self.cl_strategy_thres)**self.um_power)
                if self.cl_strategy != 'never_retrain' and not tbd:
                    self.outer_update(outer_loss*ood) #  line 22
                #else:
                #    print('--- UM, task shifted. no slow weight update')
        #--------------------------------------------------#

        results['tbd'] = task_switch.item()==tbd

        #print('{} {} loss={:.2f} curr_loss={:.2f} acc={:.2f} curr_acc={:.2f} tbd: {}'.format(
        #                                   task_switch.item(),
        #                                   mode[0],
        #                                   results['outer_loss'],
        #                                   current_outer_loss,
        #                                   results['accuracy_after'],
        #                                   current_acc,
        #                                   results['tbd']))

        return results

    # Algo 3
    def observe2(self, batch):
        if self.cl_strategy == 'never_retrain':
            self.model.eval()
        else:
            self.model.train()

        #inputs, targets, _ , _ = batch
        # inputs [1, ways*shots, 1, 28, 28], targets [1, ways*shots]
        inputs, targets, task_switch , mode, ways, shots = batch

        # for now we are doing one task at a time
        assert inputs.shape[0] == 1
        assert self.optimizer_cl != None, 'Set optimizer_cl'
        # mc sampling for bgd optimizer
        self.batch_size = inputs.shape[1]
        num_of_mc_iters = 1
        #set_trace()
        if hasattr(self.optimizer_cl, "get_mc_iters"):
            num_of_mc_iters = self.optimizer_cl.get_mc_iters()
        inputs, targets  = inputs[0], targets[0]

        results = {
            'inner_losses': np.zeros((self.num_adaptation_steps,), dtype=np.float32),
            'outer_loss': 0.,
            'tbd':0.,
        }
        if self.is_classification_task:
            results.update({
                'accuracy_before': 0.,
                'accuracy_after': 0.
            })
        else:
            results.update({
                "mse_before": 0.,
                "mse_after": 0.,
            })

        # There's no \theta_{t-1}
        if self.current_model is None: 
            self.current_model, _ = self.adapt(inputs, targets, ways[0], shots[0])
            self.last_mode = mode[0]
            return results

        same_nways = 1 if self.num_ways == ways[0] else 0
        if same_nways :
            ## try the prev model on the incoming data:
            with torch.set_grad_enabled(self.model.training):
                if isinstance(self.optimizer_cl, BGD):
                    ## using BGD:
                    acc, mse, outer_loss  = self.get_outer_loss_bgd(inputs, targets, num_of_mc_iters)
                    if self.is_classification_task:
                        results['accuracy_after'] = acc / num_of_mc_iters
                    else:
                        results["mse_after"]  = mse / num_of_mc_iters
                    results['outer_loss'] = torch.mean(torch.tensor(outer_loss)).item()
                else:
                    ## using SGD
                    # line 17, outer_loss is loss of previous fast weights \theta_{t-1}
                    logits = self.model(inputs, params=self.current_model)
                    outer_loss = self.loss_function(logits, targets) 
                    results['outer_loss'] = outer_loss.item()
                    if self.is_classification_task:
                        results['accuracy_after'] = compute_accuracy(logits, targets)
                    else:
                        results["mse_after"] = F.mse_loss(logits, targets)

        # virtual fast_weight, generated from slow weights \phi, but \phi not changed
        self.current_virtual_model, _ = self.adapt(inputs, targets, ways[0], shots[0]) 
        #with torch.no_grad():
        # line 19, current_outer_loss is second term, loss of line 18
        logits = self.model(inputs, params=self.current_virtual_model)
        current_outer_loss = self.loss_function(logits, targets)
        current_outer_loss_value = current_outer_loss.item()
        current_acc = compute_accuracy(logits, targets)
#        if not same_nways: #TODO: remove this, this is cheating
#            results['outer_loss'] = current_outer_loss_value
#            results['accuracy_after'] = current_acc

        #----------------- CL strategies ------------------#

        tbd = 0 if same_nways else 1
        if tbd == 0 and self.cl_tbd_thres != -1: # must be same nways, still need to check task shift
            if self.cl_strategy=='acc':
                if current_acc >= results['accuracy_after'] + self.cl_tbd_thres:
                    tbd = 1
            elif self.cl_strategy=='loss':
                if current_outer_loss_value + self.cl_tbd_thres <= results['outer_loss']:
                    tbd = 1  # task shifted
        if tbd == 0: # no task shift
            # update fast weight \theta_t from previous fast weight
            self.current_model, _ = self.adapt(inputs, targets, ways[0], shots[0], params=self.current_model)
        else:  # task shifted, tbd == 1
#            print('--- task shifted', current_outer_loss_value+self.cl_tbd_thres,' < ', results['outer_loss'])
            if self.um_power == 0.0: 
                # Without Update Modulation (UM)
                ood = 1
                if self.cl_strategy in ['loss', 'acc']:
                    if self.cl_strategy=='acc':
                        if same_nways:
                            if results['accuracy_after'] >= self.cl_strategy_thres:
                                ood = 0
                        else:
                            if current_acc >= self.cl_strategy_thres:
                                ood = 0
                    elif self.cl_strategy=='loss':
                        if same_nways:
                            if results['outer_loss'] <= self.cl_strategy_thres:
                                ood = 0
                        else:
                            if current_outer_loss_value >= self.cl_strategy_thres:
                                ood = 0

                # update the slow weights \phi
                if self.cl_strategy != 'never_retrain' and ood:
                    #print('!!! ood')
                    if same_nways:
                        self.outer_update(outer_loss) 
                    else:
                        self.outer_update(current_outer_loss)  
            else:
                # With Update Modulation (UM)
                ood = 1.0
                if self.cl_strategy in ['loss', 'acc']:
                    if self.cl_strategy=='acc':
                        if same_nways:
                            ood = min(1.0, (results['accuracy_after']/self.cl_strategy_thres)**self.um_power)
                        else:
                            ood = min(1.0, (current_acc/self.cl_strategy_thres)**self.um_power)
                    elif self.cl_strategy=='loss':
                        if same_nways:
                            ood = min(1.0, (results['outer_loss']/self.cl_strategy_thres)**self.um_power)
                        else:
                            ood = min(1.0, (current_outer_loss_value/self.cl_strategy_thres)**self.um_power)

                if self.cl_strategy != 'never_retrain':
                    if same_nways:
                        self.outer_update(outer_loss*ood) 
                    else:
                        self.outer_update(current_outer_loss*ood)
             
            # update fast weight \theta_t from slow weight
            self.current_model, _ = self.adapt(inputs, targets, ways[0], shots[0])


        #--------------------------------------------------#

        results['tbd'] = task_switch.item()==tbd

        #print('{} {} loss={:.2f} curr_loss={:.2f} acc={:.2f} curr_acc={:.2f} tbd: {}'.format(
        #                                   task_switch.item(),
        #                                   mode[0],
        #                                   results['outer_loss'],
        #                                   current_outer_loss,
        #                                   results['accuracy_after'],
        #                                   current_acc,
        #                                   results['tbd']))

        return results


class ProtoMAML(ModelAgnosticMetaLearning):

    def __init__(self, model, optimizer, loss_function, args):
        super(ProtoMAML, self).__init__(model, optimizer, loss_function, args)
        self.args = args
        self.num_ways = None
        self.um_power = args.um_power

    def adapt(self, inputs, targets, ways=None, shots=None, params=None):
        """
        not shuffled, same class group together
        inputs: [ways*shots, 1, 28, 28]
        targets: [ways*shots]
        """
        
        if ways is None:
            raise ValueError('Proto-MAML adapt arg ways & shots cannot be None!')

        params_local = None

        if self.num_ways is None or ways != self.num_ways:
            self.model.update_classifier(ways.to('cpu').tolist())
        self.num_ways = ways

        prototypes = self.model.forward_conv(inputs) # [ways*shots, 64]
        prototypes = torch.reshape(prototypes, (ways, shots, prototypes.shape[-1])) #[ways, shots, 64]
        prototypes = torch.mean(prototypes, axis=1) # [ways, 64] 

        # Proto-MAML: init last FC layer with prototype weights
        self.model.classifier.weight=torch.nn.Parameter(2.0*prototypes) # w_k = 2c_k
        self.model.classifier.bias=torch.nn.Parameter(
                -torch.sum(prototypes*prototypes, axis=1)) # b_k = -|c_k|^2
        #else: # param not None, must be the same task, same nway
        if params is not None:
            #TODO: need this, otherwise backprop too many steps, OOM!!
            params_local = OrderedDict({k:v.clone().detach().requires_grad_(True) for k,v in params.items()})

        results = {'inner_losses': np.zeros(
            (self.num_adaptation_steps,), dtype=np.float32)}

        for step in range(self.num_adaptation_steps):
            logits = self.model(inputs, params=params_local)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if step == 0:
                if self.is_classification_task:
                    accuracy_before = compute_accuracy(logits, targets)
                    results["accuracy_before"] = accuracy_before
                else:
                    mse_before = inner_loss
                    results["mse_before"] = mse_before

            self.model.zero_grad()

            params_local = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=params_local,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features=self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        return params_local, results


class FOMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        super(FOMAML, self).__init__(model, optimizer=optimizer, first_order=True,
            step_size=step_size, learn_step_size=learn_step_size,
            per_param_step_size=per_param_step_size,
            num_adaptation_steps=num_adaptation_steps, scheduler=scheduler,
            loss_function=loss_function, device=device)


class ModularMAML(ModelAgnosticMetaLearning):
    def __init__(self, model, optimizer, loss_function, args, wandb=None):
        super(ModularMAML, self).__init__(model, optimizer, loss_function, args)

        assert (args.kl_reg<=0) or args.mask_activation=='sigmoid'

        self.mask_activation = args.mask_activation
        self.modularity = args.modularity
        self.l1_reg = args.l1_reg
        self.kl_reg = args.kl_reg
        self.bern_prior = args.bern_prior
        self.masks_init = args.masks_init
        self.hard_masks = args.hard_masks
        self.wandb = wandb
        self.current_mask_stats = None

        self.weight_pruning = OrderedDict(self.model.meta_named_parameters())
        self.weight_total = OrderedDict(self.model.meta_named_parameters())
        self.reset_weight_pruning()

        # count total number of params
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.tot_params = sum([np.prod(p.size()) for p in model_parameters])

    def reset_weight_pruning(self):
        if self.modularity == 'param_wise':
            for (name, _) in self.weight_pruning.items():
                if 'classifier' in name:
                    continue
                self.weight_pruning[name] = torch.autograd.Variable(
                    torch.zeros_like(self.weight_pruning[name]), requires_grad=False).type(torch.int)
                self.weight_total[name] = torch.autograd.Variable(
                     torch.zeros_like(self.weight_total[name]), requires_grad=False).type(torch.int)

    def apply_non_linearity(self, masks_logits):
        if self.mask_activation in [None, 'None']:
            if self.hard_masks:
                return torch.clamp(masks_logits, 1e-8, 1-1e-8)
            else:
                return masks_logits
        elif self.mask_activation == 'sigmoid':
            return Sigmoid()(masks_logits)
        elif self.mask_activation == 'ReLU':
            if self.hard_masks:
                return torch.clamp(masks_logits, 1e-8, 1-1e-8)
            else:
                return relu(masks_logits)
        elif self.mask_activation == 'hardsrink':
            raise Exception('doesnt work yet')
            return torch.nn.Hardshrink()(masks_logits)

    def init_params(self):

        params = OrderedDict(self.model.meta_named_parameters())
        params_masked = OrderedDict(self.model.meta_named_parameters())
        masks_logits = OrderedDict(self.model.meta_named_parameters())
        masks = OrderedDict(self.model.meta_named_parameters())

        #TODO(learn the initial value)
        if self.modularity=='param_wise':
            for (name, _) in masks_logits.items():
                if 'classifier' in name:
                    continue
                else:
                    masks_logits[name] = torch.autograd.Variable(torch.ones_like(masks_logits[name])*
                            self.masks_init, requires_grad=True)
                    masks[name] = torch.autograd.Variable(torch.zeros_like(masks[name]),
                            requires_grad=True)

        return params, params_masked, masks_logits, masks

    def apply_masks(self, params, params_masked, masks_logits, masks, regularize=False, evaluate=False):

        l1_reg, kl_reg = 0, 0

        for (name, _) in masks_logits.items():

            if 'classifier' in name:
                # we are not pruning the classifier:
                params_masked[name] = masks_logits[name]

            else:
                masks[name] = self.apply_non_linearity(masks_logits[name])

                # we could to hard mask this way, but less interpretable
                #applied_masks = masks[name] * (masks[name].detach()>self.masks_thres).float()
                if self.hard_masks:
                    applied_masks = Bernoulli(probs=masks[name]).sample()
                    applied_masks = (masks[name] + applied_masks).detach() - masks[name]
                else:
                    applied_masks = masks[name]

                if self.modularity=='param_wise':
                    params_masked[name] = params[name] * applied_masks

                if regularize:
                    if self.l1_reg>0:
                        l1_reg += self.l1_reg * torch.sum(torch.abs(masks[name]))

                    if self.kl_reg>0:
                        # this will only work if masks = sigmoid(masks_logits)
                        bern_masks = Bernoulli(probs=masks[name])
                        bern_prior = Bernoulli(probs=torch.ones_like(masks[name])*self.bern_prior)
                        kl_reg += self.kl_reg * \
                                torch.distributions.kl_divergence(bern_masks, bern_prior).sum()

                # count the number of pruned neurons
                if evaluate:
                    self.weight_pruning[name] += (applied_masks==0).type(torch.int)
                    self.weight_total[name] += torch.ones_like(applied_masks).type(torch.int)

        if regularize:
            reg = l1_reg + kl_reg
            return params_masked, masks_logits, reg
        else:
            return params_masked, masks_logits

    def adapt(self, inputs, targets, ways=None, shots=None):

        results = {'inner_losses': np.zeros(
            (self.num_adaptation_steps,), dtype=np.float32)}

        params, params_masked, masks_logits, masks = self.init_params()

        for step in range(self.num_adaptation_steps):

            params_masked, masks_logits, reg = self.apply_masks(params, params_masked, masks_logits,
                    masks, regularize=True)

            logits = self.model(inputs, params=params_masked)
            inner_loss = self.loss_function(logits, targets) + reg

            results['inner_losses'][step] = inner_loss.item()

            if (step == 0) and self.is_classification_task:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()

            masks_logits = update_parameters(self.model, inner_loss,
                step_size=self.step_size, params=masks_logits,
                first_order=(not self.model.training) or self.first_order,
                freeze_visual_features = self.freeze_visual_features,
                no_meta_learning=self.no_meta_learning)

        self.current_mask_stats = masks_logits
        # final masking
        params_masked, _ = self.apply_masks(params, params_masked, masks_logits, masks,
                    regularize=False, evaluate=(not self.model.training))

        return params_masked, results

    def sparsity_monitoring(self, epoch):
        tot_sparsity, tot_dead = [], []
        params = OrderedDict(self.model.meta_named_parameters())
        for (name, _) in self.weight_pruning.items():
            if 'classifier' in name:
                continue
            sparsity = self.weight_pruning[name].float() / self.weight_total[name].float()
            spartity = sparsity.cpu().numpy()
            sparsity_mean = sparsity.mean()
            sparsity_std = sparsity.std()
            sparsity = sparsity.flatten().tolist()
            multiplier=1
            tot_sparsity += sparsity * multiplier
            dead = self.weight_pruning[name] == self.weight_total[name]
            dead = dead.type(torch.float).cpu().numpy()
            dead_mean = dead.mean()
            dead_std = dead.std()
            dead = dead.flatten().tolist()
            tot_dead += dead * multiplier
            print(name + ' : sparse={0:.3f} +\- {1:.3f} \t dead={2:.3f} +/- {3:.3f}'.format(
                sparsity_mean, sparsity_std, dead_mean, dead_std))
            if self.wandb is not None:
                self.wandb.log({name+'_sparse_mean':sparsity_mean}, step=epoch)
                self.wandb.log({name+'_sparse_std':sparsity_std}, step=epoch)
                self.wandb.log({name+'_dead_mean':dead_mean}, step=epoch)
                self.wandb.log({name+'_dead_std':dead_std}, step=epoch)

        tot_sparsity_mean = np.array(tot_sparsity).mean()
        tot_sparsity_std = np.array(tot_sparsity).std()
        tot_dead_mean = np.array(tot_dead).mean()
        tot_dead_std = np.array(tot_dead).std()
        print('Total : sparse={0:.3f} +\- {1:.3f} \t dead={2:.3f} +/- {3:.3f}'.format(
                tot_sparsity_mean, tot_sparsity_std, tot_dead_mean, tot_dead_std))
        if self.wandb is not None:
            self.wandb.log({'tot_sparsity_mean':tot_sparsity_mean}, step=epoch)
            self.wandb.log({'tot_sparsity_std':tot_sparsity_std}, step=epoch)
            self.wandb.log({'tot_dead_mean':tot_dead_mean}, step=epoch)
            self.wandb.log({'tot_dead_std':tot_dead_std}, step=epoch)

        self.reset_weight_pruning()

    def evaluate(self, dataloader, max_batches=500, verbose=True, epoch=0, **kwargs):
        mean_outer_loss, mean_inner_loss, mean_accuracy, count = 0., 0., 0, 0
        self.reset_weight_pruning()

        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                        - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                if 'inner_losses' in results:
                    mean_inner_loss += (np.mean(results['inner_losses'])
                        - mean_inner_loss) / count
                    postfix['inner_loss'] = '{0:.4f}'.format(mean_inner_loss)
                pbar.set_postfix(**postfix)

        self.sparsity_monitoring(epoch)

        results = {
            'mean_outer_loss': mean_outer_loss,
            'accuracies_after': mean_accuracy,
            'mean_inner_loss': mean_inner_loss
        }

        return results

    def reset_masks(self):
        params = OrderedDict(self.model.meta_named_parameters())
        params_masked = OrderedDict(self.model.meta_named_parameters())
        masks = OrderedDict(self.model.meta_named_parameters())
        masks_logits = OrderedDict(self.model.meta_named_parameters())

        params_masked, _ = self.apply_masks(params, params_masked, masks_logits, masks,
                                           regularize=False, evaluate=(not self.model.training))
        return params_masked


MAML = ModelAgnosticMetaLearning
