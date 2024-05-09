import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb
from model.loss import mse_loss_per_channel
from IPython import embed
from utils.util import MemoryBank


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None,
                 len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        # track the train and validation loss per band on wandb
        self.train_loss_per_band = None
        self.valid_loss_per_band = None
        # define the data key and target key
        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        # define a flag to indicate whether to stablize the gradient or not
        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            target = data_dict[self.target_key].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            # statblize the gradient if RTM is used
            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # log the loss to wandb
            metrics_per_step = {'train_step/train_loss': loss.item()}
            # log the loss per band to wandb
            # loss_per_band = mse_loss_per_channel(output, target)
            # metrics_per_step.update(
            #     {f'train_step_channel/train_loss_channel{i}':
            #         loss_per_band[i].item()
            #         for i in range(loss_per_band.shape[0])}
            # )
            # if batch_idx == 0:
            #     self.train_loss_per_band = loss_per_band.view(1, -1)
            # else:
            #     self.train_loss_per_band = torch.cat((
            #         self.train_loss_per_band, loss_per_band.view(1, -1)
            #     ), dim=0)
            wandb.log(metrics_per_step,
                      step=(epoch - 1) * self.len_epoch + batch_idx)

            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))
                # add line plot of the data and output to wandb

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        # log the train loss to wandb
        metrics_per_epoch = {'train_epoch/train_loss': log['loss']}
        # self.train_loss_per_band = torch.mean(
        #     self.train_loss_per_band, dim=0)
        # metrics_per_epoch.update(
        #     {f'train_epoch_channel/train_loss_channel{i}':
        #      self.train_loss_per_band[i].item()
        #      for i in range(self.train_loss_per_band.shape[0])}
        # )
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})
            # log the validation loss to wandb
            metrics_per_epoch.update({'train_epoch/val_loss': val_log['loss']})
            # self.valid_loss_per_band = torch.mean(
            #     self.valid_loss_per_band, dim=0)
            # metrics_per_epoch.update(
            #     {f'train_epoch_channel/val_loss_channel{i}':
            #      self.valid_loss_per_band[i].item()
            #      for i in range(self.valid_loss_per_band.shape[0])}
            # )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # log the learning rate to wandb
            metrics_per_epoch.update(
                {'train_epoch/lr': np.float32(self.lr_scheduler.get_lr()[0])}
            )
        wandb.log(metrics_per_epoch, step=epoch*self.len_epoch)
        if self.stablize_grad:
            wandb.summary['grad_stablize_count'] = self.stablize_count
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.valid_data_loader):
                data = data_dict[self.data_key].to(self.device)
                target = data_dict[self.target_key].to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                # # track the validation loss per band and log to wandb
                # loss_per_band = mse_loss_per_channel(output, target)
                # if batch_idx == 0:
                #     self.valid_loss_per_band = loss_per_band.view(1, -1)
                # else:
                #     self.valid_loss_per_band = torch.cat((
                #         self.valid_loss_per_band, loss_per_band.view(1, -1)
                #     ), dim=0)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _grad_stablizer(self, epoch, batch_idx, loss):
        para_grads = [v.grad.data for v in self.model.parameters(
        ) if v.grad is not None and torch.isnan(v.grad).any()]
        if len(para_grads) > 0:
            epsilon = 1e-7
            for v in para_grads:
                rand_values = torch.rand_like(v, dtype=torch.float)*epsilon
                mask = torch.isnan(v) | v.eq(0)
                v[mask] = rand_values[mask]
            self.stablize_count += 1
            self.logger.info(
                'epoch: {}, batch: {}, loss: {}, stablize count: {}'.format(
                    epoch, batch_idx, loss, self.stablize_count)
            )
