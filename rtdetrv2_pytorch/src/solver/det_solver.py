"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def __init__(self, cfg):
        super().__init__(cfg)
        # Early stopping configuration
        self.early_stopping_patience = getattr(cfg, 'early_stopping_patience', None)
        self.early_stopping_min_delta = getattr(cfg, 'early_stopping_min_delta', 0.0)
        self.early_stopping_metric = getattr(cfg, 'early_stopping_metric', 'coco_eval_bbox')
        self.enable_per_class_metrics = getattr(cfg, 'enable_per_class_metrics', False)
        self.epochs_without_improvement = 0
        self.best_metric_value = -float('inf')

    @staticmethod
    def _update_best_stat(best_stat, k, value, epoch):
        """Best-checkpoint selection (strictly greater, no min_delta)."""
        if k in best_stat:
            best_stat['epoch'] = epoch if value > best_stat[k] else best_stat['epoch']
            best_stat[k] = max(best_stat[k], value)
        else:
            best_stat['epoch'] = epoch
            best_stat[k] = value

    def _update_early_stopping(self, test_stats):
        """Patience counter: improvement = strictly greater than best + min_delta."""
        current_metric = test_stats.get(self.early_stopping_metric, [0])[0]
        if current_metric > self.best_metric_value + self.early_stopping_min_delta:
            self.best_metric_value = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _restore_selection_state(self, best_stat, start_epoch):
        """Rebuild best_stat, best_metric_value and epochs_without_improvement
        from log.txt using epochs < start_epoch, applying the same logic."""
        log_path = self.output_dir / "log.txt"
        if not log_path.exists():
            print('Resume: no log.txt found, best value and patience start from scratch')
            return
        by_epoch = {}
        with log_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # line truncated by a crash
                if isinstance(rec.get('epoch'), int) and rec['epoch'] < start_epoch:
                    by_epoch[rec['epoch']] = rec  # if an epoch is repeated, the last record wins
        for epoch in sorted(by_epoch):
            test_stats = {k[len('test_'):]: v for k, v in by_epoch[epoch].items()
                          if k.startswith('test_') and isinstance(v, list) and v}
            for k in test_stats:
                self._update_best_stat(best_stat, k, test_stats[k][0], epoch)
            if self.early_stopping_patience is not None:
                self._update_early_stopping(test_stats)
        print(f'Resume: state rebuilt from {len(by_epoch)} epochs of log.txt: '
              f'best_stat={best_stat}, best_metric_value={self.best_metric_value}, '
              f'epochs_without_improvement={self.epochs_without_improvement}')

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # Early stopping info
        if self.early_stopping_patience is not None:
            print(f'Early stopping enabled:')
            print(f'  - Patience: {self.early_stopping_patience} epochs')
            print(f'  - Min delta: {self.early_stopping_min_delta}')
            print(f'  - Metric: {self.early_stopping_metric}')

        if self.enable_per_class_metrics:
            print(f'Per-class metrics logging enabled')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        # On resume, best value and patience are rebuilt from log.txt
        # (only epochs before the resume epoch)
        if start_epcoch > 0 and self.output_dir:
            self._restore_selection_state(best_stat, start_epcoch)
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                # Before lr_scheduler.step(), check the types
                for group in self.optimizer.param_groups:
                    assert isinstance(group["lr"], float), f"Learning rate must be float, got {type(group['lr'])}"
                assert isinstance(self.lr_scheduler.eta_min, float), f"eta_min must be float, got {type(self.lr_scheduler.eta_min)}"
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_checkpoint_fast(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # Log per-class metrics if enabled
            if self.enable_per_class_metrics and coco_evaluator is not None:
                if hasattr(coco_evaluator, 'per_class_metrics') and coco_evaluator.per_class_metrics:
                    self._log_per_class_metrics(coco_evaluator.per_class_metrics, epoch)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                self._update_best_stat(best_stat, k, test_stats[k][0], epoch)

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_checkpoint_fast(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            # Early stopping check
            if self.early_stopping_patience is not None:
                self._update_early_stopping(test_stats)

                if dist_utils.is_main_process():
                    print(f'Early stopping: {self.epochs_without_improvement}/{self.early_stopping_patience} epochs without improvement')

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f'\n{"="*70}')
                    print(f'Early stopping triggered at epoch {epoch}')
                    print(f'No improvement in {self.early_stopping_metric} for {self.early_stopping_patience} epochs')
                    print(f'Best {self.early_stopping_metric}: {self.best_metric_value:.4f}')
                    print(f'{"="*70}\n')
                    break

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            dist_utils.save_checkpoint_fast(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_checkpoint_fast(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def _log_per_class_metrics(self, per_class_metrics, epoch):
        """
        Log per-class metrics to TensorBoard and console.

        Args:
            per_class_metrics (dict): Dictionary with category_id as key and metrics dict as value
            epoch (int): Current epoch number
        """
        if not dist_utils.is_main_process():
            return

        # Get category names from dataset
        dataset = self.val_dataloader.dataset
        category2name = getattr(dataset, 'category2name', {})

        # Console output
        print(f"\n{'='*80}")
        print(f"Per-Class Metrics (Epoch {epoch})")
        print(f"{'='*80}")
        print(f"{'Class':<20} {'AP':>12} {'AP50':>12} {'AP75':>12} {'AR':>12}")
        print(f"{'-'*80}")

        for cat_id in sorted(per_class_metrics.keys()):
            metrics = per_class_metrics[cat_id]
            cat_name = category2name.get(cat_id, f'Class_{cat_id}')

            # Highlight minority classes (Can and Metal_Debris)
            marker = ' ***' if cat_name in ['Can', 'Metal_Debris'] else ''

            print(f"{cat_name:<20} {metrics['AP']:>12.4f} {metrics['AP50']:>12.4f} "
                  f"{metrics['AP75']:>12.4f} {metrics['AR']:>12.4f}{marker}")

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar(f'PerClass/{cat_name}/AP', metrics['AP'], epoch)
                self.writer.add_scalar(f'PerClass/{cat_name}/AP50', metrics['AP50'], epoch)
                self.writer.add_scalar(f'PerClass/{cat_name}/AP75', metrics['AP75'], epoch)
                self.writer.add_scalar(f'PerClass/{cat_name}/AR', metrics['AR'], epoch)

        print(f"{'='*80}\n")

        # Summary for minority classes
        minority_classes = ['Can', 'Metal_Debris']
        minority_cat_ids = [cat_id for cat_id, name in category2name.items() if name in minority_classes]

        if minority_cat_ids:
            print(f"{'Minority Class Summary':<20} {'AP':>12} {'AR':>12}")
            print(f"{'-'*50}")
            for cat_id in sorted(minority_cat_ids):
                cat_name = category2name.get(cat_id, f'Class_{cat_id}')
                metrics = per_class_metrics.get(cat_id, {})
                print(f"{cat_name:<20} {metrics.get('AP', 0.0):>12.4f} {metrics.get('AR', 0.0):>12.4f}")
            print()
