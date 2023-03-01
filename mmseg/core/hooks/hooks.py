import math
from mmcv.runner.hooks import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class ReduceOnPlateauLrUpdaterHook(LrUpdaterHook):

    """

    Must be used together with EvalHook or DistEvalHook with save_best not being None.
    EvalHook saves best checkpoint's evaluation score and file path
    in ``runner.meta['hook_msgs']`` when the save_best is set.

    """

    def __init__(self, factor=0.5, min_lr=5e-7, patience=3, **kwargs):
        self.factor = factor
        self.num_updates = 0
        self.min_lr = min_lr
        self.patience = patience
        self.no_optim = 0
        self.score = None
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if runner.meta is None:
            return base_lr

        msgs = runner.meta.get("hook_msgs", None)
        if msgs is None:
            return base_lr

        best_score = msgs.get("best_score", None)
        best_ckpt = msgs.get("best_ckpt", None)

        if best_score is None or best_ckpt is None:
            pass
        elif self.score is None:
            self.score = best_score
        elif self.score == best_score:
            self.no_optim += 1
            if self.no_optim > self.patience:
                runner.load_checkpoint(best_ckpt)
                self.no_optim = 0
                self.num_updates += 1
                runner.logger.info(
                    f'Reduce lr by a factor of {self.factor} and load model from the last best checkpoint.')
        else:
            self.score = best_score
            self.no_optim = 0

        lr = base_lr * math.pow(self.factor, self.num_updates)
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)

        return lr


