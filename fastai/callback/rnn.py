# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/34_callback.rnn.ipynb (unless otherwise specified).

__all__ = ['ModelResetter', 'RNNCallback', 'RNNRegularizer', 'rnn_cbs']

# Cell
from ..basics import *

# Cell
@docs
class ModelResetter(Callback):
    "`Callback` that resets the model at each validation/training step"
    def before_train(self):    self.model.reset()
    def before_validate(self): self.model.reset()
    def after_fit(self):       self.model.reset()
    _docs = dict(before_train="Reset the model before training",
                 before_validate="Reset the model before validation",
                 after_fit="Reset the model after fitting")

# Cell
class RNNCallback(Callback):
    "Save the raw and dropped-out outputs and only keep the true output for loss computation"
    def after_pred(self): self.learn.pred,self.raw_out,self.out = [o[-1] if is_listy(o) else o for o in self.pred]

# Cell
class RNNRegularizer(Callback):
    "Add AR and TAR regularization"
    order,run_valid = RNNCallback.order+1,False
    def __init__(self, alpha=0., beta=0.): store_attr()
    def after_loss(self):
        if not self.training: return
        if self.alpha: self.learn.loss_grad += self.alpha * self.rnn.out.float().pow(2).mean()
        if self.beta:
            h = self.rnn.raw_out
            if len(h)>1: self.learn.loss_grad += self.beta * (h[:,1:] - h[:,:-1]).float().pow(2).mean()

# Cell
def rnn_cbs(alpha=0., beta=0.):
    "All callbacks needed for (optionally regularized) RNN training"
    reg = [RNNRegularizer(alpha=alpha, beta=beta)] if alpha or beta else []
    return [ModelResetter(), RNNCallback()] + reg