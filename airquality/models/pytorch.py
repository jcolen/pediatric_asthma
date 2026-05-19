import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import scipy.stats

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

activation_dict = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'leaky_relu': nn.LeakyReLU(),
    'none': nn.Identity(),
}

loss_dict = {
    'mse': nn.MSELoss(),
    'poisson': nn.PoissonNLLLoss(log_input=False),
}

class FCNN(L.LightningModule):
    def __init__(self, 
                 input_size,
                 output_size=1,
                 num_hidden=2,
                 hidden_size=256,
                 act_fn='tanh',
                 loss_fn='poisson',
                 lr=3e-4):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.act_fn = act_fn
        self.loss_fn = loss_fn
        self.lr = lr

        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(activation_dict[self.act_fn])

        for i in range(num_hidden - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(activation_dict[self.act_fn])

        layers.append(nn.Linear(self.hidden_size, self.output_size))

        self.model = nn.Sequential(*layers)
        self.loss = loss_dict[self.loss_fn]

        logger.info('Initialized FCNN model')

    def forward(self, inputs, offset, return_unnormalized=False):
        y0 = self.model(inputs)
        y = y0 + offset # Apply offset term
        y = y.exp() # log-link function
        if return_unnormalized: # For analysis purposes
            return y, y0
        return y
    
    def training_step(self, batch, batch_idx):
        x, y0, offset = batch
        y = self(x, offset)
        loss = self.loss(y, y0)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss
    
    def on_train_epoch_end(self):
        mean_loss = self.trainer.callback_metrics['train_loss'].mean()
        self.log('Training Loss', mean_loss, prog_bar=True, logger=True)

    def predict_step(self, batch):
        x, _, offset = batch
        y = self(x, offset)
        return y

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
def model_summary(model, datamodule):
    if torch.cuda.is_available():
        device = torch.device('gpu:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps:0')
    else:
        device = torch.device('cpu')
    model.eval().to(device)

    # Collect network saliencies
    loader = datamodule.predict_dataloader()
    K = []
    for batch in loader:
        x, _, offset = batch
        x = x.to(device)
        x.requires_grad = True
        offset = offset.to(device)

        pred, pred_unnorm = model(x, offset, return_unnormalized=True)

        grad = torch.autograd.grad(pred_unnorm.sum(), x)[0]
        K.append(grad.detach().cpu().numpy())

    # Aggregate statistics
    K = np.concatenate(K, axis=0)
    K_mean = K.mean(axis=0)
    K_ci = sms.DescrStatsW(K).tconfint_mean()
    tvalues = K_mean / scipy.stats.sem(K, axis=0)
    pvalues = 2 * scipy.stats.norm.sf(np.abs(tvalues))

    # Build dataframe
    risk_ratio_summary = pd.DataFrame({
        'parameter': datamodule.all_exog_labels,
        'risk_ratio (est.)': np.exp(K_mean),
        'lower_ci (est.)': np.exp(K_ci[0]),
        'upper_ci (est.)': np.exp(K_ci[1]),
        'p-value': pvalues
    }).set_index('parameter')

    logger.info('\n' + str(risk_ratio_summary))
    
    return risk_ratio_summary