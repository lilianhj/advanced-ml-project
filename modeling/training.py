'''
Classes and functions common to model training, evaluation, and selection

June 2020
'''

from collections import namedtuple
import copy

import torch
import torch.nn as nn
import torch.optim as optim

ModelResults = namedtuple('ModelResults',
                          ['model', 'accuracy', 'precision', 'recall'])


def prediction_stats(preds, y):
    '''
    Return all the values in a confusion matrix for a set of predictions and
    their labels. Only for use with binary prediction tasks.

    Inputs:
    preds (torch.tensor): predicted scores not yet converted to binary
        predictions
    y (torch.tensor): true labels for each of examples being predicted

    Returns: 4-ple of floats, (number of true positives, number of false
        positives, number of true negatives, number of false negatives)
    '''
    rounded_preds = torch.round(torch.sigmoid(preds))
    true_pos = ((rounded_preds == y) & (rounded_preds == 1)).sum()
    false_pos = ((rounded_preds != y) & (rounded_preds == 1)).sum()
    true_neg = ((rounded_preds == y) & (rounded_preds == 0)).sum()
    false_neg = ((rounded_preds != y) & (rounded_preds == 0)).sum()

    return true_pos.item(), false_pos.item(), true_neg.item(), false_neg.item()

class TrainingModule():
    '''
    Class for training and evaluating PyTorch models.
    '''

    def __init__(self, model, lr, pos_weight, use_cuda, num_epochs):
        '''
        Initialize a TrainingModule.

        Inputs:
        model (torch.nn.Module): the PyTorch model to be used
        lr (float): learning rate for training
        pos_weight (float): weight to assign to positive-labeled examples during
            training
        use_cuda (bool): whether or not to use a GPU during traing/evaluation
        num_epochs (int): number of epochs to run when training
        '''
        self.model = model
        self.use_cuda = use_cuda
        if self.use_cuda:
            model = model.cuda()
        self.epochs = num_epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self, iterator):
        '''
        Train the model for one epoch.

        Inputs:
        iterator (iterable): iterator over batches to train the model with in
        the epoch
        '''
        epoch_loss = 0
        epoch_tp = 0
        epoch_fp = 0
        epoch_tn = 0
        epoch_fn = 0
        self.model.train()

        for batch in iterator:
          # batch.alj_text has the texts and batch.decision_binary has the labels.

            self.optimizer.zero_grad()

            text = batch.alj_text
            target = batch.decision_binary
            if self.use_cuda:
                text = text.cuda()
                target = target.cuda()

            predictions = self.model.forward(text).squeeze()
            loss = self.loss_fn(predictions, target)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            batch_tp, batch_fp, batch_tn, batch_fn = prediction_stats(predictions, target)
            epoch_tp += batch_tp
            epoch_fp += batch_fp
            epoch_tn += batch_tn
            epoch_fn += batch_fn

        loss = epoch_loss / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        prec = ((epoch_tp) / (epoch_tp + epoch_fp) if (epoch_tp + epoch_fp) > 0
                else float('nan'))
        rec = ((epoch_tp) / (epoch_tp + epoch_fn) if (epoch_tp + epoch_fn) > 0
               else float('nan'))

        return loss, acc, prec, rec


    def train_model(self, train_iterator, dev_iterator):
        """
        Train the model for multiple epochs, and after each evaluate on the
        development set.  Return the best performing model on the basis of
        accuracy, precision, and recall.

        Inputs:
        train_iterator (iterable): iterator over batches to train the model with
        dev_iterator (iterable): iterator over batches to evaluate model
        performance and select models with
        """
        dev_accs = [0.]
        dev_precs = [0.]
        dev_recs = [0.]
        best_model_acc = None
        best_model_prec = None
        best_model_rec = None
        for epoch in range(self.epochs):
            self.train_epoch(train_iterator)
            dev_loss, dev_acc, dev_prec, dev_rec = self.evaluate(dev_iterator)
            print(f"Epoch {epoch}: Dev Accuracy: {dev_acc:.4f}; Dev Precision:",
                  f"{dev_prec:.4f}; Dev Recall: {dev_rec:.4f}; Dev Loss:{dev_loss:.4f}")
            if dev_acc > max(dev_accs) or best_model_acc is None:
                best_model_acc = ModelResults(copy.deepcopy(self.model), dev_acc, dev_prec, dev_rec)
            if dev_prec > max(dev_precs) or best_model_prec is None:
                best_model_prec = ModelResults(copy.deepcopy(self.model), dev_acc, dev_prec, dev_rec)
            if dev_rec > max(dev_recs) or best_model_rec is None:
                best_model_rec = ModelResults(copy.deepcopy(self.model), dev_acc, dev_prec, dev_rec)
            dev_accs.append(dev_acc)
            dev_precs.append(dev_prec)
            dev_recs.append(dev_rec)
        return {'accuracy': best_model_acc,
                'precision': best_model_prec,
                'recall': best_model_rec}

    def evaluate(self, iterator):
        '''
        Evaluate the performance of the model on the given examples.

        Inputs:
        iterator (iterable): iterator over batches to evaluate the model with
        '''
        epoch_loss = 0
        epoch_tp = 0
        epoch_fp = 0
        epoch_tn = 0
        epoch_fn = 0
        self.model.eval()

        with torch.no_grad():

            for batch in iterator:

                text = batch.alj_text
                target = batch.decision_binary
                if self.use_cuda:
                    text = text.cuda()
                    target = target.cuda()

                predictions = self.model.forward(text).squeeze()
                loss = self.loss_fn(predictions, target)

                epoch_loss += loss.item()

                batch_tp, batch_fp, batch_tn, batch_fn = prediction_stats(predictions, target)
                epoch_tp += batch_tp
                epoch_fp += batch_fp
                epoch_tn += batch_tn
                epoch_fn += batch_fn

        loss = epoch_loss / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        prec = ((epoch_tp) / (epoch_tp + epoch_fp) if (epoch_tp + epoch_fp) > 0
                else float('nan'))
        rec = ((epoch_tp) / (epoch_tp + epoch_fn) if (epoch_tp + epoch_fn) > 0
               else float('nan'))

        return loss, acc, prec, rec
