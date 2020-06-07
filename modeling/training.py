from collections import namedtuple
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data

ModelResults = namedtuple('ModelResults',
                          ['model', 'accuracy', 'precision', 'recall'])


def prediction_stats(preds, y):
    '''
    Return true/false positives/negatives per patch
    '''
    true_pos = ((rounded_preds == y) & (rounded_preds == 1)).float()
    false_pos ((rounded_preds != y) & (rounded_preds == 1)).float()
    true_neg = ((rounded_preds == y) & (rounded_preds == 0)).float()
    false_neg = ((rounded_preds != y) & (rounded_preds == 0)).float()
    
    return true_pos.item(), false_pos.item(), true_neg.item(), false_neg.item()

class TrainingModule():

    def __init__(self, model, lr, pos_weight, use_cuda, num_epochs):
        self.model = model
        self.use_cuda = use_cuda
        if self.use_cuda:
            model = model.cuda()
        self.epochs = num_epochs
       
        ##YOUR CODE HERE##
        # Choose an optimizer. optim.Adam is a popular choice
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def train_epoch(self, iterator):
        '''
        Train the model for one epoch. For this repeat the following, 
        going through all training examples.
        1. Get the next batch of inputs from the iterator.
        2. Determine the predictions using a forward pass.
        3. Compute the loss.
        4. Compute gradients using a backward pass.
        5. Execute one step of the optimizer to update the model paramters.
        '''
        epoch_loss = 0
        epoch_acc = 0
        epoch_prec = 0
        epoch_rec = 0
        self.model.train()
    
        for batch in iterator:
          # batch.text has the texts and batch.label has the labels.
        
            self.optimizer.zero_grad()
                
            ##YOUR CODE HERE##
            text = batch.alj_text
            target = batch.decision_binary
            if self.use_cuda:
                text = text.cuda()
                target = target.cuda()

            predictions = self.model.forward(text).squeeze()
            loss = self.loss_fn(predictions, target)
            accuracy = binary_accuracy(predictions, target)
            precision = binary_precision(predictions, target)
            recall = binary_recall(predictions, target)
        
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            epoch_prec += precision.item()
            epoch_rec += recall.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_prec / len(iterator), epoch_rec / len(iterator)

    
    def train_model(self, train_iterator, dev_iterator):
        """
        Train the model for multiple epochs, and after each evaluate on the
        development set.  Return the best performing model.
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
                accuracy = binary_accuracy(predictions, target)
                precision = binary_precision(predictions, target)
                recall = binary_recall(predictions, target)
        
                epoch_loss += loss.item()
            
                batch_tp, batch_fp, batch_tn, batch_fn = prediction_stats(predictions, target)
                epoch_tp += batch_tp
                epoch_fp += batch_fp
                epoch_tn += batch_tn
                epoch_fn += batch_fn
        
        loss = epoch_loss / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        prec = (epoch_tp) / (epoch_tp + epoch_fp)
        rec = (epoch_tp) / (epoch_tp + epoch_fn)
        
        return loss, acc, prec, rec
