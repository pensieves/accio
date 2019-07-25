import torch
from math import log10
from datetime import datetime
import copy

from eval_utils import multiclass_accuracy, Average_Tracker

from tensorboardX import SummaryWriter

def train_eval(dataloader, model, criterion, optimizer, epoch, end_epoch, 
               mode='train', report_freq=10, device=torch.device('cpu')):
    "Train or Evaluate on one epoch"
    
    if mode == 'train':
        model.train() # switch to train mode
    else:
        model.eval() # switch to eval mode
    
    loss_tracker = Average_Tracker()
    acc_tracker = Average_Tracker()
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward pass
        # track history if only in train mode
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)            

            # backward + optimize only if in training mode
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # record loss and accuracy
        acc = multiclass_accuracy(outputs, targets)
        acc_tracker.update(acc, targets.size(0))
        loss_tracker.update(loss.item(), targets.size(0))
        
    # report current epoch performance
    loss, acc = loss_tracker.avg, acc_tracker.avg
    if (epoch+1) % report_freq == 0:
        # 23 is left justified to limit time output till millisecond resolution
        time = str(datetime.now())[:23].ljust(23)
        epoch_padded_len = int(log10(end_epoch))+1
        mode_phrase = 'Train' if mode == 'train' else 'Val  '
        lr = ['{0:>8.6f}'.format(param_group['lr']) for param_group in optimizer.param_groups]
        lr = ','.join(lr)
        
        print('{0} [{1}] | Epoch = {2:{3}d} / {4} | lr = {5} | Loss = {6:>10.6f} | '
                  'Accuracy = {7:>6.2f}%'.format(mode_phrase, time, epoch+1, 
                    epoch_padded_len, end_epoch, lr, loss, acc))
    
    return loss, acc

def train_model(dataloader_dict, model, criterion, optimizer, start_epoch, 
                end_epoch, model_save_path, train_eval_func=train_eval,
                lr_scheduler=None, report_freq_dict=None, 
                model_improve_check_freq=1, save_freq=1, min_loss=float('inf'),
                device=torch.device('cpu'), scheduler_step_mode='train', 
                improve_check_mode='train', tb_viz_dir='tensorboard_viz/exp1/'):
    
    "Train a model for several epochs"
    
    # some basic checks and balances
    
    if lr_scheduler is None:
        # get a lr_scheduler for namesake which will not have any effect on optimizer
        # e.g. here setting the gamma multiplier as 1 for an arbitrary step size.
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
    
    if report_freq_dict is None:
        # set report frequency for the last epoch
        report_freq_dict = {'train' : end_epoch, 'val' : end_epoch}
    
    if model_improve_check_freq > save_freq:
        model_improve_check_freq = save_freq
        print('model_improve_check_freq > save_freq, '
              'hence will be set equal to save_freq.')
    
    # Initialize tensorboardX writer
    tb_viz_writer = SummaryWriter(log_dir=tb_viz_dir)
    
    # Start training for individual epochs
    for epoch in range(start_epoch, end_epoch):
        
        # lr scheduling independent of any metric supervision:
        if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step()
            
        # train for one epoch
        tr_loss, tr_acc = train_eval_func(dataloader_dict['train'], model, 
                                    criterion, optimizer, epoch, 
                                    end_epoch, mode='train', 
                                    report_freq=report_freq_dict['train'], 
                                    device=device)
        
        if dataloader_dict.get('val') is not None:                
            # evaluate on validation set
            val_loss, val_acc = train_eval_func(dataloader_dict['val'], model, 
                                    criterion, optimizer, epoch, 
                                    end_epoch, mode='eval',
                                    report_freq=report_freq_dict['val'], 
                                    device=device)
            
            # log the validation loss and accuracy in tensorboardX writer
            tb_viz_writer.add_scalars('loss_group', 
                                      {'Train_loss': tr_loss,
                                       'Val_loss': val_loss}, epoch)
            
            tb_viz_writer.add_scalars('accuracy_group', 
                                      {'Train_accuracy': tr_acc,
                                       'Val_accuracy': val_acc}, epoch)

        else:
            # log the training loss and accuracy in tensorboardX writer
            tb_viz_writer.add_scalar('Train_loss', tr_loss, epoch)
            tb_viz_writer.add_scalar('Train_accuracy', tr_acc, epoch)

        # lr scheduling dependent on metric supervision:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if scheduler_step_mode == 'train':
                lr_scheduler.step(tr_loss)
            else:
                lr_scheduler.step(val_loss)

        # keep track of best model
        model_improved = False
        if (epoch+1) % model_improve_check_freq == 0:
            
            if improve_check_mode == 'train' and tr_loss < min_loss:
                model_improved, min_loss, loss_mode = True, tr_loss, 'tr_loss'
                
            elif improve_check_mode == 'val' and val_loss < min_loss:
                model_improved, min_loss, loss_mode = True, val_loss, 'val_loss'

            if model_improved:
                dict_to_save = {'epoch' : epoch+1,
                                loss_mode : min_loss,
                                'state_dict': copy.deepcopy(model.state_dict()),
                                'optimizer' : copy.deepcopy(optimizer.state_dict())
                               }
                
        
        # if save_freq matches, save the current model
        if (epoch+1) % save_freq == 0:
            print('Saving current best model at epoch no. {} ...'.format(epoch+1))
            torch.save(dict_to_save, model_save_path)
            print('Current best model saved.')