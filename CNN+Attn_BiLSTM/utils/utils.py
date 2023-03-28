import torch
from sklearn.metrics import classification_report
import numpy as np


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, iterator, criterion, model_name):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        y_hat, y_label = np.empty(0, dtype=int), np.empty(0, dtype=int)
        for batch in iterator:
            if model_name == 'Attn_BiLSTM':
                predictions = model(batch.string)[0]
            else:
                predictions = model(batch.string)

            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            y_hat = np.concatenate((y_hat, predictions.argmax(1).numpy()), axis=0)
            y_label = np.concatenate((y_label, batch.label.numpy()), axis=0)

        # print classification report, including F1-score
        print(classification_report(y_label, y_hat, target_names=['background', 'method', 'result']))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, model_name):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        if model_name == 'Attn_BiLSTM':
            predictions = model(batch.string)[0]
        else:
            predictions = model(batch.string)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.INITIAL_LR * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.INITIAL_LR}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.INITIAL_LR * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
