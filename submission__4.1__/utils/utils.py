import torch
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy
import matplotlib.pyplot as plt


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
        for x, y in tqdm(iterator):
            if model_name == 'Attn_BiLSTM':
                predictions = model(x)[0]
            else:
                predictions = model(x)

            loss = criterion(predictions, y)
            acc = categorical_accuracy(predictions, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            y_hat = np.concatenate((y_hat, predictions.argmax(1).numpy()), axis=0)
            y_label = np.concatenate((y_label, y.numpy()), axis=0)

        # print classification report, including F1-score
        print(classification_report(y_label, y_hat, target_names=['background', 'method', 'result']))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_bert(model, data, data_object, device, criterion):
    # batch_size = 0
    f1s = []
    losses = []
    # accus = []
    epoch_acc = 0
    class_factor = 1.5
    accuracy_factor = 1.2
    num_of_output = 3
    c = {str(i): 0 for i in range(3)}
    p = {str(i): 0 for i in range(3)}
    for batch in tqdm(data):
        x, y = batch
        model.eval()
        y = y.type(torch.LongTensor)
        y = y.to(device)
        sentences, citation_idxs, mask, token_id_types = x
        sentences, citation_idxs, mask, token_id_types = sentences.to(device), citation_idxs.to(device), mask.to(
            device), token_id_types.to(device)
        output = model(sentences, citation_idxs, mask, token_id_types, device=device)
        # loss = F.cross_entropy(output, y, weight=torch.tensor([1.0, 5.151702786,7.234782609,43.78947368,52.82539683,55.46666667]).to(device))
        # loss = F.nll_loss(output, y, weight=torch.tensor([1.0, 500.151702786,700.234782609,4300.78947368,5200.82539683,5500.46666667]).to(device))

        _, predicted = torch.max(output, dim=1)

        loss = accuracy_factor * criterion(output, y) + class_factor * torch.log(
            (torch.subtract(y, predicted) != 0).sum())
        print("Accuracy Loss: ", accuracy_factor * criterion(output, y))
        print("Class Loss: ", class_factor * torch.log((torch.subtract(y, predicted) != 0).sum()))

        f1 = F1Score(task='multiclass', num_classes=num_of_output, average='macro').to(device)
        f1_detailed = F1Score(task='multiclass', num_classes=num_of_output, average='none').to(device)
        print("Specifically, ", f1_detailed(predicted, y))
        # self.output_types2idx = {'Background':3, 'Uses':1, 'CompareOrContrast':2, 'Extends':4, 'Motivation':0, 'Future':5}
        for x in y.cpu().detach().tolist():
            c[str(x)] += 1

        for pr in predicted.cpu().detach().tolist():
            p[str(pr)] += 1

        # accuracy = Accuracy().to(device)
        f1 = f1(predicted, y)
        # ac = accuracy(predicted, y)
        f1s.append(f1.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        # acc = categorical_accuracy(predicted, output)
        # epoch_acc += acc.item()
        # accus.append(ac.cpu().detach().numpy())

    print('y_true: ', c)
    print('y_pred: ', p)
    print('y_types: ', data_object.output_types2idx)

    f1s = np.asarray(f1s)
    f1 = f1s.mean()
    # accus = np.asarray(accus)
    losses = np.asarray(losses)
    # accus = accus.mean()
    loss = losses.mean()
    # print("Loss : %f, f1 : %f, accuracy: %f" % (loss, f1, accus))
    print("Loss : %f, f1 : %f" % (loss, f1))
    return loss, f1


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, model_name):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for x, y in tqdm(iterator):
        optimizer.zero_grad()

        if model_name == 'Attn_BiLSTM':
            predictions = model(x)[0]
        else:
            predictions = model(x)

        loss = criterion(predictions, y)

        acc = categorical_accuracy(predictions, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_bert(model, train_loader, optimizer, criterion, device, bz):
    f1s = []
    losses = []
    num_of_output = 3
    class_factor = 1.5
    accuracy_factor = 1.2
    for batch in tqdm(train_loader):
        x, y = batch
        model.train()
        assert model.training, 'make sure your network is in train mode with `.train()`'
        optimizer.zero_grad()
        model.to(device)
        y = y.type(torch.LongTensor)
        y = y.to(device)
        sentences, citation_idxs, mask, token_id_types = x
        sentences, citation_idxs, mask, token_id_types = sentences.to(device), citation_idxs.to(device), mask.to(
            device), token_id_types.to(device)
        output = model(sentences, citation_idxs, mask, token_id_types, device=device)
        _, predictted_output = torch.max(output, dim=1)
        loss = accuracy_factor * criterion(output, y) * torch.pow(torch.tensor(1.8), (
            (torch.subtract(y, predictted_output) == 0).sum()) / bz) + class_factor * torch.exp(
            ((torch.subtract(y, predictted_output) != 0).sum()) / bz) * torch.log(
            torch.square(torch.subtract(y, predictted_output)).sum())
        loss.backward()
        optimizer.step()

        f1 = F1Score(task='multiclass', num_classes=num_of_output, average='macro').to(device)
        f1 = f1(predictted_output, y)
        f1s.append(f1.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        # acc = categorical_accuracy(predictted_output, output)
        # epoch_acc += acc.item()
    f1s = np.asarray(f1s)
    f1 = f1s.mean()
    losses = np.asarray(losses)
    # accus = accus.mean()
    loss = losses.mean()
    return loss, f1


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


def train_results_plot(model_name, total_train_loss, total_valid_loss, total_train_acc, total_valid_acc, save_path):
    fig, ax = plt.subplots()
    ax.plot(range(len(total_train_loss)), total_train_loss, label='training loss')
    ax.plot(range(len(total_valid_loss)), total_valid_loss, label='validation loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'{model_name} loss fig')
    ax.legend()
    plt.savefig(f'{save_path}-loss.png')

    fig, ax = plt.subplots()
    ax.plot(range(len(total_train_acc)), total_train_acc, label='training accuracy')
    ax.plot(range(len(total_valid_acc)), total_valid_acc, label='validation accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title(f'{model_name} accuracy fig')
    ax.legend()
    plt.savefig(f'{save_path}-accuracy.png')
