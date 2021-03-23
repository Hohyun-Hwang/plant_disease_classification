import os
import wandb
import torch  # for set model status(cpu, gpu), optimizer, torch.no_grad()
import argparse  # for using argparser
import numpy as np
import pandas as pd  # for make some dataframe of train,test loss. it will be printed at last
from model import CNN  # for using CNN model
import torch.nn as nn  # for define loss function
import seaborn as sns  # for styling on plot
from tqdm import tqdm  # for checking how many time on 'for' statement processing.
import matplotlib.pyplot as plt  # for draw plot
from efficientnet_pytorch import EfficientNet  # from EfficientNet
from dataloader import plant_image  # from dataset.py
from torchvision.transforms import transforms  # for augmentation on image
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import DataLoader  # for insert data from Custom dataloader and setting parameter

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

parser = argparse.ArgumentParser(description='Process to set parameters.')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--model', type=str, default="efficient")
parser.add_argument('--device', type=str, default="gpu")
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

# wandb.init(project= "plant_disease", entity='')


def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    model.to(device)
    trn_loss = 0
    acc_sum = 0
    trn_loss_sum = 0
    trn_loss_list = []
    trn_acc_list = []
    for i, (data, target) in enumerate(tqdm(trn_loader)):  
        optimizer.zero_grad()  # pytorch has gradient before nodes
        data = data.to(device)
        output = model(data)  # input data in model
        target = target.to(device)

        trn_loss = criterion(output, target)  # cost fcn is Cross_entropy
        trn_loss.backward()  # backpropagation
        optimizer.step()  # training model

        # wandb.log({"Train Loss": trn_loss})

        trn_loss_sum += trn_loss
        trn_loss_list.append(float(trn_loss))
        output_softmax = torch.log_softmax(output, dim=1)
        _, output_label = torch.max(output_softmax, dim=1)
        correct_pred = (output_label == target).float()

        # accuracy
        acc = correct_pred.sum() / len(correct_pred)
        acc_sum += acc * len(correct_pred)  # calculate accuracy
        # wandb.log({"Train Acc": acc})
        trn_acc_list.append(float(acc))

    acc_sum = int(acc_sum)
    trn_loss_sum /= len(trn_loader.dataset)  # print trn_loss mean value
    trn_acc = 100. * acc_sum / len(trn_loader.dataset)  # print trn_acc mean value

    print('\nTrain set: Average loss: {:.4f},Train Accuracy: {}/{} ({:.1f}%)\n'
          .format(trn_loss_sum, acc_sum, len(trn_loader.dataset), trn_acc))
    print('Finished Training Trainset')

    return trn_loss, trn_acc, trn_loss_list, trn_acc_list


def test(model, tst_loader, device, criterion):
    model.eval()
    model.to(device)
    tst_loss_sum = 0
    acc = 0
    acc_sum = 0
    tst_loss_list = []
    tst_acc_list = []

    with torch.no_grad():
        for data, target in tqdm(tst_loader):
            data = data.to(device)
            output = model(data)
            target = target.to(device)
            tst_loss = criterion(output, target)

            # wandb.log({"Test Loss " : tst_loss})
            tst_loss_sum += tst_loss
            tst_loss_list.append(float(tst_loss))
            output_softmax = torch.log_softmax(output, 1)
            _, output_tags = torch.max(output_softmax, 1)
            correct_pred = (output_tags == target).float()
            acc = correct_pred.sum() / len(correct_pred)
            acc_sum += acc * len(correct_pred)
            # wandb.log({"Test Acc": acc})
            tst_acc_list.append(float(acc))

    acc_sum = int(acc_sum)
    tst_loss /= len(tst_loader.dataset)
    tst_acc = 100. * acc_sum / len(tst_loader.dataset)
    print('\nTest set: Average loss: {:.4f},Test Accuracy: {}/{} ({:.1f}%)\n'
          .format(tst_loss, acc_sum, len(tst_loader.dataset), tst_acc))
    print('Finished Testing Test set')
    return tst_loss, tst_acc, tst_loss_list, tst_acc_list

def visualizing(model, test_data, device, model_type, result_dir):
    columns = 5
    rows = 5
    fig = plt.figure(figsize=(10, 10))
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(1, columns * rows + 1):
        data_idx = np.random.randint(len(test_data))
        input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)
        output = model(input_img)
        output = torch.sigmoid(output)
        probability, argmax = torch.max(output, 1)
        pred = argmax.item()
        label = test_data[data_idx][1]

        fig.add_subplot(rows, columns, i)

        label_list = ['healthy',
                      'Early_blight',
                      'Septoria_leaf_spot',
                      'Tomato_mosaic_virus',
                      'Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Spider_mites Two-spotted_spider_mite',
                      'Cercospora_leaf_spot Gray_leaf_spot',
                      'Northern_Leaf_Blight',
                      'Bacterial_spot',
                      'Powdery_mildew',
                      'Common_rust_',
                      'Cedar_apple_rust',
                      'Leaf_scorch',
                      'Late_blight',
                      'Tomato_Yellow_Leaf_Curl_Virus',
                      'Target_Spot',
                      'Esca_(Black_Measles)',
                      'Leaf_Mold',
                      'Apple_scab',
                      'Black_rot',
                      'Haunglongbing_(Citrus_greening)']
        
        pred_title = label_list[pred]

        if pred == label:
            plt.title(pred_title + '(O),\n probability : ' + str(round(probability.item(),2)))
        else:
            plt.title(pred_title + '(X),\n probability : ' + str(round(probability.item(),2)))
        plot_img = test_data[data_idx][0]
        plot_img[0, :, :] = plot_img[0, :, :] * std[0] + mean[0]
        plot_img[1, :, :] = plot_img[1, :, :] * std[1] + mean[1]
        plot_img[2, :, :] = plot_img[2, :, :] * std[2] + mean[2]
        plot_img = TF.to_pil_image(plot_img)
        plt.imshow(plot_img)
        plt.axis('off')
    fig.subplots_adjust(hspace=1, wspace=1)
    plt.savefig(result_dir + '/{}_visualization.png'.format(model_type))



def main(mode: str, total_epoch: int, graphic_device: str = 'cpu', _model: str = 'efficient', batch_size: int = 4):
    result_path = "./"+ "train"  + "_epoch_" + str(total_epoch) + "_graphic_device_" + graphic_device + "_model_" + _model + "_batch_size_" + str(batch_size)
    visualization_path = "./"+ "visualization"  + "_epoch_" + str(total_epoch) + "_graphic_device_" + graphic_device + "_model_" + _model + "_batch_size_" + str(batch_size)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    if not os.path.isdir(visualization_path):
        os.mkdir(visualization_path)
        
    epoch = total_epoch

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    batch_size = batch_size

    train_dataset_dir = "dataset_itr2/train"
    test_dataset_dir = "dataset_itr2/test"

    train_dataset = plant_image(data_dir=train_dataset_dir,
                                transform=transform)
    test_dataset = plant_image(data_dir=test_dataset_dir,
                               transform=transform)

    trn_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    tst_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

    if graphic_device == 'gpu':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    criterion = nn.CrossEntropyLoss().to(device)
    if _model == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=21)
    elif _model == 'resnet':
        model = torch.hub.load("pytorch/vision:v0.6.0",'resnet101',pretrained=True)
        model.fc = nn.Linear(2048,21)
    else:
        model = CNN()
    
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if mode == 'train':

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        print("model = {}, total epoch : {}".format(_model, epoch))

        for j in range(epoch):
            print("epoch = {} \n".format(j + 1))
            trainer = train(model, trn_loader, device, criterion, optimizer)
            train_loss_list += trainer[2]
            train_acc_list += trainer[3]
        tester = test(model, tst_loader, device, criterion)
        test_loss_list += tester[2]
        test_acc_list += tester[3]

        train_acc = pd.DataFrame(train_acc_list)
        train_loss = pd.DataFrame(train_loss_list)
        test_acc = pd.DataFrame(test_acc_list)
        test_loss = pd.DataFrame(test_loss_list)
        train_measure = pd.concat([train_acc, train_loss], axis=1)
        train_measure.columns = ['train_acc', 'train_loss']
        test_measure = pd.concat([test_acc, test_loss], axis=1)
        test_measure.columns = ['test_acc', 'test_loss']

        plt.rcParams["figure.figsize"] = (16, 9)
        sns.set(style="darkgrid")

        plt.title('Visualisation of the training(pytorch, {})'.format(_model))
        plt.ylabel('Loss/Accuracy')
        plt.xlabel('# Epochs')
        sns.lineplot(data=train_measure, linewidth=3.5)
        plt.savefig(result_path + '/{}_train_result.png'.format(_model))
        plt.close()

        plt.title('Visualisation of the testing(pytorch, {})'.format(_model))
        plt.ylabel('Loss/Accuracy')
        plt.xlabel('# Epochs')
        sns.lineplot(data=test_measure, linewidth=2.0)
        plt.savefig(result_path + '/{}_test_result.png'.format(_model))
        plt.close()

        train_measure.to_csv(result_path + "/train_measure.csv", index=False)
        test_measure.to_csv(result_path + "/test_measure.csv", index=False)
        torch.save(model, result_path + "/" +_model + '_model_epochs_'+str(total_epoch) + '_graphic_device_' +graphic_device+'_batch_size_'+ str(batch_size) +'.pth')
        print("All Task is end")
        
    else:
        model_path = result_path + "/" +_model + '_model_epochs_'+str(total_epoch) + '_graphic_device_' +graphic_device+'_batch_size_'+ str(batch_size) +'.pth'
        if os.path.isfile(model_path):
            print("Start Visualizing...")
            model = torch.load(model_path)
            visualizing(model, test_data = test_dataset, device = device, model_type = _model, result_dir = visualization_path)
            print("Visualizing is completed!")
        else:
            print("Model was not saved. Please do Training.")

if __name__ == '__main__':
    print('pytorch version : {}'.format(torch.__version__))
    print('GPU available : {}'.format(torch.cuda.is_available()))
    cuda_device = args.device
    model = args.model
    epoch = args.epoch
    batch_size = args.batch_size
    mode = args.mode
    main(mode, epoch, cuda_device, model, batch_size)
