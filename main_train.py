import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchsummary import summary
from models.QTransferLearning2 import QuantumImagenetTransferLearning as QTL
from train import train_model
import models.utilsNet as utn

import sys
import configparser


# read parameters
args = str(sys.argv)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# read configuration parameters
config = configparser.ConfigParser()
config.read(sys.argv[1])

#[CHEKPOINTS]
DATA_DIR = config.get('CHEKPOINTS', 'DATA_DIR')
WEIGHTS_DIR = config.get('CHEKPOINTS', 'WEIGHTS_DIR')

#[MODEL]
MODEL_NAME = config.get('MODEL', 'MODEL_NAME')
PRETRAINED = True if config.get('MODEL', 'PRETRAINED') == 'True' else False
FINETUNE = True if config.get('MODEL', 'FINETUNE') == 'True' else False
N_RESBLOCKS = int(config.get('MODEL', 'N_RESBLOCKS'))
SEED = int(config.get('MODEL', 'SEED'))

#[OPTIMIZER]
LR = float(config.get('OPTIMIZER', 'LR'))
MOMENTUM = float(config.get('OPTIMIZER', 'MOMENTUM'))
BS = int(config.get('OPTIMIZER', 'BS'))
EPOCHS = int(config.get('OPTIMIZER', 'EPOCHS'))
LR_DECAY = float(config.get('OPTIMIZER', 'LR_DECAY'))
LR_PATIENCE = int(config.get('OPTIMIZER', 'LR_PATIENCE'))

#[QUANTUM]
QUANTUM = True if config.get('QUANTUM', 'QUANTUM') == 'True' else False
L2_NORM = True if config.get('QUANTUM', 'L2_NORM') == 'True' else False
N_QBITS = int(config.get('QUANTUM', 'N_QBITS'))
N_VQC = int(config.get('QUANTUM', 'N_VQC'))
Q_DEPTH = int(config.get('QUANTUM', 'Q_DEPTH'))

# set manual seed
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device))

# create dataloaders
# if pretrained, apply imagenet normalization
dataloaders, dataset_sizes = utn.createTrainValDataLoaders(DATA_DIR, BS, 3, PRETRAINED)

backbone = torchvision.models.resnet18(pretrained=PRETRAINED)

if PRETRAINED and not FINETUNE:
    for param in backbone.parameters():
        param.requires_grad = False

model = QTL(num_target_classes=2, backbone=backbone, num_residuals=N_RESBLOCKS,
            use_l2=L2_NORM, use_quantum=QUANTUM, q_depth=Q_DEPTH,
            n_qubits=N_QBITS, q_delta=0.1, n_qlayers=N_VQC)
model = model.to(device)
summary(model, (3, 32, 32))

criterion= nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=LR_DECAY,
                                           patience=LR_PATIENCE, verbose=True)

model_checkpoint = '%s/%s.pth' % (WEIGHTS_DIR, MODEL_NAME)
model_history = '%s/%s_history.json' % (WEIGHTS_DIR, MODEL_NAME)
model_time = '%s/%s_time.json' % (WEIGHTS_DIR, MODEL_NAME)
# train the model
print('*' * 50)
model, m_history, train_time = train_model(device=device, model=model, criterion=criterion,
                                           optimizer=optimizer, scheduler=scheduler,
                                           num_epochs=EPOCHS, batch_size=BS,
                                           dataloaders=dataloaders,
                                           dataset_sizes=dataset_sizes,
                                           PATH_MODEL=model_checkpoint,
                                           PATH_HISTORY=model_history,
                                           PATH_TIME=model_time,
                                           show_plot=False)

if L2_NORM:
    print('Normalization scale value: ', model.normalization.scale.cpu())
