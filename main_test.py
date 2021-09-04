import torch
import torchvision
from torchvision import datasets, transforms
from test import test_model, eval_preds
from models.QTransferLearning2 import QuantumImagenetTransferLearning as QTL
from gradcam.GradCam import GradCam, computeGradCam
import models.utilsNet as utn

import os
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
RESULTS_DIR = config.get('CHEKPOINTS', 'RESULTS_DIR')

#[MODEL]
MODEL_NAME = config.get('MODEL', 'MODEL_NAME')
N_RESBLOCKS = int(config.get('MODEL', 'N_RESBLOCKS'))
PRETRAINED = True if config.get('MODEL', 'PRETRAINED') == 'True' else False

#[QUANTUM]
QUANTUM = True if config.get('QUANTUM', 'QUANTUM') == 'True' else False
L2_NORM = True if config.get('QUANTUM', 'L2_NORM') == 'True' else False
N_QBITS = int(config.get('QUANTUM', 'N_QBITS'))
N_VQC = int(config.get('QUANTUM', 'N_VQC'))
Q_DEPTH = int(config.get('QUANTUM', 'Q_DEPTH'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE: {}'.format(device))

# if pretrained, apply imagenet normalization
test_loader = utn.createTestDataLoaders(DATA_DIR, 1, 3, PRETRAINED)

backbone = torchvision.models.resnet18(pretrained=False)

model = QTL(num_target_classes=2, backbone=backbone, num_residuals=N_RESBLOCKS,
            use_l2=L2_NORM, use_quantum=QUANTUM, q_depth=Q_DEPTH,
            n_qubits=N_QBITS, q_delta=0.1, n_qlayers=N_VQC)
model = model.to(device)

# load weights
model_weights = '%s/%s.pth' %(WEIGHTS_DIR, MODEL_NAME)
print('Loading weights from: ', model_weights)
pretrained_checkpoint = torch.load(model_weights, map_location=device)
model.load_state_dict(pretrained_checkpoint['model_state_dict'])
model.eval()

if L2_NORM:
    print('Normalization scale value: ', model.normalization.scale.cpu())

result_report = '%s/%s_eval.json' %(RESULTS_DIR, MODEL_NAME)
print('*' * 50)
print('Testing started:')
y_true, y_pred = test_model(device=device, model=model, test_loader=test_loader)
eval_preds(y_true, y_pred, result_report)

#gradcam = GradCam(model=model, cam_layer_name='resblock3', imagenet_norm=True)
#computeGradCam(gradcam, test_loader, device, RESULTS_DIR, MODEL_NAME)