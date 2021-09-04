# PyTorch
import torch
import torch.nn as nn

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Set devices
wires = 4 * 1  # n_qubits*n_qlayers
dev = qml.device("default.qubit", wires=wires)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat, q_depth=1, n_qubits=4):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)



class NormLayer(nn.Module):
    """ Feature normalization layer """
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))


    def forward(self, x):
        # l2- normalization of the input features, and scaling 
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_normalized = self.scale * x.div(norm.expand_as(x))

        return x_normalized 
        

class QuantumLayer(nn.Module):
    """ Quantum mapping layer """
    def __init__(self, q_depth=1, n_qubits=4, q_delta=0.01):
        super().__init__()
        self.q_depth = q_depth
        self.n_qubits = n_qubits
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        #self.reduction = nn.Linear(num_features, n_qubits)

    def forward(self, x):
        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from num_features to n_qubits
        #x = self.reduction(x)

        # embeding function
        q_in = torch.tanh(x) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(device)

        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params, self.q_depth, self.n_qubits).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        return q_out
        

class QuantumImagenetTransferLearning(nn.Module):
    def __init__(self, num_target_classes, backbone, num_residuals=4, use_l2 = False, use_quantum=False, q_depth=1, n_qubits=4, q_delta=0.01, n_qlayers = 1):
        super().__init__()

        ## sanity check for n_qubits: must be the same as global variable wires
        if (n_qubits*n_qlayers) != wires:
            print('[WARNING]: Number of qubits*n_qlayers: {}*{} must be the same number of wires: {}'.format(n_qubits, n_qlayers, wires))
            raise Exception("Please set a corret number of wires in .py file or change the number of qubits")
            
        # classical cnn parameters
        # number of output features
        self.n_qubits = n_qubits
        self.num_residuals = num_residuals
        self.num_filters = 512 #backbone.fc.in_features
        # all conv layers, remove last (classification) layer
        # split by conv1, resblock1,2,3, & gap
        #layers = list(backbone.children())[:-1]
        self.conv1 = nn.Sequential(*list(backbone.children())[0:4])
        if num_residuals == 1:
            self.resblock1 = nn.Sequential(*list(backbone.children())[4])
            self.num_filters = 64
        if num_residuals == 2:
            self.resblock1 = nn.Sequential(*list(backbone.children())[4])
            self.resblock2 = nn.Sequential(*list(backbone.children())[5])
            self.num_filters = 128
        if num_residuals == 3:
            self.resblock1 = nn.Sequential(*list(backbone.children())[4])
            self.resblock2 = nn.Sequential(*list(backbone.children())[5])
            self.resblock3 = nn.Sequential(*list(backbone.children())[6])
            self.num_filters = 256
        if num_residuals == 4:
            self.resblock1 = nn.Sequential(*list(backbone.children())[4])
            self.resblock2 = nn.Sequential(*list(backbone.children())[5])
            self.resblock3 = nn.Sequential(*list(backbone.children())[6])
            self.resblock4 = nn.Sequential(*list(backbone.children())[7])
            self.num_filters = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        # number of classes
        self.num_target_classes = num_target_classes
        # pre-trained network
        #self.backbone = backbone
        # use l2-gap
        self.use_l2 = use_l2
        # use quantum dresses network
        self.use_quantum = use_quantum
        # feature extractor network
        #self.feature_extractor = nn.Sequential(*layers)
        
        # normalization layer in case of use_l2 = True
        if use_l2:
            self.normalization = NormLayer()
        
        if use_quantum:
            # quantum parameters
            self.reduction = nn.Linear(self.num_filters, n_qubits* n_qlayers)

            self.n_qlayers = n_qlayers
            self.q_layers_list = list([QuantumLayer(q_depth=q_depth, n_qubits=n_qubits, q_delta=q_delta) for _ in range(n_qlayers)])
            # Using ModuleList so that this layer list can be moved to CUDA                      
            #self.q_layers = torch.nn.Sequential(*q_layers_list) #torch.nn.ModuleList(self.q_layers)

            # classification layer in case of use_quantum == True
            self.q_classifier = nn.Linear(n_qubits * n_qlayers, num_target_classes)
            
        else:
            # classification layer in case of use_quantum == False
            self.classifier = nn.Linear(self.num_filters, num_target_classes)


    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        if self.num_residuals == 1:
            x = self.resblock1(x)
        #print(x.size())
        if self.num_residuals == 2:
            x = self.resblock1(x)
            x = self.resblock2(x)
        #print(x.size())
        if self.num_residuals == 3:
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.resblock3(x)
        #print(x.size())
        if self.num_residuals == 4:
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.resblock3(x)
            x = self.resblock4(x)
        #print(x.size())
        x = self.avgpool(x)
        #print(x.size())
        x = torch.flatten(x, 1)
        #print(x.size())

        #self.feature_extractor.eval()
        #features = self.feature_extractor(x).flatten(1)
        #if self.use_l2:
        #    x = self.normalization(x)
        if not self.use_quantum:
            y = self.classifier(x)
        else:
            # split the feature in n_qlayers
            #print(features.size())
            x = self.reduction(x)

            if self.use_l2:
                 x = self.normalization(x)

            #print(x.size())

            features_split = torch.split(x, self.n_qlayers, dim=1)
            #print('featutes splited')
            #for split in features_split:
            #    print(split.size())
            # send each sub-set to one q_layer
            q_features = [q_layer(feature) for q_layer, feature in zip(self.q_layers_list, features_split)]
            #print('q_features computed')
            #for q_feature in q_features:
            #    print(q_feature.size())
            q_features = torch.cat(q_features, axis=1)
            #print('q_features merged')

            #print('q_features: ', q_features.size())

            y =  self.q_classifier(q_features)

        return y