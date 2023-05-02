
#Imports
from torchvision import datasets
from torchvision import transforms 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import Tensor
from torch.quantization import QuantStub, DeQuantStub, quantize_jit, default_qconfig
from torch.hub import load_state_dict_from_url
import torch.nn.utils.prune as prune
import numpy as np
import time
from typing import Type, Any, Callable, Union, List, Optional
import matplotlib.pyplot as plt

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None) -> None:

        super(BasicBlock, self).__init__()
        
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None) -> None:

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.ff = nn.quantized.FloatFunctional() # to simulate qunatized operations on float tensors (https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.FloatFunctional.html)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ff.add(out, identity)# (https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.FloatFunctional.html)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        ) -> None:

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # stubs to aid qunatization process
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        # adding quant stub to indicate qunatization start
        x = self.quant(x) 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # adding dequant stub to indicate qunatization end
        x = self.dequant(x) 

        return x


def qResNet50(
    arch: str = 'resnet50',
    pretrained: bool = True, 
    progress: bool = True, **kwargs: Any) -> ResNet:

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 10
batch_size = 128
learning_rate = 0.01

classes = ('plane', 'car' , 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])

train_dataset = torchvision.datasets.CIFAR10( root= './data', train = True,
                                                download =True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(  root= './data', train = False,
                                                download =True, transform = transform_test)


train_loader = torch.utils.data.DataLoader(train_dataset
    , batch_size = batch_size
    , shuffle = True,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset
    , batch_size = batch_size
    , shuffle = True,
    num_workers=2)

n_total_step = len(train_loader)


# model_ = models.resnet50(pretrained = True)
model = qResNet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def prune_model(model, pruning_rate, mode):

    if mode == 'global_unstructured':
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate)

    if mode == 'local_unstructured':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=pruning_rate)

    if mode == 'local_structured':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)

    return model

train_loss = []
train_acc = []

model.train()
model.to(device)
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# torch.quantization.prepare_qat(model,inplace=True)
start_time = time.time()


for i in range(1): 
    # print(f"Iteration {i+1}: Pruning model...")
    # model = prune_model(model, 0.25, mode = 'local_structured')

    # print("Training pruned model...")

    for epoch in range(100):
        if epoch == 90:
            model = prune_model(model, 0.94, mode = 'local_unstructured')      
            print("Training pruned model...")
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            n_corrects = (output.argmax(axis=1)==target).sum().item()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx+1) % 20 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step: {batch_idx+1}/{n_total_step}: loss = {loss:.5f}, acc = {100*(n_corrects/target.size(0)):.2f}%')

        train_loss.append(loss.item())
        train_acc.append(100*(n_corrects/target.size(0)))
            
            

for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        # if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

end_time = time.time()

plt.plot(train_loss, label='loss')
plt.plot(train_acc, label= 'accuracy')
plt.xlabel('iter')
plt.ylabel('loss and accuracy')
plt.legend()
plt.savefig(f'oneshot_training_curve.png')

print('total time taken for training iterative pruning model in cuda is : ', (end_time-start_time))



model.to('cpu')

model.eval()
# torch.quantization.convert(model, inplace = True)

# #inference latency
inf_base_start = time.time()

with torch.no_grad():
    number_corrects = 0
    number_samples = 0
    for i, (test_images_set , test_labels_set) in enumerate(test_loader):
        test_images_set = test_images_set.to('cpu')
        test_labels_set = test_labels_set.to('cpu')
    
        y_predicted = model(test_images_set)
        labels_predicted = y_predicted.argmax(axis = 1)
        number_corrects += (labels_predicted==test_labels_set).sum().item()
        number_samples += test_labels_set.size(0)
    print(f'Overall accuracy of baseline model is {(number_corrects / number_samples)*100}%')

inf_base_end = time.time()

print('total time taken for inference in cpu for pruned model is : ', (inf_base_end-inf_base_start))

# model.eval()

# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# torch.quantization.prepare(model, inplace = True)


# calib = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=32,
#     sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=1000),
#     num_workers=4,
#     pin_memory=True
# )

# with torch.no_grad():
#     for i, (test_images_set , test_labels_set) in enumerate(calib):
#         test_images_set = test_images_set.to('cpu')
#         test_labels_set = test_labels_set.to('cpu')
    
#         y_predicted = model(test_images_set)


# torch.quantization.convert(model, inplace = True) 

# model.eval()

# inf_quant_start = time.time()

# with torch.no_grad():
#     number_corrects = 0
#     number_samples = 0
#     for i, (test_images_set , test_labels_set) in enumerate(test_loader):
#         test_images_set = test_images_set.to('cpu')
#         test_labels_set = test_labels_set.to('cpu')
#         y_predicted = model(test_images_set)
#         labels_predicted = y_predicted.argmax(axis = 1)
#         number_corrects += (labels_predicted==test_labels_set).sum().item()
#         number_samples += test_labels_set.size(0)
#     print(f'Overall accuracy of quantized model is {(number_corrects / number_samples)*100}%')

# inf_quant_end = time.time()


# print('total time taken for inference in cpu for pruned and qunatized model is : ', (inf_quant_end-inf_quant_start))


# def print_size_of_model(model, label=""):
#     torch.save(model.state_dict(), "temp.p")
#     size=os.path.getsize("temp.p")
#     print("model: ",label,' \t','Size (KB):', size/1e3)
#     os.remove('temp.p')
#     return size

# # compare the sizes
# f=print_size_of_model(float_lstm,"fp32")
# q=print_size_of_model(quantized_lstm,"int8")
# print("{0:.2f} times smaller".format(f/q))



#April 21st
# total time taken for training in cuda is :  1355.2856440544128
# Overall accuracy of baseline model is 87.79%
# total time taken for inference in cpu for baseline is :  6.678455114364624
# /home/snaray23/anaconda3/envs/ro/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# Overall accuracy of quantized model is 87.57000000000001%
# total time taken for inference in cpu for qunatized model is :  2.690927505493164

# local_unstructured

# 100*.8**10
# total time taken for training iterative pruning model in cuda is :  1449.4395537376404
# Overall accuracy of baseline model is 87.47%
# total time taken for inference in cpu for pruned model is :  6.844406366348267
# /home/snaray23/anaconda3/envs/ro/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# Overall accuracy of quantized model is 87.1%
# total time taken for inference in cpu for pruned and qunatized model is :  2.7638072967529297


# 100*.75**10
# total time taken for training iterative pruning model in cuda is :  1437.9992098808289
# Overall accuracy of baseline model is 87.35000000000001%
# total time taken for inference in cpu for pruned model is :  6.825281143188477
# /home/snaray23/anaconda3/envs/ro/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# Overall accuracy of quantized model is 86.69%
# total time taken for inference in cpu for pruned and qunatized model is :  2.6152946949005127


# global_unstructured
# total time taken for training iterative pruning model in cuda is :  1441.416865348816
# Overall accuracy of baseline model is 88.03%
# total time taken for inference in cpu for pruned model is :  6.759032964706421
# /home/snaray23/anaconda3/envs/ro/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# Overall accuracy of quantized model is 87.78%
# total time taken for inference in cpu for pruned and qunatized model is :  3.0035369396209717


#local structured
# total time taken for training iterative pruning model in cuda is :  1427.3718922138214
# Overall accuracy of baseline model is 75.94999999999999%
# total time taken for inference in cpu for pruned model is :  6.84194278717041
# /home/snaray23/anaconda3/envs/ro/lib/python3.10/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# Overall accuracy of quantized model is 74.53999999999999%
# total time taken for inference in cpu for pruned and qunatized model is :  2.7970638275146484

#one shot local unstructured 0.94 pruning
# total time taken for training iterative pruning model in cuda is :  1374.6262357234955
# Overall accuracy of baseline model is 84.35000000000001%
# total time taken for inference in cpu for pruned model is :  6.834751605987549

# QAT
# total time taken for training iterative pruning model in cuda is :  4070.051666021347
# Overall accuracy of baseline model is 87.8%
# total time taken for inference in cpu for pruned model is :  2.7828962802886963