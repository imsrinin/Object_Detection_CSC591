
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
import copy


###################################################################################################################

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:

        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:

        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

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

        x = self.dequant(x) 

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def qResNet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
###################################################################################################################

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
    #  transforms.RandomRotation(15),
    #  transforms.RandomVerticalFlip(),
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


class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
    
# model = QuantizedResNet18(model_)

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

    if mode == 'global_structured':
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.LnStructured,
                amount=pruning_rate)

    if mode == 'local_unstructured':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=pruning_rate)

    if mode == 'local_structured':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # prune.ln_structured(module, 'weight', amount=pruning_rate)
                prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)



    return model

model.train()
model.to(device)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare_qat(model,inplace=True)
start_time = time.time()


for i in range(1): 
    # print(f"Iteration {i+1}: Pruning model...")
    # model = prune_model(model, 0.94, mode = 'local_unstructured')

    # print("Training pruned model...")

    for epoch in range(100):
        # if epoch == 90:
            # model = prune_model(model, 0.94, mode = 'local_unstructured')      
            # print("Training pruned model...")
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
            



# for name, module in model.named_modules():
#         # if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#         if isinstance(module, nn.Conv2d):
#             prune.remove(module, 'weight')

end_time = time.time()


print('total time taken for training iterative pruning model in cuda is : ', (end_time-start_time))

# torch.save(model.state_dict(), 'model.pth')

# model = qResNet50()
# model.load_state_dict(torch.load('model.pth'))

model.to('cpu')

model.eval()
torch.quantization.convert(model, inplace = True)

#inference latency
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

# # model.qconfig = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint4x2, qscheme=torch.per_tensor_symmetric),
# #                                     weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint4x2, qscheme=torch.per_tensor_symmetric))
# # model = torch.quantization.quantize_dynamic(
# #     model,  # the model to be quantized
# #     {torch.nn.Conv2d, torch.nn.Linear},
# #      dtype=torch.qint8
# # )
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


# torch.quantization.convert(model, inplace = True) # Quantize the model

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