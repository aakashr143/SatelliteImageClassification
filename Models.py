import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet152, densenet161, efficientnet_b0, vit_b_16, \
    ResNet50_Weights, ResNet152_Weights, DenseNet161_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights

from utils import NUM_FEATURES_EXTRACTED, extract_features


def weights_init(m):
    torch.nn.init.xavier_normal_(m.weight)
    torch.nn.init.constant_(m.bias, 0)
    return m


def normalize(x):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)


def create_classifier_layer(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features + NUM_FEATURES_EXTRACTED, 1024)),
        nn.ReLU(),
        weights_init(nn.Linear(1024, num_classes)),
        nn.Sigmoid()
    )
    return layer


def create_classifier_layer_no_features(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features, 1024)),
        nn.ReLU(),
        weights_init(nn.Linear(1024, num_classes)),
        nn.Sigmoid()
    )
    return layer


def create_classifier_layer_no_features_nllloss(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features, 1024)),
        nn.ReLU(),
        weights_init(nn.Linear(1024, num_classes)),
        nn.LogSoftmax(dim=1)
    )
    return layer


def create_classifier_layer_no_features_nllloss_1(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features, num_classes)),
        nn.LogSoftmax(dim=1)
    )
    return layer


def create_classifier_layer_no_features_nllloss_2(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features, 1024)),
        nn.ReLU(),
        weights_init(nn.Linear(1024, 512)),
        nn.ReLU(),
        weights_init(nn.Linear(512, num_classes)),
        nn.LogSoftmax(dim=1)
    )
    return layer


# With features --------------------------------------------------------------------------------------------------------
class LandUseModelResnet50(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelResnet50, self).__init__()
        self.name = "resnet50_eech"
        self.device = device

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.base_model.conv1(normalize(x))
        out = self.base_model.relu(self.base_model.bn1(out))
        out = self.base_model.maxpool(out)
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        out = self.base_model.layer4(out)
        out = self.base_model.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        features = extract_features(x, self.device)

        out = torch.cat([out, features], dim=1)

        return self.classifier(out)


class LandUseModelResnet152(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelResnet152, self).__init__()
        self.name = "resnet152"
        self.device = device

        self.base_model = resnet152(weights=ResNet152_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.base_model.conv1(normalize(x))
        out = self.base_model.relu(self.base_model.bn1(out))
        out = self.base_model.maxpool(out)
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        out = self.base_model.layer4(out)
        out = self.base_model.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        features = extract_features(x, self.device)

        out = torch.cat([out, features], dim=1)

        return self.classifier(out)


class LandUseModelDensenet161(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelDensenet161, self).__init__()
        self.name = "densenet161"
        self.device = device

        self.base_model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        out = self.base_model.features(normalize(x))
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))

        features = extract_features(x, self.device)

        out = out.reshape(out.shape[0], -1)
        out = torch.cat([out, features], dim=1)

        return self.classifier(out)


class LandUseModelEfficientNetB0(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelEfficientNetB0, self).__init__()
        self.name = "efficientnetb0"
        self.device = device

        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer(1280, num_classes)

    def forward(self, x):
        out = self.base_model.features(normalize(x))
        out = self.base_model.avgpool(out)

        features = extract_features(x, self.device)

        out = out.reshape(out.shape[0], -1)
        out = torch.cat([out, features], dim=1)

        return self.classifier(out)


class LandUseModelVisionTransformerB16(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelVisionTransformerB16, self).__init__()
        self.name = "visiontransformerb16"
        self.device = device

        self.base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer(768, num_classes)

    def forward(self, x):
        out = self.base_model._process_input(normalize(x))

        n = out.shape[0]

        batch_class_token = self.base_model.class_token.expand(n, -1, -1)
        out = torch.cat([batch_class_token, out], dim=1)

        out = self.base_model.encoder(out)

        out = out[:, 0]

        features = extract_features(x, self.device)

        out = out.reshape(out.shape[0], -1)
        out = torch.cat([out, features], dim=1)

        return self.classifier(out)


# No Features ----------------------------------------------------------------------------------------------------------
class LandUseModelResnet50NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelResnet50NoFeatures, self).__init__()
        self.name = "resnet50_nofeatures"
        self.device = device

        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.base_model.conv1(normalize(x))
        out = self.base_model.relu(self.base_model.bn1(out))
        out = self.base_model.maxpool(out)
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        out = self.base_model.layer4(out)
        out = self.base_model.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)


class LandUseModelResnet152NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelResnet152NoFeatures, self).__init__()
        self.name = "resnet152_nofeatures"
        self.device = device

        self.base_model = resnet152(weights=ResNet152_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.base_model.conv1(normalize(x))
        out = self.base_model.relu(self.base_model.bn1(out))
        out = self.base_model.maxpool(out)
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        out = self.base_model.layer4(out)
        out = self.base_model.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)


class LandUseModelDensenet161NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelDensenet161NoFeatures, self).__init__()
        self.name = "densenet161_nofeatures"
        self.device = device

        self.base_model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features_nllloss(self.base_model.classifier.in_features,
                                                                      num_classes)

    def forward(self, x):
        out = self.base_model.features(normalize(x))
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)


class LandUseModelEfficientNetB0NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelEfficientNetB0NoFeatures, self).__init__()
        self.name = "efficientnetb0_nofeatures"
        self.device = device

        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features(1280, num_classes)

    def forward(self, x):
        out = self.base_model.features(normalize(x))
        out = self.base_model.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)


class LandUseModelVisionTransformerB16NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelVisionTransformerB16NoFeatures, self).__init__()
        self.name = "visiontransformerb16_nofeatures"
        self.device = device

        self.base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features(768, num_classes)

    def forward(self, x):
        out = self.base_model._process_input(normalize(x))

        n = out.shape[0]

        batch_class_token = self.base_model.class_token.expand(n, -1, -1)
        out = torch.cat([batch_class_token, out], dim=1)

        out = self.base_model.encoder(out)

        out = out[:, 0]

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)
