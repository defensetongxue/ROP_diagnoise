from torchvision import models
import os
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def build_vgg16(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, config["num_classes"])
    return model
    
def build_mobilenetv3_large(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(1280, config["num_classes"])
    print(f"MobileNetV3 Large has {count_parameters(model)} parameters")
    return model

def build_mobilenetv3_small(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, config["num_classes"])
    print(f"MobileNetV3 Small has {count_parameters(model)} parameters")
    return model

def build_resnet18(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, config["num_classes"])  # ResNet18 has 512 out_features in its last layer
    print(f"ResNet18 has {count_parameters(model)} parameters")
    return model

def build_resnet50(config):
    os.environ['TORCH_HOME'] = config["official_model_save"]
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, config["num_classes"])  # ResNet50 has 2048 out_features in its last layer
    print(f"ResNet50 has {count_parameters(model)} parameters")
    return model
def build_class_model(model_name, model_configs):
    builders = {
        "vgg16": build_vgg16,
        "mobilenetv3_large": build_mobilenetv3_large,
        "mobilenetv3_small": build_mobilenetv3_small,
        "resnet18": build_resnet18,
        "resnet50": build_resnet50,
    }
    
    try:
        return builders[model_name](model_configs),nn.CrossEntropyLoss()
    except KeyError:
        raise ValueError(f"Invalid model name: {model_name}. Available options are: {list(builders.keys())}")

if __name__ =="__main__":
    os.makedirs('./experiments',exist_ok=True)
    configs={
        "official_model_save":'./experiments',
        "num_classes":3
    }
    build_resnet18(configs)
    build_resnet50(configs)