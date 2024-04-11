from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch


class CustomMobileNetV2(torch.nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        # load weights and model for mobilenet_v2
        self.pretrained_weights = MobileNet_V2_Weights.DEFAULT
        self.pretrained_model = mobilenet_v2(weights=self.pretrained_weights)
        # freeze parameters from original model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # replace final linear classifier with binary classifier (trainable by default)
        self.pretrained_model.classifier[1] = torch.nn.Linear(in_features=self.pretrained_model.classifier[1].in_features, out_features=2)

        # load better weights for linear classifier
        if load_weights == True:
            self.pretrained_model.classifier[1].load_state_dict(torch.load('linear_classifier.th'))

        # load transforms for inference
        self.preprocess = self.pretrained_weights.transforms(antialias=True)

    def forward(self, x):
        batch = self.preprocess(x)
        outputs = self.pretrained_model(batch)
        return outputs

def save_model(model, index):
    from torch import save
    from os import path
    base_name = 'linear_classifier'
    suffix = '.th'
    epoch = str(index)
    training_path = 'training_outputs'
    new_name = base_name + epoch + suffix
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), training_path, new_name))

def load_model(pretrained_model):
    from torch import load
    from os import path

    completed_model = pretrained_model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'linear_classifier.th'), map_location='cpu'))
    return completed_model