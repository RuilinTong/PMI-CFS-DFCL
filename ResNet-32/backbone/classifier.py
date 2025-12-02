# -*-coding:utf8-*-

import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    feature_mode: bool = False

    def __init__(
        self,
        num_classes: int,
        num_features: int = 2048,
        bias: bool = True,
        neck: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.bias = bias

        self.setup(neck)

    @property
    def embeddings(self):
        return self.classifier.weight

    def setup(self, neck: nn.Module = nn.Identity()):
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = neck

        args = [self.num_features, self.num_classes]
        self.classifier = self._create_classifier(*args, bias=self.bias)

    def classify(self, input: torch.Tensor):
        return self.classifier(input)

    def forward(self, input):
        output = self.pool(input)
        output = output[:, :, 0, 0]
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)
        return output

    def _create_classifier(
        self,
        num_features: int,
        num_classes: int,
        bias=True,
    ):
        assert num_features > 0 and num_classes > 0

        classifier = nn.Linear(num_features, num_classes, bias=bias)
        weights_init_classifier(classifier)

        return classifier


class DynamicSimpleHead(SimpleHead):
    def __init__(
        self,
        num_classes: int = 0,
        num_features: int = 2048,
        bias: bool = True,
        neck: nn.Module = nn.Identity(),
    ):
        super().__init__(num_classes, num_features, bias, neck)

    @property
    def embeddings(self):
        weights = [classifier.weight for classifier in self.classifiers]
        return torch.cat(weights)

    def setup(self, neck: nn.Module = nn.Identity()):
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = neck

        self.classifiers = nn.ModuleList()

        if self.num_classes > 0:
            self.append(self.num_classes)

    def append(self, num_classes: int):
        args = [self.num_features, num_classes]
        classifier = self._create_classifier(*args, bias=self.bias)

        self.classifiers.append(classifier)
        if len(self.classifiers) > 1 or self.num_classes == 0:
            self.num_classes += num_classes

    def classify(self, input: torch.Tensor):
        output = [classifier(input) for classifier in self.classifiers]
        output = torch.cat(output, dim=1)
        return output

    def forward(self, input):
        output = self.pool(input)
        output = output[:, :, 0, 0]
        output = self.neck(output)

        if self.feature_mode:
            return output

        output = self.classify(output)

        return output


class ExpandableHead(torch.nn.Module):
    def __init__(self, in_features, num_classes, bias=False):
        super(ExpandableHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.bias = bias
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = []
        if self.num_classes > 0:
            self.heads.append(
                torch.nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=self.bias))
        self.feature_mode = False

    def forward(self, x):
        feat = self.pool(x)[:, :, 0, 0]
        if self.feature_mode:
            return feat
        outputs = []
        for hi in self.heads:
            y = hi(x)
            outputs.append(y)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def add_classifier(self, classifier):
        self.heads.append(classifier)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
