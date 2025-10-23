import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetworkEnhanced(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=6, use_chunk_fusion=True):
        super(SiameseNetworkEnhanced, self).__init__()
        self.use_chunk_fusion = use_chunk_fusion

        self.query_encoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        if self.use_chunk_fusion:
            self.chunk_encoder = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            self.fusion_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU()
            )
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU()
            )

        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(128, num_classes)
        self.param_regressor = nn.Linear(128, 3)

    def forward(self, query_embed, chunk_embed=None):
        query_features = self.query_encoder(query_embed)

        if self.use_chunk_fusion and chunk_embed is not None:
            chunk_features = self.chunk_encoder(chunk_embed)
            fused = torch.cat([query_features, chunk_features], dim=1)
            fused_features = self.fusion_layer(fused)
        else:
            fused_features = self.fusion_layer(query_features)

        features = self.feature_extractor(fused_features)
        class_output = self.classifier(features)
        param_output = self.param_regressor(features)

        return class_output, param_output


class ContrastiveAndRegressionLossEnhanced(nn.Module):
    def __init__(self, margin=1.0, alpha=1.0, beta=1.0, gamma=1.0):
        super(ContrastiveAndRegressionLossEnhanced, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def forward(self, output1, output2, label,
                class_output1, class_output2, class_label1, class_label2,
                param_output1, param_output2, param_label1, param_label2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        classification_loss1 = self.classification_loss(class_output1, class_label1)
        classification_loss2 = self.classification_loss(class_output2, class_label2)
        classification_loss = (classification_loss1 + classification_loss2) / 2

        regression_loss1 = self.regression_loss(param_output1, param_label1)
        regression_loss2 = self.regression_loss(param_output2, param_label2)
        regression_loss = (regression_loss1 + regression_loss2) / 2

        total_loss = (self.alpha * contrastive_loss +
                      self.beta * classification_loss +
                      self.gamma * regression_loss)

        return total_loss


class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=6):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.param_regressor = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        class_output = self.classifier(x)
        param_output = self.param_regressor(x)
        return class_output, param_output


class ContrastiveAndRegressionLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveAndRegressionLoss, self).__init__()
        self.margin = margin
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def forward(self, output1, output2, label, class_output1, class_output2,
                class_label1, class_label2, param_output1, param_output2,
                param_label1, param_label2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        classification_loss1 = self.classification_loss(class_output1, class_label1)
        classification_loss2 = self.classification_loss(class_output2, class_label2)
        classification_loss = (classification_loss1 + classification_loss2) / 2

        regression_loss1 = self.regression_loss(param_output1, param_label1)
        regression_loss2 = self.regression_loss(param_output2, param_label2)
        regression_loss = (regression_loss1 + regression_loss2) / 2

        total_loss = contrastive_loss + classification_loss + regression_loss
        return total_loss
