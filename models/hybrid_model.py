import torch
import torch.nn as nn
import timm

class HybridOCTModel(nn.Module):
    """
    Hybrid architecture combining EfficientNet-B3 (CNN) and Swin Transformer.
    This model fuses local features from CNN and global dependencies from Transformer.
    """
    def __init__(self, num_classes=4):
        super(HybridOCTModel, self).__init__()
        # Load Pretrained Models
        self.cnn = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
        self.transformer = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=0)
        
        # Feature Concatenation
        cnn_out = self.cnn.num_features
        transformer_out = self.transformer.num_features
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + transformer_out, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        # Fuse features
        combined_features = torch.cat((cnn_features, transformer_features), dim=1)
        return self.fc(combined_features)
