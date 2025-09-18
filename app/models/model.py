import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptchaCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(CaptchaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: 64x64 -> After 4 pooling operations: 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth conv block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ImprovedCaptchaCNN(nn.Module):
    """Improved CNN with residual connections and better architecture for GPU utilization"""
    def __init__(self, num_classes, input_channels=1):
        super(ImprovedCaptchaCNN, self).__init__()
        
        # Initial convolution with larger channels for better GPU utilization
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with more channels
        self.layer1 = self._make_layer(128, 128, 3)
        self.layer2 = self._make_layer(128, 256, 3, stride=2)
        self.layer3 = self._make_layer(256, 512, 3, stride=2)
        self.layer4 = self._make_layer(512, 1024, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with dropout
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class BasicBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class LightweightCaptchaCNN(nn.Module):
    """Truly lightweight CNN for captcha recognition - <500K parameters"""
    def __init__(self, num_classes, input_channels=1):
        super(LightweightCaptchaCNN, self).__init__()
        
        # Very simple feature extraction
        self.features = nn.Sequential(
            # Block 1 - Small channels
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # Smaller final size
        )
        
        # Very simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),  # High dropout
            nn.Linear(128 * 2 * 2, 128),  # Much smaller
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MicroCaptchaCNN(nn.Module):
    """Ultra-lightweight CNN - <200K parameters"""
    def __init__(self, num_classes, input_channels=1):
        super(MicroCaptchaCNN, self).__init__()
        
        # Minimal feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Minimal classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.8),  # Very high dropout
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class OptimizedCaptchaCNN(nn.Module):
    """Optimized CNN with good GPU utilization but appropriate size to prevent overfitting"""
    def __init__(self, num_classes, input_channels=1):
        super(OptimizedCaptchaCNN, self).__init__()
        
        # Feature extraction with moderate complexity
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),  # Spatial dropout
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # Higher dropout
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class HighPerformanceCaptchaCNN(nn.Module):
    """High-performance CNN optimized for GPU utilization"""
    def __init__(self, num_classes, input_channels=1):
        super(HighPerformanceCaptchaCNN, self).__init__()
        
        # Feature extraction with more channels for GPU utilization
        self.features = nn.Sequential(
            # Block 1 - Larger channels
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier with more parameters
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_model(model_type='lightweight', num_classes=10, input_channels=1):
    """Create a model based on the specified type"""
    
    if model_type == 'basic':
        return CaptchaCNN(num_classes, input_channels)
    elif model_type == 'improved':
        return ImprovedCaptchaCNN(num_classes, input_channels)
    elif model_type == 'lightweight':
        return LightweightCaptchaCNN(num_classes, input_channels)
    elif model_type == 'micro':
        return MicroCaptchaCNN(num_classes, input_channels)
    elif model_type == 'optimized':
        return OptimizedCaptchaCNN(num_classes, input_channels)
    elif model_type == 'high_performance':
        return HighPerformanceCaptchaCNN(num_classes, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the models
    num_classes = 10  # Example: digits 0-9
    
    models = {
        'micro': create_model('micro', num_classes),
        'lightweight': create_model('lightweight', num_classes),
        'basic': create_model('basic', num_classes),
        'optimized': create_model('optimized', num_classes),
        'improved': create_model('improved', num_classes),
        'high_performance': create_model('high_performance', num_classes)
    }
    
    # Test with dummy input
    dummy_input = torch.randn(1, 1, 64, 64)
    
    for name, model in models.items():
        print(f"\n{name.upper()} Model:")
        print(f"Parameters: {count_parameters(model):,}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Expected shape: (1, {num_classes})")
