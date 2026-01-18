import torch
import torch.nn as nn
import torch.nn.functional as F

class ScriptClassificationCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(
            in_features=64*150*150,
            out_features=512
        )
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=30)
    
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # print(x.shape)

        x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x





import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = SimpleNN()
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        """Called for each training batch"""
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # Log training loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Called for each validation batch (gradients disabled)"""
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Called for each test batch (gradients disabled)"""
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        
        # Log test metrics
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate
        )
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'  # Update learning rate after each epoch
            }
        }


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def setup(self, stage=None):
        """Called at the start of fit/validate/test"""
        # Create datasets
        train_x = torch.randn(1000, 28*28)
        train_y = torch.randint(0, 10, (1000,))
        self.train_data = TensorDataset(train_x, train_y)
        
        val_x = torch.randn(200, 28*28)
        val_y = torch.randint(0, 10, (200,))
        self.val_data = TensorDataset(val_x, val_y)
        
        test_x = torch.randn(200, 28*28)
        test_y = torch.randint(0, 10, (200,))
        self.test_data = TensorDataset(test_x, test_y)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

