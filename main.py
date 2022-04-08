import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
#Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#plotting
import matplotlib.pyplot as plt 
import matplotlib 
import os 
import numpy as np 
from median_filter import MedianFilterer
import sys 

'''
median_filterer = MedianFilterer(kernel_size = 2, stride=1, padding=0, same=True)
#test_one =  torch.tensor(np.arange(1,10).reshape((3,3)))
test_two = torch.tensor(np.arange(1,28).reshape((3,3,3)), dtype=torch.float64)
test_two = torch.unsqueeze(test_two, dim=0)

#test_one = median_filterer(test_one)
test_two = median_filterer(test_two)
print(f'test_two {test_two}') 
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
test_set = CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False)
CHECKPOINT_PATH = "./checkpoints/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

test_image = train_dataset[0][0]
WIDTH = test_image.size()[0]
HEIGHT = test_image.size()[1]


###HELPER FUNCTIONS
def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

def add_noise(x, prob_zero=0.3):
    #zero out mask with uniform probability of being zero at each element
    random_zero_mask = torch.bernoulli((1-prob_zero)*torch.ones_like(x))
    return torch.mul(x, random_zero_mask)

class ZeroMaskNoise(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if self.training:
            return add_noise(x)
        else:
            return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding="same")
        ##TODO: change so normalize only the feature axis
        self.bn1 = nn.BatchNorm2d(out_channels)
        ##no shared parameters across filter
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=stride, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample 
    def forward(self,x):
        residual = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual=self.downsample(x)
        out += residual 
        #ONE MORE RELU
        out = self.relu(out)
        return out 
        
class Fully_Conv(pl.LightningModule):
    n_init_features = 64
    def __init__(self, width: int, height: int, num_input_channels: int = 3,
        res_block: object = ResidualBlock, median_filterer: object = MedianFilterer):
        super().__init__()
        self.save_hyperparameters()
        self.res_block = res_block(in_channels=n_init_features) 
        self.median_filterer = MedianFilterer(kernel_size = 5, stride=1, padding=0, same=True)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self,x):
        x5 = median_filterer(x)
        x5 = median_filterer(x5)
        x = x5
        x = nn.Conv2d(num_input_channels, n_init_features, kernel_size=(3,3), stride=stride, padding="same")(x)
        x = nn.ReLU(inplace=True)(x)
        for i in range(32):
            x = nn.Conv2d(n_init_features, n_init_features, kernel_size=(3,3), stride=stride, padding="same")(x)
            ##TODO: change so normalize only the feature axis
            x = nn.BatchNorm2d(n_init_features)
            x = nn.Relu(inplace=True)(x)
            x = self.res_block(x)
            if i < 16:
                x = self.median_filterer(x)
        x_hat = nn.Conv2d(n_init_features, 3, kernel_size=(3,3), stride=stride, padding="same")(x)
        return x_hat 
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)                             
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
    
class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs 
        self.every_n_epochs = every_n_epochs
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

def train_cifar(experiment_name):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{experiment_name}"), max_epochs=10, 
        gpus=2 if str(device).startswith("cuda") else 0,
    callbacks=[ModelCheckpoint(save_weights_only=True, dirpath=CHECKPOINT_PATH, filename="denoiser-{epoch:02d}-{val_loss:.2f}",save_top_k=1, mode="min", monitor="val_loss"),
                GenerateCallback(get_train_images(8), every_n_epochs=10),
                LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True 
    trainer.logger._default_hp_metric = None 
    PRELOAD = False  
    if PRELOAD:
        print("Loading pretrained..")
        pretrained_filename = os.path.join(CHECKPOINT_PATH, f"denoiser-epoch=495-val_loss=28.88.ckpt")
        model = Fully_Conv.load_from_checkpoint(pretrained_filename)
    else:
        model = Fully_Conv(width=WIDTH, height=HEIGHT)
        trainer.fit(model, train_loader, val_loader)
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result 

#train
model_ld, result_ld = train_cifar()

def visualize_reconstructions(model, input_imgs):
    noisy_imgs = add_noise(input_imgs, prob_zero=0.60)
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(noisy_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()
    #plotting
    imgs = torch.stack([input_imgs, noisy_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, range = (-1,1))
    grid = grid.permute(1,2,0)
    plt.figure(figsize=(7,4.5))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'reconstruction.png')
input_imgs = get_train_images(4)
visualize_reconstructions(model_ld, input_imgs)

def visualize_noise(input_imgs):
    noisy_imgs = add_noise(input_imgs)
    #plotting
    imgs = torch.stack([input_imgs, noisy_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range = (-1,1))
    grid = grid.permute(1,2,0)
    plt.figure(figsize=(7,4.5))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'noisy.png')
    print("showed")
#visualize_noise(input_imgs)





