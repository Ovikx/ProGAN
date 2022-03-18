import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import time
from ckpt_manager import CheckpointManager

# Use CUDA (GPU) if available
device = torch.device(type='cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Constants
ALPHA_INTERVAL = 1
BATCH_SIZE = 4
UPSCALE_INTERVAL = 100
EPOCHS = 10000
SAVE_INTERVAL = 10
MODEL_SAVE_INTERVAL = 10
BLOCK_ADD_INTERVAL = 20

class DataPreprocessor:
    def __init__(self, image_size=(8,8), cache=False):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.ToTensor()
        ])
        # C:/Users/ovikd/Code/Python/PyTorch GAN/Images
        self.train_loader = DataLoader(datasets.ImageFolder('C:/Users/ovikd/Code/Python/Pokemon GAN/images', transform=self.transform), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        self.cache = cache
        if self.cache:
            print('Caching all data...')
            start = time.time()
            self.train_cached = [batch for batch in self.train_loader]
            end = time.time()
            print(f'Cached all data in {round(end-start, 3)} s')
    
    def increase_res(self, factor):
        self.__init__(tuple(map(lambda a: a*2, self.image_size)), self.cache)

class ShapeLayer(nn.Module):
    '''
    Custom debugging layer for printing the tensor shape at any given moment
    '''
    def __init__(self):
        super(ShapeLayer, self).__init__()
    
    def forward(self, x):
        print(x.size())
        return x

class Generator(nn.Module):
    '''
    Takes in a 1D tensor (essentially a 1D array in this context) of random numbers and outputs an image represented by a 3D tensor
    '''
    def __init__(self, max_stage=8, alpha_rate=0.1):
        '''
        Defines the layer structure.
        Conv2dTranspose layers expand the image
        '''
        super(Generator, self).__init__()
        self.stage = 1
        self.max_stage = max_stage
        self.alpha = 1
        self.alpha_rate = alpha_rate
        
        self.dense_stack = nn.Sequential(
            # Creates a massive 1D tensor for the following layers to reshape into a 3D tensor
            nn.Linear(
                in_features=100,
                out_features=8*8*2048,
                bias=False,
                device=device
            ),

            # Batch normalization helps according to https://arxiv.org/pdf/1701.00160.pdf
            nn.BatchNorm1d(8*8*2048, device=device),
            nn.LeakyReLU(0.3),

            # Turns that 1D tensor into a 3D tensor
            # The target shape is (3, 256, 256) for a 256x256 RGB image
            # We will progressively transform the 3D tensor into the target shape using convolution 2D transpose layers
            nn.Unflatten(
                dim=1,
                unflattened_size=(2048,8,8)
            )

            # Output size: 2048, 8, 8
        )

        self.conv_blocks = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=1024,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    padding_mode='zeros',
                    bias=False,
                    device=device
                ),
                nn.BatchNorm2d(1024, device=device),
                nn.LeakyReLU(0.3)
                # Output size: 1024, 8, 8
            )
        ]

        self.to_rgb_new = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=1024,
                    out_channels=3,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    padding_mode='zeros',
                    bias=False,
                    device=device
                )
            )
        
        self.conv_block_container = nn.Sequential(*self.conv_blocks)
    
    def layers_to_list(self, seq):
        return [layer for layer in seq]

    def add_conv_block(self):
        self.to_rgb_old = self.to_rgb_new
        self.alpha = 0
        out_channels = int(512/(2**(self.stage-1)))

        self.conv_blocks = self.layers_to_list(self.conv_block_container)

        self.conv_blocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=int(1024/(2**(self.stage-1))),
                    out_channels=out_channels,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    padding_mode='zeros',
                    bias=False,
                    device=device
                ),
                nn.BatchNorm2d(out_channels, device=device),
                nn.LeakyReLU(0.3)
            )
        )

        self.to_rgb_new = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1,
                padding=0,
                stride=1,
                padding_mode='zeros',
                bias=False,
                device=device
            )
        )

        self.conv_block_container = nn.Sequential(*self.conv_blocks)
        self.stage += 1

    def forward(self, x):
        '''
        Passes the input through the layer structure (aka forward propagation)
        '''
        self.conv_blocks = self.layers_to_list(self.conv_block_container)
        x = self.dense_stack(x)
        for layer in self.conv_blocks[:-1]:
            x = layer(x)
        
        fading_x = self.conv_blocks[-1](x)
        fading_x = self.to_rgb_new(fading_x)

        if self.alpha < 1.0:
            x2 = F.upsample_nearest(x, scale_factor=2)
            x2 = self.to_rgb_old(x2)
            fading_x = fading_x*self.alpha + x2*(1-self.alpha)

        x = torch.sigmoid(fading_x)
        return x

class Discriminator(nn.Module):
    '''
    Predicts whether or not the input image is real. The discriminator's output range is (-inf, inf),
    but the loss function will map the output to (0, 1). 0 means fake and 1 means real.
    '''
    def __init__(self, max_stage=8, in_channels=64, alpha_rate=0.1):
        '''
        Defines the layer structure. Pretty standard convolutional network
        '''
        super(Discriminator, self).__init__()
        self.stage = 1
        self.max_stage = max_stage
        self.in_channels = in_channels
        self.alpha = 1
        self.alpha_rate = alpha_rate

        self.from_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                device=device
            ),
            nn.LeakyReLU(0.3)
        )

        self.conv_blocks = [
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=8*8*self.in_channels,
                    out_features=1,
                    device=device
                )
            )
        ]
    
        self.conv_block_container = nn.Sequential(*self.conv_blocks)
    def layers_to_list(self, seq):
        return [layer for layer in seq]

    def add_conv_block(self): 
        self.conv_blocks = self.layers_to_list(self.conv_block_container)
        self.alpha = 0
        self.conv_blocks.insert(
            0,
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    device=device
                ),
                nn.LeakyReLU(0.3)
            )
        )

        self.conv_block_container = nn.Sequential(*self.conv_blocks)
        self.stage += 1
            
    def forward(self, x):
        self.conv_blocks = self.layers_to_list(self.conv_block_container)
        x = self.from_rgb(x)
        fading_x = self.conv_blocks[0](x)
        
        if self.alpha < 1.0:
            x2 = F.avg_pool2d(x, kernel_size=2)
            fading_x = fading_x*self.alpha + x2*(1-self.alpha)
        
        for layer in self.conv_blocks[1:]:
            fading_x = layer(fading_x)

        return fading_x

class Models:
    def __init__(self, max_stage, device, alpha_rate):
        self.max_stage = max_stage
        self.alpha_rate = alpha_rate
        self.generator = Generator(max_stage=max_stage, alpha_rate=alpha_rate).to(device)
        self.discriminator = Discriminator(max_stage=max_stage, alpha_rate=alpha_rate).to(device)
    
    def restate_device(self, device):
        self.generator = Generator(max_stage=self.max_stage, alpha_rate=self.alpha_rate).to(device)
        self.discriminator = Discriminator(max_stage=self.max_stage, alpha_rate=self.alpha_rate).to(device)
    
    def add_conv_blocks(self):
        self.generator.add_conv_block()
        self.discriminator.add_conv_block()
    
    def can_grow(self):
        return self.generator.stage < self.generator.max_stage + 1

# Create the dataset
preprocessor = DataPreprocessor(
    image_size=(8,8),
    cache=True
)

# Create the generator and discriminator objects
models = Models(max_stage=4, device=device, alpha_rate=0.01)

# Define the loss function we are going to use for both the generator and the discriminator
loss_function = nn.BCEWithLogitsLoss().to(device)

# Define the models' respective optimizers
class Optimizers:
    def __init__(self, GEN_LR=0.0002, DISC_LR=0.0002):
        self.gen_opt = torch.optim.Adam(models.generator.parameters(), lr=GEN_LR)
        self.disc_opt = torch.optim.Adam(models.discriminator.parameters(), lr=DISC_LR)
    
    def refresh_params(self):
        self.__init__()

optimizers = Optimizers()

# Define the checkpoint manager that will help save/load models automatically (docs: https://pypi.org/project/pytorch-ckpt-manager/)
manager = CheckpointManager(
    assets={
        'gen' : models.generator.state_dict(),
        'disc' : models.discriminator.state_dict(),
        'gen_opt' : optimizers.gen_opt.state_dict(),
        'disc_opt' : optimizers.disc_opt.state_dict()
    },
    directory='training_ckpts',
    file_name='progan_states',
    maximum=3
)

# Load the states from the checkpoint directory if they exist
load_data = manager.load()
models.generator.load_state_dict(load_data['gen'])
models.discriminator.load_state_dict(load_data['disc'])
optimizers.gen_opt.load_state_dict(load_data['gen_opt'])
optimizers.disc_opt.load_state_dict(load_data['disc_opt'])

# Create a bunch of tensors populated by random floats
# This is the input for the generator to create sample images that will be saved
seed = torch.randn((16, 100), device=device)

# Pray to NVIDIA that this line actually helps performance
torch.backends.cudnn.benchmark = True
'''def n_params():
    cnt = 0
    for param in models.generator.parameters():
        for v in param:
            cnt +=1 
    return cnt

print(n_params())
for i in range(3):
    models.generator.add_conv_block()
    print(n_params())
    print(models.generator)
import sys;sys.exit()'''
def save_predictions(epoch, z):
    '''
    Saves a sample of generated images
    '''
    with torch.no_grad():
        predictions = (models.generator(z).cpu().detach().numpy()*255).astype('int32')
    fig = plt.figure(figsize=(12, 12))

    for i, image in enumerate(predictions):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.moveaxis(image, 0, -1), cmap=None)
        plt.axis('off')
    
    plt.savefig(f'pokemon_generations/gen_{epoch}')
    plt.close()

def train(epochs):
    '''
    Trains the generator and discriminator
    '''

    loader_size = len(preprocessor.train_cached)
    print('Starting training...')
    for epoch in range(epochs):
        # Start recording stats for display
        start = time.time()
        running_g_loss = torch.tensor([0], dtype=torch.float16, device=device)
        running_d_loss = torch.tensor([0], dtype=torch.float16, device=device)
        
        # Train each batch in the dataset
        for data in preprocessor.train_cached:
            # Get the real images and their respective labels
            images, labels = data[0].to(device), torch.ones((BATCH_SIZE,1), device=device)

            # Zero the gradients
            for param in models.generator.parameters():
                param.grad = None
            for param in models.discriminator.parameters():
                param.grad = None

            # Get the generator's images
            noise = torch.randn((BATCH_SIZE, 100), device=device)
            fake_images = models.generator(noise)

            # Backprop for the discriminator's guesses on the real images
            # Goal for the discriminator is to correctly guess that the real images are real
            # In theory, the loss function will compare the discriminator's predictions to 1.0 because 1.0 means real
            # We are using 0.9 because discriminator over-confidence can harm the generator's training
            real_guess = models.discriminator(images)
            disc_real_loss = loss_function(real_guess, torch.full_like(labels, 0.9, device=device))
            running_d_loss += disc_real_loss.to(torch.float16)
            disc_real_loss.backward()

            # Backprop for the discriminator's guesses on the fake images
            # 2nd goal for the discriminator is to correctly guess that the fake images are fake
            # The loss function will compare the discriminator's predictions to 0.0 because 0.0 means fake
            fake_guess = models.discriminator(fake_images.detach())
            disc_fake_loss = loss_function(fake_guess, torch.zeros_like(fake_guess, device=device))
            running_d_loss += disc_fake_loss.to(torch.float16)
            disc_fake_loss.backward()
            
            # Generator loss
            # Goal for the generator is to fool the discriminator into thinking that its generated images are real
            # The loss function will compare the discriminator's predictions to 1.0 because the generator wants the discriminator to think its images are real
            fake_guess = models.discriminator(fake_images)
            gen_loss = loss_function(fake_guess, torch.ones_like(fake_guess, device=device))
            running_g_loss += gen_loss.to(torch.float16)
            gen_loss.backward()

            # Update the weights for both models
            optimizers.gen_opt.step()
            optimizers.disc_opt.step()

        # Save sample images
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_predictions(epoch+1, seed)

        # Adds a new conv block to both models after a variable # of epochs
        if (epoch + 1) % UPSCALE_INTERVAL == 0:
            if models.can_grow():
                preprocessor.increase_res(factor=2)
                models.add_conv_blocks()
                optimizers.refresh_params()
                print(f'Switched to image size {preprocessor.image_size}')
        
        if (epoch + 1) % ALPHA_INTERVAL == 0 and models.can_grow():
            models.generator.alpha += models.generator.alpha_rate
            models.discriminator.alpha += models.discriminator.alpha_rate
        
        # Save the model and optimizer states to a folder
        '''if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            manager.save()'''

        end = time.time()

        # Print the stats for the current epoch
        print(f'Epoch {epoch+1} || Gen loss: {(running_g_loss/loader_size).item()} || Disc loss: {(running_d_loss/loader_size).item()} || {round(end-start, 3)} seconds')

# Set your computer on fire
train(EPOCHS)