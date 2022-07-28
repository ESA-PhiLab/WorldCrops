import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding='same', groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvDown(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvDown, self).__init__()

        self.conv_block = nn.Sequential(  
            depthwise_separable_conv(input_dim, 1, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim),
            nn.ReLU()     
        )

    def forward(self, x):
        return self.conv_block(x)



class ResidualDown(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualDown, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2),
        )

    def forward(self, x):
        xpad = F.pad(x, [0,1,0,1], mode='replicate')
        return self.conv_skip(xpad)



class ResidualBridge(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBridge, self).__init__()

        self.conv_block = nn.Sequential(  
            depthwise_separable_conv(input_dim, 1, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            depthwise_separable_conv(output_dim, 1, output_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.conv_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, padding='same'),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class ResidualUp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualUp, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, padding='same'),
        )

    def forward(self, x):
        return self.conv_skip(x)


class ConvUp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvUp, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim +input_dim //2, output_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')        
        )

    def forward(self, x):
        return self.conv_block(x) 



class ResUnet(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256], dropout = 0.3):
        super(ResUnet, self).__init__()

        ## first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            depthwise_separable_conv(filters[0], 1, filters[0]),
            nn.ReLU()
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=2, )
        )
        self.pool = nn.MaxPool2d(2) 


        self.down1 = ConvDown(filters[0], filters[1])
        self.res1 = ResidualDown(filters[0], filters[1])
        self.down2 = ConvDown(filters[1], filters[2])
        self.res2 = ResidualDown(filters[1], filters[2])
        
        self.bridge = ResidualBridge(filters[2], filters[3])

        self.up1 = ConvUp(filters[3], filters[2])
        self.res3 = ResidualUp(filters[3], filters[2])
        self.up2 = ConvUp(filters[2], filters[1])
        self.res4 = ResidualUp(filters[2], filters[1])

      ## last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[1] + filters[1] //2, filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(filters[0], 1, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )


    def forward(self, x):
        xpad = F.pad(x, [0,1,0,1], mode='replicate')
        x1 = self.input_layer(x) 
        x1b = self.pool(x1) + self.input_skip(xpad)

        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)

        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)

        x4 = self.bridge(x3b)
        
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4)

        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)

        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output

class ResUnetEncoder(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256], dropout = 0.3):
        super(ResUnetEncoder, self).__init__()

        ## first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            depthwise_separable_conv(filters[0], 1, filters[0]),
            nn.ReLU()
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=2, )
        )
        self.pool = nn.MaxPool2d(2) 


        self.down1 = ConvDown(filters[0], filters[1])
        self.res1 = ResidualDown(filters[0], filters[1])
        self.down2 = ConvDown(filters[1], filters[2])
        self.res2 = ResidualDown(filters[1], filters[2])
        
        self.bridge = ResidualBridge(filters[2], filters[3])

    def forward(self, x):
        xpad = F.pad(x, [0,1,0,1], mode='replicate')
        x1 = self.input_layer(x) 
        x1b = self.pool(x1) + self.input_skip(xpad)

        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)

        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)

        x4 = self.bridge(x3b) 

        embeddings = [x1, x2, x3, x4 ]
        return embeddings

class ResUnetDecoder(nn.Module):
    def __init__(self, filters=[32, 64, 128, 256], dropout = 0.3):
        super(ResUnetDecoder, self).__init__()

        self.pool = nn.MaxPool2d(2) 

        self.up1 = ConvUp(filters[3], filters[2])
        self.res3 = ResidualUp(filters[3], filters[2])
        self.up2 = ConvUp(filters[2], filters[1])
        self.res4 = ResidualUp(filters[2], filters[1])

      ## last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[1] + filters[1] //2, filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(filters[0], 1, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )


    def forward(self, x):
        #x ist a list of embeddings from encoder
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4)

        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)

        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output

class UNet_Transfer(pl.LightningModule):

    def __init__(self, lr = 0.0002, backbone=None, dropout = 0.3, filters=[32, 64, 128, 256], batch_size  = 3, finetune= False, seed=42):
        super().__init__()
        """
        Args:
            backbone: pretrained encoder
            finetune: if false -> don't update parameters of encoder 
                     if true > update all parameters (encoder+ decoder)
        """

        self.model_type = 'UNET_Transfer'
        self.finetune = finetune

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.ce = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()
        self.save_hyperparameters()

        if backbone == None:
            print('Backbone/head not loaded')
            return

        #layers
        self.encoder = backbone
        self.decoder = ResUnetDecoder(filters = filters, dropout = dropout)

        if self.finetune == False:
            # freeze params of encoder
            for param in self.encoder.parameters():
                param.requires_grad = False


    def forward(self,x):
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        #self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        #self.logger.experiment.add_scalar('train_loss', loss, global_step=self.global_step)

        y_true = y.detach()
        #y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()
        #add here accuracy

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.decoder.parameters(),lr = self.lr)
        
        return optimizer

    def test_step(self, test_batch, batch_idx):
        pass

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass