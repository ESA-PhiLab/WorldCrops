import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

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
            nn.ReLU()
        )


    def forward(self, x):
        return self.conv_block(x) 

class ResidualUp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualUp, self).__init__()

        self.conv_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, padding='same')
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


class ResUnetEncoder(nn.Module):
    def __init__(self, channel=1, dropout = 0.3, filters=None):
        super(ResUnetEncoder, self).__init__()
        args = filters

        ## first layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, args[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            depthwise_separable_conv(args[0], 1, args[0]),
            nn.ReLU()
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, args[0], kernel_size=3, stride=2, )
        )
        self.pool = nn.MaxPool2d(2) 
        self.down1 = ConvDown(args[0], args[1])
        self.res1 = ResidualDown(args[0], args[1])
        self.down2 = ConvDown(args[1], args[2])
        self.res2 = ResidualDown(args[1], args[2])
        self.bridge = ResidualBridge(args[2], args[3])

    def forward(self, x):
        xpad = F.pad(x, [0,1,0,1], mode='replicate')
        x1 = self.input_layer(x) 
        x1b = self.pool(x1) + self.input_skip(xpad)
        x2 = self.down1(x1b)
        x2b = self.pool(x2) + self.res1(x1b)
        x3 = self.down2(x2b)
        x3b = self.pool(x3) + self.res2(x2b)
        x4 = self.bridge(x3b) 
        
        embeddings = [x1, x2, x3, x3b, x4 ]
        return embeddings

class ResUnetDecoder(nn.Module):
    def __init__(self, dropout = 0.3, filters=None):
        super(ResUnetDecoder, self).__init__()

        args = filters
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.res_bridge = ResidualUp(args[2],args[3])
        self.up1 = ConvUp(args[3], args[2])
        self.res3 = ResidualUp(args[3], args[2])
        self.up2 = ConvUp(args[2], args[1])
        self.res4 = ResidualUp(args[2], args[1])

      ## last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(args[1] + args[1] //2, args[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(args[0], args[0], kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(args[0], 1, kernel_size=1, padding='same'),
            # nn.Sigmoid(), # done by BCEWithLogitsLoss
        )


    def forward(self, x1, x2, x3, x3b, x4):
        
        x4b = self.upsample(x4) + self.res_bridge(x3b)
        x5 = torch.cat([x4b, x3], dim=1)
        x6 = self.up1(x5) + self.res3(x4b)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up2(x7) + self.res4(x6)
        x9 = torch.cat([x8, x1], dim=1)
        output = self.output_layer(x9)

        return output




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
            #nn.Sigmoid(), # done by BCEWithLogitsLoss
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
        x1 = embedding[0]
        x2 = embedding[1]
        x3 = embedding[2]
        x3b = embedding[3]
        x4 = embedding[4]
        x = self.decoder(x1, x2, x3, x3b, x4)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        #self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('train_loss', loss, global_step=self.global_step)

        y_true = y.detach()
        #y_pred = y_pred.argmax(-1).detach()
        return {'loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def training_epoch_end(self, outputs):
        y_true_list = list()
        y_pred_list = list()
        #add here accuracy
        
        # tried to log and return loss
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()  
        # logging using tensorboard logger
        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        #epoch_dictionary={'loss': avg_loss}
        #return epoch_dictionary

    def validation_step(self, val_batch, batch_idx):     
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        self.logger.experiment.add_scalar('val_loss', loss, global_step=self.global_step)
        y_true = y.detach()
        return {'val_loss' : loss}
        #pass
        
    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  
        avg_loss=float(avg_loss)
        dict_={'val_loss' : avg_loss}
        print(dict_)
        return dict_
        #pass

    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.decoder.parameters(),lr = self.lr)
        
        return optimizer

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self.ce(y_pred, y)
        #self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalar('test_loss', loss, global_step=self.global_step)
        y_true = y.detach()
        return {'test_loss' : loss, 'y_pred' : y_pred, 'y_true' : y}

    def test_step_end(self, outputs):
        return outputs

    def test_epoch_end(self, outputs):

        y_true_list = list()
        y_pred_list = list()

        for item in outputs:
            y_true_list.append(item['y_true'])
            y_pred_list.append(item['y_pred'])
        
        print(len(y_pred_list))

        #acc = accuracy_score(torch.cat(y_true_list).cpu(),torch.cat(y_pred_list).cpu())
        #Overall accuracy
        #self.log('OA',round(acc,2), logger=True)

    def evaluate_performance(y_true, y_pred):

        pred_areas=[]
        true_areas=[]
        for i in range(len(y_true)):
            prediction=y_pred[i, :,:]
            berg_samples=prediction[y_true[i]==1]
            background_samples=prediction[y_true[i]==0]
            TP= sum(berg_samples)
            FP= sum(background_samples)
            FN= len(berg_samples)-sum(berg_samples)
            TN= len(background_samples)-sum(background_samples)
            
            pred_areas.append(sum(y_pred[i, :,:].flatten()))
            true_areas.append(sum(y_true[i].flatten()))

        true_areas=np.array(true_areas)
        pred_areas=np.array(pred_areas)

        flat_pred=y_pred.flatten()
        val_arr=np.concatenate(y_true, axis=0 )
        flat_true=val_arr.flatten()

        berg_samples=flat_pred[flat_true==1]
        background_samples=flat_pred[flat_true==0]

        TP= sum(berg_samples)
        FP= sum(background_samples)
        FN= len(berg_samples)-sum(berg_samples)
        TN= len(background_samples)-sum(background_samples)
        
        # dice
        dice=2*TP/(2*TP+FP+FN)

        print('overall accuracy')
        print((TP+TN)/(TP+TN+FP+FN)*100)
        print('false pos rate')
        print(FP/(TN+FP)*100)
        print('false neg rate')
        print(FN/(TP+FN)*100)
        print('area deviations')
        print((pred_areas-true_areas)/true_areas*100)
        print('abs mean error in area')
        print(np.mean(abs((pred_areas-true_areas)/true_areas))*100)
        print('area bias')
        print(np.mean((pred_areas-true_areas)/true_areas)*100)
        print('f1')
        print(dice)
  
