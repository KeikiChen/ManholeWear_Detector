import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.optim import lr_scheduler

# Dataset
class manholeDataset(Dataset):
    def __init__(self, csv_file, root_dir1, root_dir2, transform=None):
        self.depths_frame = pd.read_csv(csv_file)
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform

    def __len__(self):
        return len(self.depths_frame)

    ##def use_canny(self, image):
        image =np.array(image)
        edges =cv2.Canny(image, 50, 100)
        edges_to_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_to_rgb)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name1 = os.path.join(self.root_dir1, self.depths_frame.iloc[idx, 0])
        img_name2 = os.path.join(self.root_dir2, self.depths_frame.iloc[idx, 0])
        image1 = Image.open(img_name1)
        image2 = Image.open(img_name2)
        #use canny
        ##image1 =self.use_canny(image1)
        ##image2 =self.use_canny(image2)

        depth = self.depths_frame.iloc[idx, 1]
        depth = np.array([depth])
        depth = depth.astype('float').reshape(-1)
        depth = torch.from_numpy(depth).float()
        sample = {'image1': image1, 'image2': image2, 'depth': depth}

        if self.transform:
            sample['image1'] = self.transform(sample['image1'])
            sample['image2'] = self.transform(sample['image2'])

        return sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

#myresnetblock
class RESNETBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, pad=1, dilation=1):
        super(RESNETBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


#cost-volume
def cost_volume(feature_left,feature_right,max_disp):
    ##print("feature_left size:", feature_left.size())
    ##print("feature_right size:", feature_right.size())
    batch_size,channels,height,width = feature_left.size()
    cost_volume = torch.zeros(batch_size,channels*2,max_disp,height,width).to(feature_left.device)

    for i in range(max_disp):
        if i> 0:
            cost_volume[:,:channels,i,:,i:] =feature_left[:,:,:,i:]
            cost_volume[:,channels:,i,:,i:] =feature_right[:,:,:,:-i]
        else:
            cost_volume[:, :channels, i, :, :] = feature_left
            cost_volume[:, channels:, i, :, :] = feature_right
        
    return cost_volume

            
#模型

'''
class MYmodel(nn.Module):
    def __init__(self):
        super(MYmodel, self).__init__()
        self.resnet1 = resnet18(pretrained=True)
        self.resnet2 = resnet18(pretrained=True)
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        self.fc = nn.Linear(512 * 2, 1)

       

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)

        return x 
'''


'''
class ModifiedResNet(nn.Module):
    def __init__(self,orgininal_model):
        super(ModifiedResNet,self).__init__()
        self.features = nn.Sequential(*list(orgininal_model.children())[:-2])

    def forward(self,x):
        x = self.features(x)
        return x
    
class MYmodel(nn.Module):
    def __init__(self):
        super(MYmodel, self).__init__()
        resnet1 = resnet18(pretrained=True)
        resnet2 = resnet18(pretrained=True)
        self.resnet1 = ModifiedResNet(resnet1)
        self.resnet2 = ModifiedResNet(resnet2)
        self.conv1 = nn.Conv3d(512 * 2, 512,kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv2 = nn.Conv3d(512, 256,kernel_size=3,stride=1,padding=2,dilation=2)
        self.conv3 = nn.Conv3d(256, 128,kernel_size=3,stride=1,padding=3,dilation=3)
        self.global_avg_pool=nn.AdaptiveAvgPool3d((1,1,1))
        self.fc =nn.Linear(128,1)

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        cost = cost_volume(x1,x2)
        x =F.relu(self.conv1(cost))
        x =F.relu(self.conv2(x))
        x =F.relu(self.conv3(x))
        x =self.global_avg_pool(x)
        x=torch.flatten(x,1)
        depth =self.fc(x)

        return depth 
'''





class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 128
        self.conv1=nn.Sequential(convbn(3,32,3,2,1,1),
                                 nn.ReLU(inplace=True),
                                 convbn(32,64,3,2,1,1),
                                 nn.ReLU(inplace=True),
                                 convbn(64,128,3,2,1,1),
                                 nn.ReLU(inplace=True)
                                 )
        
        self.conv2 =self._make_layer(RESNETBlock,256,2,2,1,1)
        self.conv3 =self._make_layer(RESNETBlock,512,2,1,1,1)
        self.conv4 =self._make_layer(RESNETBlock,512,2,1,1,2)
        self.conv5 =self._make_layer(RESNETBlock,512,2,1,1,4)


       
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output_raw = self.conv3(output)
        output =self.conv4(output_raw)
        output_atrous =self.conv5(output)

        output_feature = torch.cat((output_raw,output_atrous),1)

        return output_feature


class MYmodel(nn.Module):
    def __init__(self, maxdisp, feature_extractor):
        super(MYmodel, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extractor = feature_extractor

        # 定义三个连续的 CNN 层
        self.conv1 = nn.Conv3d(512*2*2, 512*2, 3, padding=1)
        self.conv2 = nn.Conv3d(512*2, 512, 3, padding=1)
        self.conv3 = nn.Conv3d(512, 256, 3, padding=1)

        self.global_avg_pool=nn.AdaptiveAvgPool3d((1,1,1))

        self.fc =nn.Linear(256,1)

    def forward(self, left, right):
        left_feature = self.feature_extractor(left)
        right_feature = self.feature_extractor(right)

        cost = cost_volume(left_feature,right_feature,self.maxdisp)

        x =F.relu(self.conv1(cost))
        x =F.relu(self.conv2(x))
        x =F.relu(self.conv3(x))
        x =self.global_avg_pool(x)
        x=torch.flatten(x,1)
        depth =self.fc(x)

        return depth



'''
class MYmodel(nn.Module):
    def __init__(self):
        super(MYmodel, self).__init__()
        self.resnet1 = resnet18(pretrained=True)
        self.resnet2 = resnet18(pretrained=True)
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        self.conv1 = nn.Conv2d(1024, 512,kernel_size=1)
        self.conv2 = nn.Conv2d(512, 256,kernel_size=1)
        self.conv3 = nn.Conv2d(256, 1,kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

       

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x = torch.cat((x1, x2), dim=1)
        x=x.view(x.size(0),x.size(1),1,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x= self.global_avg_pool(x)
        x= torch.flatten(x,1)

        return x
'''


def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_ard = 0.0
    total_delta = 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs1, inputs2, labels = data['image1'].to(device), data['image2'].to(device), data['depth'].to(device)
            labels = labels.float()
            outputs = model(inputs1, inputs2)

            # MSE
            mse = round(((outputs - labels) ** 2).mean().item(),4)
            total_mse += mse

            # ARD (REL)
            ard = round((torch.abs(outputs - labels) / labels).mean().item(),4)
            total_ard += ard

            # δ
            ratio = outputs / labels
            delta = round(((ratio >= 0.9) & (ratio <= 1.1)).float().mean().item(),4)
            total_delta += delta

    avg_mse = total_mse / len(dataloader)
    avg_ard = total_ard / len(dataloader)
    avg_delta = total_delta / len(dataloader)

    return avg_mse, avg_ard, avg_delta

def save_checkpoint(model, filepath):
    torch.save(model.state_dict(),filepath)
    
def save_results(results, filename):
    with open(filename,'w') as file:
        json.dump(results, file)


class PreserveAspectRatioResize:
    def __init__(self,target_size):
        self.target_size =target_size

    def __call__(self, img):
        img_aspect = img.width / img.height
        target_aspect = self.target_size[0] / self.target_size[1]

    # Compare aspect ratios
        if img_aspect > target_aspect:
            new_width = int(self.target_size[1] * img_aspect)
            return transforms.Resize((new_width, self.target_size[1]))(img)
        else:
            new_height = int(self.target_size[0] / img_aspect)
            return transforms.Resize((self.target_size[0], new_height))(img)

def train_model():
# 预处理
    transform = transforms.Compose([
        PreserveAspectRatioResize(target_size=(224, 224)),
        transforms.CenterCrop((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.500, 0.495, 0.501], std=[0.295, 0.294, 0.297]),
    ])

# 创建数据集

    train_dataset = manholeDataset(csv_file='C:\\Users\\kk\\Desktop\\data\\image_shape\\train_data\\teacher.csv', root_dir1='C:\\Users\\kk\\Desktop\\data\\image_shape\\train_data\\masked_L', root_dir2='C:\\Users\\kk\\Desktop\\data\\image_shape\\train_data\\masked_R', transform=transform)
    test_dataset = manholeDataset(csv_file='C:\\Users\\kk\\Desktop\\data\\image_shape\\test_data\\teacher.csv', root_dir1='C:\\Users\\kk\\Desktop\\data\\image_shape\\test_data\\masked_L', root_dir2='C:\\Users\\kk\\Desktop\\data\\image_shape\\test_data\\masked_R', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    maxdisp = 32
    feature_extractor =feature_extraction()
    model = MYmodel(maxdisp,feature_extractor).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

    Scheduler = lr_scheduler.StepLR(optimizer, step_size=30,gamma=0.1)

    results = {
        'epochs': [],
        'train_loss': [],
        "mse": [],
        'ard': [],
        'delta': []
    }



    for epoch in range(10):  
        running_loss = 0.0
        model.train()
        for i, data in enumerate(tqdm(train_dataloader), 0):
            inputs1, inputs2, labels = data['image1'].to(device), data['image2'].to(device), data['depth'].to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        Scheduler.step()

        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{10}], Loss: {epoch_loss:.4f}')

        mse, ard, delta = evaluate(model, test_dataloader, device)
        results['epochs'].append(epoch + 1)
        results['train_loss'].append(epoch_loss)
        results['mse'].append(mse)
        results['ard'].append(ard)
        results['delta'].append(delta)
    

    save_checkpoint(model, 'C:\\Users\\kk\\Desktop\\data\\model\\shape\\model_weights_self256_shape.pth')
   
    save_results(results, 'C:\\Users\\kk\\Desktop\\data\\results\\shape\\model_results_self256_shape.json')
    
    print('Finished Training')

if __name__ == '__main__':
    train_model()
