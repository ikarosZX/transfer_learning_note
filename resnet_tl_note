import torch                
import torch.nn as nn
import torchvision.models as models


'''if you don't have the pretrained model, then use Model_1()'''
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        '''[resnet18,resnet34,resnet50,resnet101,resnet152]'''
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False 
 
        #self.model.fc = nn.Linear(self.model.fc.in_features,2,bias=False) 
        #print(self.model.layer4) 
    def forward(self, x):
        x = self.model(x)
        return x

'''if you've already got the pretrained model, use Model_2()'''
class Model_2(nn.Module):
    def __init__(self,model_path):
        super().__init__()
        self.model_path = model_path
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path))
        for param in self.model.parameters():
            param.requires_grad = False 
 
        self.model.fc = nn.Linear(self.model.fc.in_features,2,bias=False)
        '''the parameters in the () can be printed'''
        #print(self.model)    #the whole structure of the network
        #print(self.model.fc) #the full connect layer
        #print(self.model.avgpool)  
        #print(self.model.layer4)   
        #print(self.model.layer3)
        #print(self.model.layer2)
        #print(self.model.layer1)
        #print(self.model.layer1[0])
        #print(self.model.layer1[1])
        #print(self.model.layer1[2])
        #print(self.model.layer4[2].conv1)
        #print(self.model.layer4[2].bn1)
        #print(self.model.layer4[2].conv2)
        #print(self.model.layer4[2].bn2)
        #print(self.model.layer4[2].conv3)
        #print(self.model.layer4[2].bn3)
        #print(self.model.layer4[2].relu)
        
        '''modify the network'''
        #self.model.layer4[2].conv1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #print(self.model.layer4[2].conv1)
    def forward(self, x):
        x = self.model(x)
        return x

 
if __name__ == '__main__':
    model_path = './pths/resnet50-19c8e357.pth' 
    model = Model_2(model_path)
    #print(model)
