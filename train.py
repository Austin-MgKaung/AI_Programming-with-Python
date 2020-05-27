# Import Pytorch Packages

import torch  
from torch import nn,optim
from torchvision import datasets,models,transforms
from PIL import Image
from collections import OrderedDict

from save_and_load import save_checkpoint
# import get_input_args 
from get_input_args import get_input_args

in_arg=get_input_args()



def train_test(mode=0,model=None,criterion=None,optimizer=None,loader=None,device='cpu',gpu=True) :
    
    accuracy=0
    running_loss=0
    # Check GPU avilable if available move model to GPU
    if torch.cuda.is_available() and gpu:
        model.cuda()
        print("GPU mode")
    else :
        model.cpu()
        print("CPU mode")
        
    #Check mode to dropout
    if mode == 0 :
        model.train()
    else :
        model.eval()
        
    for images,labels in loader :
        
        # Run model on gpu if available
        images,labels=images.to(device),labels.to(device)
        
        #clear gradients 
        optimizer.zero_grad()
        
        #forward pass
        output=model.forward(images)
        
        #calculate Loss
        loss=criterion(output,labels)
        
        #if training mode else skip backward propogation
        if mode == 0 :
        
            #Backward Propogation
            loss.backward()
        
            #Update Weight 
            optimizer.step()
        
        #Keep track loss 
        running_loss +=loss.item()
        
        #Calculate Accuracy
        if mode == 0:
            accuracy=0
            
        else: 
            ps=torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    return (running_loss/len(loader)),(accuracy/len(loader))*100
        
def main(): 
    # Directory of Image Data 
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose( [ transforms.Resize(255),
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(),
                                            transforms.Normalize( (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) ) ] )
    
    
    ## transforms for the training set using Data Argumentation 
    train_transforms = transforms.Compose( [ transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize( (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) ) ] )

    ## transforms for validation set
    valid_transforms = data_transforms
    
    ## transforms for testing set
    test_transforms = data_transforms
        
    # Load the Datasets with ImageFolder 
    image_datasets =datasets.ImageFolder(data_dir,transform=data_transforms)

    # Load training Dataset
    train_datasets =datasets.ImageFolder(train_dir,transform=train_transforms)
    
    # Load validation Dataset
    valid_datasets =datasets.ImageFolder(valid_dir,transform=valid_transforms)
    
    # Load test Dataset
    test_datasets =datasets.ImageFolder(test_dir,transform=test_transforms)

    # Dataloader and batch size 
    dataloaders =torch.utils.data.DataLoader(image_datasets,batch_size=64,shuffle=True)

    #Trainloader
    trainloaders =torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)

    #Validloader
    validloaders =torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=True)

    #Testloader
    testloaders =torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True)

    #Use two Pretrained model
    models_dict={}
    vgg13=models.vgg13(pretrained=True)
    vgg16 =models.vgg16(pretrained=True)
    print("OK1")
    models_dict={'vgg13' :vgg13, 'vgg16' : vgg16 }
    print("OK2")
    model=models_dict[in_arg.arch]
    print("OK3")
    #Frozen the parameters 
    #They can't  get Updated during Training 
    #Turning off gradient 
    for param in model.parameters():
        param.requires_grad=False

    # Define our new Classifier using only 1 hidden Layer
    
    classifier = nn.Sequential(OrderedDict ( [ 
                                ('fc1'   ,   nn.Linear(25088,in_arg.hidden_units) ),
                                ('relu'  ,   nn.ReLU()             ),
                                ('dou'   ,   nn.Dropout(p=0.2)     ),
                                ('fc2'   ,   nn.Linear(in_arg.hidden_units,102)   ),
                                ('output',  nn.LogSoftmax(dim=1)   ) ] ) )
    
    ## Update Classifier and check model again 
    model.classifier=classifier

    # define loss Since we use logsoftmax as output we use  negative log likelihood loss

    criterion = nn.NLLLoss()
    
    # define optimizer to update the weights with gradients
    
    optimizer = optim.Adam(model.classifier.parameters(),lr=in_arg.learning_rate)
    
    
    epochs=in_arg.epochs

    # Use GPU if avaliable 
    device=torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")

    for epoch in range(epochs):
    
        # Train model
        train_loss,train_accuracy = train_test(0,model,criterion,optimizer,trainloaders,device,in_arg.gpu)
    
        # Validate model 
        with torch.no_grad():
            valid_loss,valid_accuracy =train_test(1,model,criterion,optimizer,validloaders,device,in_arg.gpu)
        
        # print description 
        print("Epoch  :{}/{} \n ".format(epoch+1,epochs))
        print("Traning Loss :{} \n ".format(train_loss))
        print("Validation Loss :{} \n ".format(valid_loss))
        print("Validation Accuracy :{} \n ".format(valid_accuracy))
        
    model.class_to_idx = train_datasets.class_to_idx
        
    save_checkpoint(model,optimizer,in_arg)
        
        
        
if __name__== "__main__" :
    main()
    
    


