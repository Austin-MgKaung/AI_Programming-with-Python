# Import Pytorch Packages

import torch  
from torch import nn,optim
from torchvision import datasets,models,transforms
from get_input_args import get_input_args

in_arg=get_input_args()

def save_checkpoint(model,optimizer,args) :
    #Create Checkpoint 
    checkpoint = {  'arch'            : args.arch,
                    'model'           : model,
                    'input_size'      : 25088,
                    'output_size'     : 102,
                    'opti_state_dict' : optimizer.state_dict(),
                    'learning_rate'   :args.learning_rate,
                    'state_dict'      : model.state_dict(),
                    'classifier'      : model.classifier,
                    'epochs'          : args.epochs,
                    'class_to_idx'    : model.class_to_idx
             }
    #Save Checkpoint 
    torch.save(checkpoint,'checkpoint.pth')
    
    
def load_checkpoint(filepath) :
    checkpoint = torch.load(filepath)
    model=checkpoint["model"]
    model.classifier=checkpoint["classifier"]
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(),checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['opti_state_dict'])
    epochs = checkpoint['epochs']
    model.class_to_idx=checkpoint['class_to_idx']
    
    return model
    