# Import Pytorch Packages

import torch 
import os, random
from torch import nn,optim
from torchvision import datasets,models,transforms
from PIL import Image
from collections import OrderedDict

from save_and_load import load_checkpoint
# import get_input_args 
from get_input_args import get_input_args
import json


in_arg=get_input_args()

import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
   # Open image file 
    img=Image.open(image)
    
    # find the shorter size and set to 256 pixels
    width,height=img.size

    if width > height :
        img=img.resize((width,256)) 
    else :
        img =img.resize((256,height))
        
   #Crop the image as 224 x 224
    width,height=img.size    
    left =(width -224)/2
    upper=(height -224)/2
    right =(width +224)/2
    lower=(height +224)/2
    img=img.crop((left,upper,right,lower))
    
    #Changing pillow image to np array
    img=np.array(img)
    img=img/[255,255,255]
    
    #Normalize the image 
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    img=(img - mean)/std
    
    #To match with pytorch 
    img = np.transpose(img,(2,0,1))
    
    return img 
    
def predict(image_path, model, topk=5,gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    mapped_lst=[]
    device=torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    # Check GPU avilable if available move model to GPU
    if torch.cuda.is_available() and gpu :
        model.cuda()
        print("GPU mode")
    else :
        model.cpu()
        print("CPU mode")
        
        
    # predict image (like testing data set)
    with torch.no_grad():
        model.eval()
        img=process_image(image_path) # rreturn numpy array but match with torch
        img=torch.from_numpy(img).unsqueeze(0).to(device)
        
        output=model.forward(img.float())
        ps=torch.exp(output)
        top_p, top_class = ps.topk(topk)
        
        # change back to cpu and numpy
        top_p=top_p.detach().cpu().numpy().tolist()[0]
        top_class=top_class.detach().cpu().numpy().tolist()[0]
        
        # invert dictionary 
        idx_to_class ={value : key for (key, value) in model.class_to_idx.items()}
        
        #find the class in class to idx and mapped it
        for item in top_class :
            mapped_lst.append(idx_to_class[item])
            
        return top_p,mapped_lst
        
def main():
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #device=torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model=load_checkpoint(in_arg.save_dir+'checkpoint.pth')
    if in_arg.file_path is None :
        randclass=random.randint(1,102)
        randir=test_dir+"/"+str(randclass)+"/"
        image=random.choice(os.listdir(randir))
        image_path =randir+image 
    prob,classes=predict(image_path, model,in_arg.top_k,in_arg.gpu)
    print(prob)
    print(classes)
    print([cat_to_name[item] for item in classes])
    
    
    
if __name__ == "__main__" :
    main()