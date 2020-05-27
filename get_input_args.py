                    
# PROGRAMMER: Kaung Myat Tun

import argparse


def get_input_args():
    """
   
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("--data_dir", type=str ,default='flowers' ,help="direction of saving Data_Dir ")
    parser.add_argument("--file_path", type=str ,default=None ,help="direction of iamge File ")
    parser.add_argument("--save_dir", type=str ,default='./' ,help="direction of saving Check point ")
    parser.add_argument("--arch",  type=str ,default='vgg16' ,help="Which Architecture ")
    parser.add_argument("--learning_rate",  type=float,default=0.002,help="Define your learning rate")
    parser.add_argument("--hidden_units",  type=int,default=4096 ,help="Define your hidden Layer")
    parser.add_argument("--epochs",  type=int,default=5 ,help="Define your epochs")
    parser.add_argument("--top_k",  type=int,default=5 ,help="Define your top_k")    
    parser.add_argument("--category_names",  type=str,default="cat_to_name.json" ,help="Define your category_names")
    parser.add_argument("--gpu",  type=bool,default=True ,help="Define your GPU or CPU") 
    return parser.parse_args()
