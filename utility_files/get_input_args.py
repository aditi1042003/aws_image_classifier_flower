# Imports python modules
import argparse
def get_input_args():
    """
    Retrieves and parses the 9 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 9 command line arguments. If 
    the user fails to provide some or all of the 9 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg16'
      3. Text File with Dog Names as --flowrname with default value 'cat_to_name.json'
      ...
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    #1->data directory
    parser.add_argument('--data_dir', type = str, default = 'flowers/',  help = 'path to the folder of flower images')
    #2->save path
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Enables user to choose directory for saving')
    #3->model
    parser.add_argument('--arch', type = str, default = 'vgg16',  help = 'defining architechture of the model to be used default is vgg16')
    #4->name file of flower json
    parser.add_argument('--flowername', type = str, default = 'cat_to_name.json',  help = 'text file containing the name of dogs')
    #5->learning_rate
    parser.add_argument('--learning_rate', type=float, default=0.001, help='sets the rate at which the model does its learning')
    #6->no. of hidden units
    parser.add_argument('--hidden_layer', type=int, default=1024, help='Number of hidden units for the hidden layer')
    #->7 device for processing
    parser.add_argument('--device', default='cuda',  help='Determines where to run model: CPU vs. GPU')
    #8-> number of epochs
    parser.add_argument('--epochs', type=int, default=5, help='Determines number of epochs to train the model')
    #9->dropout rate
    parser.add_argument('--dropout', type=float, default=0.4, help='Determines probability rate for dropouts')

    return parser.parse_args()
