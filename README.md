## aws_image_classifier_flower
 
# Part 1 - Development Notebook

## completed:

### main - files

+ html_file:  Final_AIPND_image_classifier_project.html
+ ipynb_notebook: Final_AIPND_image_classifier_project.ipynb 
+ notebook_link: https://colab.research.google.com/drive/1sDwGd_Cq5XFbURrz5XrZN2Lc9YSEz2lJ?usp=sharing

used colab notebook to complete and train project due to Udacity workspace error and exponential training times

->  train time on cloab : 30 min average

+ Package Imports 
+ Training data augmentation
+ Data normalization
+ Data batching (batches of 32)
+ Data loading
+ Pretrained Network (vgg16)
+ Feedforward Classifier (1024 nodes hidden layer)
+ Training the network 
+ Testing Accuracy
+ Validation Loss and Accuracy
+ Loading checkpoints
+ Saving the model
+ Image Processing
+ Class Prediction
+ Sanity Checking with matplotlib

# output
![output image]()


### saved file 
+   checkpoint.pth

## Resluts:
+   Validation Accuracy: 83.3%
+   Test Accuracy: 80.6%

# Part 2 - Command Line Application

## completed:
used udacity workspace for part-2
upload 'checkpoint.pth'  created in part-1 to workspace to create CLI application
### main - files
+   train.py
+   predict.py
+   cat_to_name.json --contain names of flower mapped to idx
+   checkpoint.pth --saved model

### utility files 
to use utiliity file add "utility_files/" to path of file imported

+   Network.py --defines network architechture
+   get_input_args.py --to get input arguments for train.py file
+   train_util.py --to train model with given input args
+   test_util.py --to test the accuracy of trained model
+   load_model_util.py --to load save model in checkpoint.pth
+   predict_util.py  --helps to pre_process image and feed into model to get results in form of probs,classes


# terminal output

Output of running classifier in CLI stored:
+   train.txt
+   predict.txt