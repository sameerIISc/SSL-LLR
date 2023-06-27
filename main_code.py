import sys

sys.path.insert(0, 'Train/')
sys.path.insert(0, 'Test/')
sys.path.insert(0, 'pyramids/')
sys.path.insert(0, 'qa_codes/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'loss/ssim_lp/')
sys.path.insert(0, 'loss/')

from compute_lp import compute_lp
from compute_pyr import compute_pyr

from Train_LowpassCNN_lbld import Train_LowpassCNN_lbld
from Train_BandpassCNN_lbld import Train_BandpassCNN_lbld
from compute_lp_unlbld import compute_lp_unlbld
from test_LowpassCNN_pt_unlbld import test_LowpassCNN_pt_unlbld

from compute_lp_ensemble_data import compute_lp_ensemble_data
from Train_model_ensemble import Train_model_ensemble
from test_ensemble_on_unlbld import test_ensemble_unlbld

from Train_qa_mv import Train_qa_mv
from test_qa import test_qa

from Train_sublp_with_pseudo_labels import Train_sublp_with_pseudo_labels
from test_ssl import test_ssl

import torch

dataset = 'sony'
frac_lbld = 5
splits = 10

'''
In the following code short refers to low-light images and long refers to ground-truth images.
Keep the lbld images in '../dataset_name/str(frac_lbld)%_data/split + str(split)/train_lbld/short(long)/
Keep the unlbld images in '../dataset_name/str(frac_lbld)%_data/split + str(split)/train_unlbld/short/
Keep the all training low-light images in '../dataset_name/full_data/train/short/
Keep the test low-light images in '../dataset_name/full_data/test/short/

We have followed the naming of images according to the SID and LOL dataset. 

Number of scales in pyramid are 5 for images of resolution 832x1248. For different size images scale 
the number of levels accordingly. 

'''

if dataset == 'lol':
    num_levels = 4
else:
    num_levels = 5

'''
For around 90 images in the labelled data, 901 epochs are used. Scale the number 
of epochs and lr_decay accordingly. Following parameters are used for training the 
lowpassCNN and bandpassCNN of the SMSNet on the lbld data and also to train the ensemble.
'''

if frac_lbld == 5:
    if dataset == 'lol':
        train_params = {'num_epochs': 4001,
                      'lr_decay': [2000, 3000],
                      'lr': 1e-3,
                      'batch_size': 16,
                      }
    else:
        train_params = {'num_epochs': 901,
                      'lr_decay': [500, 700],
                      'lr': 1e-3,
                      'batch_size': 16,
                      }

elif frac_lbld == 10:
   if dataset == 'lol':
       train_params = {'num_epochs': 2001,
                     'lr_decay': [1000, 1500],
                     'lr': 1e-3,
                     'batch_size': 16,
                     }
   else:
       train_params = {'num_epochs': 451,
                     'lr_decay': [250, 350],
                     'lr': 1e-3,
                     'batch_size': 16,
                     }

train_params['batch_size_qa']=16
train_params['lr_qa'] = 1e-4
train_params['num_epochs_qa'] = 11 
train_params['temperature_qa'] = 0.07

#### Scale the following according to full dataset size
if dataset=='lol':
    train_params['num_epochs_full'] = 451
    train_params['lr_decay_full'] = [250, 350]
else:
    train_params['num_epochs_full'] = 91
    train_params['lr_decay_full'] = [50, 70]
    
full_data_dir = '../' + dataset + '/full_data/'
    
compute_lp(full_data_dir + 'train/short/', num_levels)
compute_lp(full_data_dir + 'test/short/', num_levels)

'''
The implementation has some matlab codes also. First run just the 1st step in the 
following code by commenting out the further steps. Then run the first step of the 
main_code.m and so on.
'''

################# 1st step ######################
for num_split in range(1,splits+1):
    
    print('\n \n ######### Running split number ' + str(num_split) + ' ############')

    io_dir = '../' + dataset + '/' + str(frac_lbld) + '%_data/split' + str(num_split) + '/'
         
    print('\n ******** generating subs for lablelled data ************ \n') 
    compute_pyr(dataset, 'long', io_dir, num_levels)
    compute_pyr(dataset, 'short', io_dir, num_levels)
    
    print('\n ******** Training on lablelled data ************ \n') 
    Train_LowpassCNN_lbld(dataset, io_dir, train_params, num_levels)
    Train_BandpassCNN_lbld(dataset, io_dir, train_params, num_levels)
    compute_lp_unlbld(io_dir, num_levels)
    test_LowpassCNN_pt_unlbld(dataset, io_dir)

### Please run the the 1st step the main_code.m in matlab
################# 2nd step ######################

# for num_split in range(1,splits+1)

    # print('\n******** Creating subs of l2s data ************\n') 
    # compute_lp_ensemble_data(dataset, io_dir, num_levels)
    # print('\n******** Training l2s models ************\n') 
    # Train_model_ensemble(dataset, io_dir, train_params, num_levels)        
    # print('\n \n ******** Testing on unlabelled data ********** \n')
    # test_ensemble_unlbld(io_dir)
    # print('\n \n ******** Training the qa model **********\n')
    # Train_qa_mv(dataset, io_dir, train_params)
    # print(' \n \n ******** Testing contrastive **********\n')
    # test_qa(io_dir)   

#### Please run the 2nd step in main_code.m
################# 3rd step #######################

# for num_split in range(1,splits+1)
    
#     print('\n ******** Training with pseudo labels **********\n')
#     Train_sublp_with_pseudo_labels(dataset, train_params, io_dir, full_data_dir)
#     print('\n******** Testing with pseudo labels **********\n')
#     test_ssl(io_dir, full_data_dir, num_levels)