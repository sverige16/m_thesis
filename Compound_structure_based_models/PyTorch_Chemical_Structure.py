#!/usr/bin/env python
# coding: utf-8

# In[15]:


#get_ipython().system('pip install rdkit-pypi')


# In[1]:


# import statements
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torchinfo  

# print(sklearn.__version__)
# In[2]:
start = time.time()

# get the dictionary for compound_id-SMILES pair 
compound_smiles_dictionary = pickle.load(open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_image_based_model/dictionary2.pickle", "rb"))


# In[3]:


# print(list(compound_smiles_dictionary.items())[:10])


# In[4]:


# 10 selected MoAs 
moas_to_use = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 
        'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']


# In[5]:


# read the data 
all_data = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_image_based_model/all_data.csv')


# In[6]:


# Limiting the data to the first 2000 rows
testing = False
if testing == True:
    all_data = all_data[0:1000]


# In[7]:


# drop random previous index in table above
all_data.drop(all_data.columns[[0,11, 12]], axis=1, inplace=True) # Check that reading the data worked 
#all_data.head(10)


# In[8]:


dictionary = {'ATPase inhibitor': 7, 'Aurora kinase inhibitor': 0,
 'HDAC inhibitor': 4, 'HSP inhibitor': 9, 'JAK inhibitor': 2, 'PARP inhibitor': 6,
 'protein synthesis inhibitor': 3, 'retinoid receptor agonist': 8,
 'topoisomerase inhibitor': 5, 'tubulin polymerization inhibitor': 1}


# In[9]:


num_classes = len(dictionary) 
#num_classes


# In[10]:


# change moa to classes 
all_data['classes'] = None
for i in range(all_data.shape[0]):
    all_data.iloc[i, 10] = dictionary[all_data.iloc[i, 9]]


# In[11]:


# get the compound-MoA pair 
compound_moa = all_data[['compound','moa']].drop_duplicates()


# In[12]:


# creating tensor from all_data.df
target = torch.tensor(all_data['classes'].values.astype(np.int64))
target, target.shape


# In[13]:

# For each row, take the index of the tar- get label
# (which coincides with the score in our case) and use it as the column index to set the value 1.0.” 
target_onehot = torch.zeros(target.shape[0], num_classes)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)


# In[14]:


# split dataset into test and training/validation sets (10-90 split)
compound_train_valid, compound_test, moa_train_valid, moa_test = train_test_split(
  compound_moa.compound, compound_moa.moa, test_size = 0.10, stratify = compound_moa.moa, 
  shuffle = True, random_state = 1)


# In[15]:


# Split data set into training and validation sets (1 to 9)
# Same as above, but does split of only training data into training and validation data (in order to take advantage of stratification parameter)
compound_train, compound_valid, moa_train, moa_valid = train_test_split(
  compound_train_valid, moa_train_valid, test_size = 10/90, stratify = moa_train_valid,
  shuffle = True, random_state = 62757)


# In[16]:


# get the train, valid and test set by for every compound in data set, if it is in train, valid or test, return all info from all_data in new df.
train = all_data[all_data['compound'].isin(compound_train)]
valid = all_data[all_data['compound'].isin(compound_valid)]
test  = all_data[all_data['compound'].isin(compound_test)]
''' Explanation
reset_index = create new index starting from zero 
drop: drop previous index.
'''


# In[17]:


# create dictionary for parition
partition = {"train": [], "valid": [], "test": []}

# create lists with indexes in splits
tr_list = train.index.tolist()
va_list = valid.index.tolist()

# place index into correct bin
for index in all_data.index.tolist():
#for index in compound_moa.index.tolist():
    if index in tr_list:
        partition["train"] += [index]
    elif index in va_list:
        partition["valid"]   += [index]
    else:
        partition["test"]  += [index]


# In[18]:


# create dictionary for labels, using PyTorch
labels = {}
for index,compound in zip(all_data.index.tolist(), target_onehot):
    labels[index] = compound


# In[73]:


batch_size = 200 
# parameters
params = {'batch_size' : 200,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 
          
# shuffle isn't working

# Datasets
partition = partition
labels = labels

# maxepochs
max_epochs = 1500


# In[74]:


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )


# In[75]:


# A function changing SMILES to Morgan fingerprints 
def smiles_to_array(smiles):
    molecules = Chem.MolFromSmiles(smiles) 
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(molecules, 2)
    x_array = []
    arrays = np.zeros(0,)
    DataStructs.ConvertToNumpyArray(fingerprints, arrays)
    x_array.append(arrays)
    x_array = np.asarray(x_array)
    x_array = ((np.squeeze(x_array)).astype(int)) 
    x_array = torch.from_numpy(x_array)
    return x_array                  


# In[76]:


# split into training, validation and test data


# In[77]:


#nrow = all_data.iloc[4]
#nrow
#a = nrow['compound']


# In[78]:


#b = compound_smiles_dictionary[a]
#b


# In[79]:


# create Torch.dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, partition, labels, transform=None):
        self.compound_labels = labels    # the entire length of the correct classes that we are trying to predict
        # print(self.img_labels)
        self.list_ID = partition         # list of indexes that are a part of training, validation, tes sets
        self.transform = transform       # any transformations done

    def __len__(self):
        ''' The number of data points '''
        return len(self.compound_labels)      

    def __getitem__(self, idx):
        '''Retrieving the compound '''
        #print(idx)                                    # check to see if idx is monotonically growing starting from zero
        ID = self.list_ID[idx]                        # we extract an index in partition representing training/valid/test set
        nrow = all_data.iloc[ID]                                  # extract row from csv using index
        compound = compound_smiles_dictionary[nrow["compound"]]      # returns smiles by using compound as key
        compound_array = smiles_to_array(compound)
        #print(f' return from function: {compound}')
        #print(f' matrix: {compound_array}')
        label = self.compound_labels[ID]             # extract classification using index
        #print(f' label: {label}')
        #label = torch.tensor(label, dtype=torch.float)
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        #return image.float(), label.long()
        return compound_array.float(), label.float()


# ## Generators

# In[80]:


# Create a dataset with all indices and labels
everything = Dataset(partition["train"]+partition['valid']+partition['test'], labels)


# In[81]:


# generator for training data


# In[82]:


# generator: training
# create a subset with only train indices
training_set = torch.utils.data.Subset(everything, partition["train"])

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_set, **params)
# training_set = Dataset(partition["train"], labels)


# In[83]:


# training data loader
# Display image and label   # functional
train_features, train_labels = next(iter(training_generator))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")


# In[84]:


# Validation Set
# create a subset with only valid indices
valid_set = torch.utils.data.Subset(everything, partition["valid"])
    
# create generator that randomly takes indices from the validation set
valid_generator = torch.utils.data.DataLoader(valid_set, **params)


# In[85]:


# Test set
# create a subset with only test indices
test_set = torch.utils.data.Subset(everything, partition["test"])

# create generator that randomly takes indices from the test set
test_generator = torch.utils.data.DataLoader(test_set, **params)# generator for validation data


# In[86]:


# dir(torch.nn)


# In[87]:


# If applying class weights
apply_class_weights = True
if apply_class_weights:     # if we want to apply class weights
    counts = compound_moa.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    #print(counts)
    class_weights = []   # create list that will hold class weights
    for moa in compound_moa.moa.unique():       # for each moa   
        #print(moa)
        counts[moa]
        class_weights.append(counts[moa])  # add counts to class weights
    #print(len(class_weights))
    #print(class_weights)
    #print(type(class_weights))
    # class_weights = 1 / (class_weights / sum(class_weights)) # divide all class weights by total moas
    class_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device) # transform into tensor, put onto device
#print(class_weights)


# In[88]:


# Creating Architecture
units = 64
drop  = 0.89

seq_model = nn.Sequential(
    nn.Linear(2048, 64),
    nn.ReLU(),
    nn.Dropout(p = drop),
    nn.Linear(64, 10),
    nn.Softmax(dim = 1))


# In[89]:


#seq_model


# In[90]:


# optimizer_algorithm
#cnn_optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
optimizer = torch.optim.Adam(seq_model.parameters(), lr = 1e-4)
# loss_function
if apply_class_weights == True:
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()


# In[ ]:





# In[91]:


'''# complete the architecture of MLP and compile MLP 

units = 64  
drop = 0.89  

model_mlp = Sequential()
model_mlp.add(Dense(units, input_dim = 2048, activation = 'relu'))
model_mlp.add(Dropout(drop))
model_mlp.add(Dense(10, activation = 'softmax'))
model_mlp.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
         metrics = ['accuracy'])'''


# In[92]:


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    #optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, max_epochs +1), desc = "Epoch", position = 0, leave = True):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for compounds, labels in tqdm(train_loader, desc = "Batches w/in epoch", 
                                      position = 0,
                                     leave = True):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(compounds)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,labels)
            # Update weights
            loss.backward()
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        val_accuracy = val_correct/val_total
        # printing results for epoch
        if epoch == 1 or epoch %2 == 0:
            print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch


# In[93]:


def validation_loop(model, loss_fn, valid_loader, best_val_loss):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    loss_val = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for compounds, labels in tqdm(valid_loader, 
                                               desc = "Val Batches w/in Epoch",
                                              position = 0,
                                              leave = True):
            # Move to device MAY NOT BE NECESSARY
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(compounds)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        #print(f' best_val_loss {best_val_loss},loss_val: {loss_val}')
        if best_val_loss > loss_val:
            best_val_loss = loss_val
            now = datetime.datetime.now()
            now = now.strftime("%d_%m_%Y-%H:%M:%S")
            torch.save(
                {
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val,
                    'time_create' : now
            },  '/home/jovyan/Tomics-CP-Chem-MoA/Compound_structure_based_models/saved_models' +'/' + 'ChemStruc_Least_Loss_Model'
            )
    return correct, total, avg_val_loss, best_val_loss


# In[101]:

def results_assessment(y_true, y_pred):
    now = datetime.datetime.now()
    now = now.strftime("%d_%m_%Y-%H:%M:%S")
    save_path = '/home/jovyan/Tomics-CP-Chem-MoA/Compound_structure_based_models/'
    target_names = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor',
     'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 'topoisomerase inhibitor', 
     'PARP inhibitor','ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']
    #print(f' true {y_true}')
    #print(f' prediction {y_pred}')
    for count, ele in enumerate(target_names):
        #print(count)
        #print(ele)
        #print(dictionary[ele])
        assert count == dictionary[ele], "Mismatch with dictionary"
    
    class_report_output = classification_report(y_true, y_pred)
    try:
        chem_struc_file= open(save_path + 'saved_classification_reports' + '/' + now + '_classif_report.txt', 'a')
        chem_struc_file.write((class_report_output))
        chem_struc_file.close()
    except:
        print("Unable to append to file")
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    np.save((save_path + 'saved_confusion_matrices/'  + now + '_confusion_matrix.npy'), conf_matrix)



def test_loop(model, loss_fn, test_loader):
    '''
    Assessing trained model on test dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    test_loader: generator creating batches of test data
    '''
    loss_test = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on test data.
        for compounds, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = True):
            # Move to device MAY NOT BE NECESSARY
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(compounds)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,labels)
            loss_test += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
            #print(f' Predicted: {predicted.tolist()}')
            #print(f' Labels: {predicted.tolist()}')
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + labels.tolist()
        results_assessment(all_predictions, all_labels)
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss


# In[94]:

print("Beginning Training")
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = seq_model,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=valid_generator)


# In[ ]:


#print(train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch)


# In[ ]:


def val_vs_train_loss(epochs, train_loss, val_loss):
    ''' 
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_loss: training loss per epoch (python list)
    val_loss: validation loss per epoch (python list)
    ''' 
    loss_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/Compound_structure_based_models/saved_images'
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_loss, label = "train_loss")
    plt.plot(x_axis, val_loss, label = "val_loss")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.title('Validation versus Training Loss: Chemical Structure Model')
    plt.legend()
    # plot
    now = datetime.datetime.now()
    now = now.strftime("%m_%d_%Y-%H:%M:%S")
    plt.savefig(loss_path_to_save + '/' + 'loss_train_val_' + now)


# In[100]:


def val_vs_train_accuracy(epochs, train_acc, val_acc):
    '''
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_acc: accuracy loss per epoch (python list)
    val_acc: accuracy loss per epoch (python list)
    '''
    acc_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/Compound_structure_based_models/saved_images'
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_acc, label = "train_acc")
    plt.plot(x_axis, val_acc, label = "val_acc")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation versus Training Accuracy: Chemical Structure Image Model')
    plt.legend()
    # plot
    now = datetime.datetime.now()
    now = now.strftime("%m_%d_%Y-%H:%M:%S")
    plt.savefig(acc_path_to_save + '/' + 'acc_train_val_' + now)
    


# In[ ]:


val_vs_train_loss(max_epochs,train_loss_per_epoch, val_loss_per_epoch)


# In[ ]:


val_vs_train_accuracy(max_epochs, train_acc_per_epoch, val_acc_per_epoch)


# In[107]:


correct, total, avg_test_loss = test_loop(model = seq_model,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)


# In[108]:
end = time.time()
def program_elapsed_time(start, end):
    program_time = round(end - start, 2) 
    print(program_time)
    if program_time > float(60) and program_time < 60*60:
        program_time =  program_time/60
        time_elapsed = str(program_time) + ' min'
    elif program_time > 60*60:
        program_time = program_time/3600
        time_elapsed = str(program_time) + ' hrs'
    else:
        time_elapsed = str(program_time) + ' sec'
    return time_elapsed
program_elapsed_time = program_elapsed_time(start, end)

test_set_acc = f' {round(correct/total*100, 2)} %'
table = [["Time to Run Program", program_elapsed_time],
['Accuracy of Test Set', test_set_acc]]
print(tabulate(table, tablefmt='fancy_grid'))

