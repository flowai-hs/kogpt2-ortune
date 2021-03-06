import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import torchvision.transforms as transforms
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

use_cuda = torch.cuda.is_available()
print(use_cuda)


alpha=1       # AWC alpha for training           
beta=1          # AWC beta AWC for inference 
gamma=0.9      # AWCnew <-- (1-gamma) * AWCcurrent + gamma * AWCpast      

InitLR=0.05*10**(-2)
MinLR=InitLR*0.01
MaxEpoch=400

# calculate an initial loss
epoch=-1
running_loss = 0.0
LR_factor=1         # should be 1 for non-zero Lflag

HH=20
DR=0
batch_size = 1024               

MODEL_PATH = './models/'
PLOT_PATH = './plot/'
PATH='ACW1H_'+ str(HH) +'_alpha'+ str(alpha) +'_beta'+ str(beta) +'_gamma'+ str(gamma)
PATH = PATH +'_lr'+ str(InitLR) +'_bs'+ str(batch_size) +'_dr'+ str(DR)
PATH_A = MODEL_PATH + PATH + '.pth'

MaxNoClasses=10
NoTrainSamples = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000 ]  # Imbalanced Data 1
#NoTrainSamples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]    # Imbalanced Data 2
MaxNoTrainSamples=sum(NoTrainSamples)
NoTestSamples=[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
#NoTestSamples=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]               # set the proportion to the NoTrainSamples
MaxNoTestSamples=sum(NoTestSamples)

criterion = nn.CrossEntropyLoss()   # has softmax already and class weights may be defined

class_prob = torch.FloatTensor(NoTrainSamples)
class_prob = class_prob / sum(class_prob)

class_weights = 1 / class_prob ** alpha   # initialize class-weights
class_weights = class_weights / sum(class_weights)


num_classes = MaxNoClasses

number_per_class = {} 
for i in range(num_classes): 
    number_per_class[i] = 0



from torchvision.datasets import ImageFolder

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = ImageFolder(root='./data/Traindata-balanced', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = ImageFolder(root='./data/Testdata-balanced', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # may add batch normailization
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, HH)
        self.fc3 = nn.Linear(HH, 10)
        self.dropout = nn.Dropout(DR)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # ????????? ????????? ?????? ????????? ?????????(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if DR > 0.0 :
            x = self.dropout(x)
        x = self.fc3(x)
        return x

net = Net()


LR_list = []
train_loss_list = []
test_loss_list = []
iteration_list = []
train_class_acc_list=[]
test_class_acc_list=[]
train_class_loss_list=[]
test_class_loss_list=[]
accuracy_list = []
predictions_list = []
labels_list = []
train_total_acc_list = []
train_ave_class_acc_list = []
train_std_class_acc_list = []
test_total_acc_list = []
test_ave_class_acc_list = []
test_std_class_acc_list = []

# initial loss

LR=InitLR


print('Training Start')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
net.to(device)
if torch.cuda.device_count() > 1:
    print('\n===> Training on GPU!')
    net = nn.DataParallel(net)

for i, data in enumerate(trainloader, 0):
    # [inputs, labels]??? ????????? data????????? ????????? ?????? ???;
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)

    loss = criterion(outputs, labels)
    running_loss += loss.item()

# print loss for each epoch
print(epoch + 1, running_loss / MaxNoTrainSamples, LR)
# save loss
#train_loss_list.append(running_loss)
prev_loss = running_loss
LR_list.append(LR)


min_loss=999999
iepoch=0
optimizer = optim.SGD(net.parameters(), lr=InitLR, momentum=0.9)

for epoch in range(MaxEpoch):   # ??????????????? ????????? ???????????????.

    # training
    running_loss = 0.0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    train_class_loss = torch.zeros(MaxNoClasses)

    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]??? ????????? data????????? ????????? ?????? ???;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # ?????????(Gradient) ??????????????? 0?????? ?????????
        optimizer.zero_grad()

        # ????????? + ????????? + ???????????? ??? ???
        outputs = net(inputs)
        outputs = outputs.to(device)
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False, reduction='none') 
        
        # get lotal and class losses
        loss = criterion(outputs, labels)
        loss = loss.to(device)

        for label, one_loss in zip(labels, loss):
            train_class_loss = train_class_loss.to(device)
            train_class_loss[label] += one_loss

        # add loss
        loss=sum(loss)
        running_loss += loss.item()

        # backprop and weight adjustment
        loss.backward()
        optimizer.step()

        # check accuracy 
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    # get lists
    running_loss=running_loss/MaxNoTrainSamples
    print('Training  ',epoch + 1, running_loss, LR)
    print("Class Weights: ",class_weights)
    # save loss
    train_loss_list.append(running_loss)
    LR_list.append(LR)
    # reduce learning rate if loss increases
    if prev_loss < running_loss :
        LR=LR*LR_factor
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    prev_loss = running_loss
    
    # save the best model
    if running_loss < min_loss :
        min_loss=running_loss
        PATH_B = MODEL_PATH + PATH +'_Best.pth'
        #torch.save(net.state_dict(), PATH)
        #torch.save(net, PATH_B)

    # training accuracy check 
    # total and class accuracy
    sum_correct_count = 0
    sum_class_accuracy = 0
    sum_class_acc_sq = 0
    sum_total_count = 0

    # get class accuracy
    ii=0
    train_class_acc_list.append([0,0,0,0,0,0,0,0,0,0])
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        #print("   Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy), end='')  
        print('   ',accuracy, end='') 
        sum_correct_count += correct_count
        sum_total_count += total_pred[classname]
        sum_class_accuracy += accuracy
        sum_class_acc_sq += accuracy * accuracy
        train_class_acc_list[iepoch][ii] = accuracy
        ii += 1
    print('')
    # average of class accucaries
    total_accuracy = 100 * float(sum_correct_count) / sum_total_count  
    ave_class_accuracy = sum_class_accuracy / len(classes)
    std_class_accuracy = math.sqrt(sum_class_acc_sq / len(classes) - ave_class_accuracy * ave_class_accuracy)
    print("   Average Class Accuracy is: {:.1f} %".format(ave_class_accuracy))
    print("   STD of Class Accuracy is: {:.1f} %".format(std_class_accuracy))
    print("   Weighted Accuracy is: {:.1f} %".format(total_accuracy))
    train_total_acc_list.append(total_accuracy)
    train_ave_class_acc_list.append(ave_class_accuracy)
    train_std_class_acc_list.append(std_class_accuracy)


    # --------------------------------------------------------------------
    # performace testing for all test data
    test_running_loss = 0.0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    test_class_loss = torch.zeros(MaxNoClasses)


    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            criterion = nn.CrossEntropyLoss(weight=class_weights, reduce=False, reduction='none') 
            loss = criterion(outputs, labels) 

            for label, one_loss in zip(labels, loss):
                test_class_loss[label] = test_class_loss[label] + one_loss
        
            loss=sum(loss)
            test_running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
        
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    
    test_running_loss=test_running_loss / MaxNoTestSamples
    print('Test   ',epoch + 1, test_running_loss, LR)
    test_loss_list.append(test_running_loss)
    
    # total and class accuracy
    sum_correct_count = 0
    sum_class_accuracy = 0
    sum_class_acc_sq = 0
    sum_total_count = 0

    # get class accuracy
    ii=0
    test_class_acc_list.append([0,0,0,0,0,0,0,0,0,0])
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        
        sum_class_accuracy += accuracy
        sum_class_acc_sq += accuracy * accuracy

        #weighted accuracy
        #sum_total_count += total_pred[classname]
        sum_correct_count += correct_count 
        test_class_acc_list[iepoch][ii] = accuracy
        ii += 1 
        #print("   Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy), end='')
        print('  ',accuracy, end='')
    print('')

    # average of class accucaries
    ave_class_accuracy = sum_class_accuracy / len(classes)
    std_class_accuracy = math.sqrt(sum_class_acc_sq / len(classes) - ave_class_accuracy * ave_class_accuracy)
    #weighred accuracy 
    total_accuracy = 100*float(sum_correct_count) /MaxNoTestSamples 
    print("   Average Class Accuracy is: {:.1f} %".format(ave_class_accuracy))
    print("   STD of Class Accuracy is: {:.1f} %".format(std_class_accuracy))
    print("   Weighted Accuracy is: {:.1f} %".format(total_accuracy))
    test_total_acc_list.append(total_accuracy)
    test_ave_class_acc_list.append(ave_class_accuracy)
    test_std_class_acc_list.append(std_class_accuracy)

    # prepare for the next epoch
    iepoch +=1
    train_class_loss_list.append(train_class_loss)
    test_class_loss_list.append(test_class_loss)

    with torch.no_grad():
        dummy = train_class_loss ** beta
        dummy = dummy/sum(dummy)
        class_weights = (1 - gamma) * dummy  + gamma * class_weights


print('Finished Training: '+ PATH)


# save state_dict
torch.save(net.state_dict(), PATH_A)


# load state_dict
net = Net()
#net=TheModelClass(*args, **kwargs)
net.load_state_dict(torch.load(PATH_A))
#net.eval()


# In[ ]:


# save each variables
'''
torch.save(train_loss_list,'train_loss_list'+PATH+'.pth')
torch.save(test_loss_list,'test_loss_list'+PATH+'.pth')
torch.save(train_class_loss_list,'train_class_loss_list'+PATH+'.pth')
torch.save(test_class_loss_list,'test_class_loss_list'+PATH+'.pth')
torch.save(train_total_acc_list,'train_total_acc_list'+PATH+'.pth')
torch.save(test_total_acc_list,'test_total_acc_list'+PATH+'.pth')
torch.save(train_ave_class_acc_list,'train_ave_class_acc_list'+PATH+'.pth')
torch.save(test_ave_class_acc_list,'test_ave_class_acc_list'+PATH+'.pth')
torch.save(train_std_class_acc_list,'train_std_class_acc_list'+PATH+'.pth')
torch.save(test_std_class_acc_list,'test_std_class_acc_list'+PATH+'.pth')

# load each variables
train_loss_list=torch.load('train_loss_list'+PATH+'.pth')
test_loss_list=torch.load('test_loss_list'+PATH+'.pth')
train_class_loss_list=torch.load('train_class_loss_list'+PATH+'.pth')
test_class_loss_list=torch.load('test_class_loss_list'+PATH+'.pth')
train_total_acc_list=torch.load('train_total_acc_list'+PATH+'.pth')
test_total_acc_list=torch.load('test_total_acc_list'+PATH+'.pth')
train_ave_class_acc_list=torch.load('train_ave_class_acc_list'+PATH+'.pth')
test_ave_class_acc_list=torch.load('test_ave_class_acc_list'+PATH+'.pth')
train_std_class_acc_list=torch.load('train_std_class_acc_list'+PATH+'.pth')
test_std_class_acc_list=torch.load('test_std_class_acc_list'+PATH+'.pth')
'''


# Plot loss learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
ax.plot(x,train_loss_list,'r',label='Total Loss for Train Data')
ax.plot(x,test_loss_list,'g',label='Total Loss for Test Data')

legend = ax.legend(loc='upper right', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Loss',
       title='Training and Test Loss during Learning')

fig.savefig(PLOT_PATH+"PLOT_PATHLearning Loss Curves-"+PATH+'.png',dpi=200)


# In[ ]:


# Plot accuracy learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
ax.plot(x,train_total_acc_list,'r',label='Total Acc for Train Data')
ax.plot(x,test_total_acc_list,'g',label='Total Acc for Test Data')
ax.plot(x,train_ave_class_acc_list,'r--',label='Ave Accuracy for Train Data')
ax.plot(x,test_ave_class_acc_list,'g--',label='Ave Accuracy for Test Data')
ax.plot(x,train_std_class_acc_list,'r:',label='Std Acc for Train Data')
ax.plot(x,test_std_class_acc_list,'g:',label='Std Acc for Test Data')

legend = ax.legend(loc='center right', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Accuracy (%)',
       title='Total, Average, and STD Accuracies during Learning')

fig.savefig(PLOT_PATH+"Learning Accuracy Curves: " +PATH+'.png',dpi=200)


# In[ ]:


# Plot class accuracy learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
trans_list=np.transpose(train_class_acc_list)

ax.plot(x,trans_list[0],'#8c564b',label='plane')
ax.plot(x,trans_list[1],'#d62728',label='var')
ax.plot(x,trans_list[2],'#ff7f0e',label='bird')
ax.plot(x,trans_list[3],'#bcbd22',label='cat')
ax.plot(x,trans_list[4],'#2ca02c',label='deer')
ax.plot(x,trans_list[5],'#17becf',label='dog')
ax.plot(x,trans_list[6],'#1f77b4',label='frog')
ax.plot(x,trans_list[7],'#9467bd',label='horse')
ax.plot(x,trans_list[8],'#e377c2',label='ship')
ax.plot(x,trans_list[9],'#7f7f7f',label='truck')

legend = ax.legend(loc='upper left', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Class Accuracy (%)',
       title='Class Accuracies during Learning for Training Data')

fig.savefig(PLOT_PATH+"Learning Class Accuracy Curves for Training Data: " +PATH+'.png',dpi=200)


# In[ ]:


# Plot class accuracy learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
trans_list=np.transpose(test_class_acc_list)

ax.plot(x,trans_list[0],'#8c564b',label='plane')
ax.plot(x,trans_list[1],'#d62728',label='var')
ax.plot(x,trans_list[2],'#ff7f0e',label='bird')
ax.plot(x,trans_list[3],'#bcbd22',label='cat')
ax.plot(x,trans_list[4],'#2ca02c',label='deer')
ax.plot(x,trans_list[5],'#17becf',label='dog')
ax.plot(x,trans_list[6],'#1f77b4',label='frog')
ax.plot(x,trans_list[7],'#9467bd',label='horse')
ax.plot(x,trans_list[8],'#e377c2',label='ship')
ax.plot(x,trans_list[9],'#7f7f7f',label='truck')

legend = ax.legend(loc='upper left', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Class Accuracy (%)',
       title='Class Accuracies during Learning for Training Data')

fig.savefig(PLOT_PATH+"Learning Class Accuracy Curves for Test Data: " +PATH+'.png',dpi=200)


# In[ ]:


# Plot class loss learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
trans_list=np.transpose(train_class_loss_list)

ax.plot(x,trans_list[0],'#8c564b',label='plane')
ax.plot(x,trans_list[1],'#d62728',label='var')
ax.plot(x,trans_list[2],'#ff7f0e',label='bird')
ax.plot(x,trans_list[3],'#bcbd22',label='cat')
ax.plot(x,trans_list[4],'#2ca02c',label='deer')
ax.plot(x,trans_list[5],'#17becf',label='dog')
ax.plot(x,trans_list[6],'#1f77b4',label='frog')
ax.plot(x,trans_list[7],'#9467bd',label='horse')
ax.plot(x,trans_list[8],'#e377c2',label='ship')
ax.plot(x,trans_list[9],'#7f7f7f',label='truck')

legend = ax.legend(loc='upper left', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Class Loss',
       title='Class Loss during Learning for Training Data')

fig.savefig(PLOT_PATH+"Class Loss Curves for Training Data: " +PATH+'.png',dpi=200)


# In[ ]:


# Plot class loss learning curves
x = np.arange(1,MaxEpoch+1,1)
fig, ax =plt.subplots()
trans_list=np.transpose(test_class_loss_list)

ax.plot(x,trans_list[0],'#8c564b',label='plane')
ax.plot(x,trans_list[1],'#d62728',label='var')
ax.plot(x,trans_list[2],'#ff7f0e',label='bird')
ax.plot(x,trans_list[3],'#bcbd22',label='cat')
ax.plot(x,trans_list[4],'#2ca02c',label='deer')
ax.plot(x,trans_list[5],'#17becf',label='dog')
ax.plot(x,trans_list[6],'#1f77b4',label='frog')
ax.plot(x,trans_list[7],'#9467bd',label='horse')
ax.plot(x,trans_list[8],'#e377c2',label='ship')
ax.plot(x,trans_list[9],'#7f7f7f',label='truck')

legend = ax.legend(loc='upper left', shadow=False, fontsize='small')
ax.set(xlabel='Epoch', ylabel='Class Loss',
       title='Class Loss during Learning for Test Data')

fig.savefig(PLOT_PATH+"Class Loss Curves for Test Data: " +PATH+'.png',dpi=200)


# save entire model
torch.save(net,PATH_A)


# load entire model
#APath='dr00h20'
model=torch.load(PATH_A)
model.eval()

max_loss=min(test_loss_list)
max_ind=test_loss_list.index(max_loss)
print('Min Test Loss: Train/Test Ave/STD Acc:  ', train_ave_class_acc_list[max_ind],
        test_ave_class_acc_list[max_ind], train_std_class_acc_list[max_ind],test_std_class_acc_list[max_ind])
max_loss=max(test_ave_class_acc_list)
max_ind=test_ave_class_acc_list.index(max_loss)
print('Max Test Ave: Train/Test Ave/STD Acc:  ', train_ave_class_acc_list[max_ind],
        test_ave_class_acc_list[max_ind], train_std_class_acc_list[max_ind],test_std_class_acc_list[max_ind])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CUDA ????????? ???????????????, ?????? ????????? CUDA ????????? ???????????????:

print(device)

del dataiter


