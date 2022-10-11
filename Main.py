
# Author 
# SeongDeok Ko : 2020-38259
# With Younghae Kim

#%%
import numpy as np
import pandas as pd 
from collections import Counter
from google.colab import drive
drive.mount('/content/drive')
import multiprocessing as mp

#추가함
import math
import torch

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)


#%% Data Loading (interaction은 메모리 한계로 인해서 제외 
# Ram 한계로 인해서 Float32, Int32로 변경
new_firm_data = pd.read_csv('/content/drive/MyDrive/x_y_wo_inter.csv')

new_firm_data[new_firm_data.columns[2:96]] = new_firm_data[new_firm_data.columns[2:96]].astype('float32')
new_firm_data[new_firm_data.columns[0:2]] = new_firm_data[new_firm_data.columns[0:2]].astype('int32')
new_firm_data[new_firm_data.columns[96:170]] = new_firm_data[new_firm_data.columns[96:170]].astype('int8')
new_firm_data[new_firm_data.columns[170]] = new_firm_data[new_firm_data.columns[170]].astype('float32')

#%% Change Sample Period 1957.3 ~ 2016.12  > 1963.01 ~ 2016.12    
# You can change 1963.01 with 'All_sample_Start'
from datetime import datetime
from dateutil import relativedelta

date = new_firm_data['DATE']
result = Counter(date)
date_firm = list(result.keys())
date_num_firm = list(result.values())

del date
del result 

# Gu,Kelly,Xiu sample period
Gu_Start = '1957-03-31'   
Gu_sample_End = '2016-12-31'     
Gu_delta = relativedelta.relativedelta(datetime.strptime(Gu_sample_End, "%Y-%m-%d"), datetime.strptime(Gu_Start, "%Y-%m-%d"))
Gu_sp_len = 12*Gu_delta.years + Gu_delta.months + 1  


# Our All sample period
All_sample_Start = '1963-01-31'  
All_sample_End = Gu_sample_End       # final month is the same
All_delta = relativedelta.relativedelta(datetime.strptime(All_sample_End, "%Y-%m-%d"), datetime.strptime(All_sample_Start, "%Y-%m-%d"))
All_sp_len = 12*All_delta.years + All_delta.months + 1  

delete_months = Gu_sp_len - All_sp_len

date_firm = date_firm[delete_months:]
date_num_firm = date_num_firm[delete_months:]
new_firm_data = new_firm_data.iloc[-sum(date_num_firm):,:]

#%%  Setting sample period
# training : 17 years,  cv : 10 years, test : 27 years
# training : 1963.1~1979.12, cv : 1980.1 ~ 1989.12, test : 1990.1 ~ 2016.12
# after 1 year, we re-estimate our model with 1 more year training sample period  (CV last 10 year, 10 is fixed)

# OOS period 
OOS_Start = '1990-01-31'  
OOS_End = All_sample_End    
OOS_delta = relativedelta.relativedelta(datetime.strptime(OOS_End, "%Y-%m-%d"), datetime.strptime(OOS_Start, "%Y-%m-%d"))
OOS_len = 12*OOS_delta.years + OOS_delta.months + 1     
OOS_num_estimate = math.ceil(OOS_len/12)    # = 27

# CV
CV_Start = '1980-01-31'  
CV_End = '1989-12-31'    # 1month before OOS_End
CV_delta = relativedelta.relativedelta(datetime.strptime(CV_End, "%Y-%m-%d"), datetime.strptime(CV_Start, "%Y-%m-%d"))
CV_len = 12*CV_delta.years + CV_delta.months + 1   

#%% To define Function

def R2OOS(y_true, y_forecast):
    
    import numpy as np
   
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true))

    return 1-SSres/SStot


# =========================================================================
#   PCR, 94 + SIC Indicators  
# =========================================================================

def Pca_regression(X, Y, numpc, num_tr_cv_te):
    # numpc (list) : # of principal component ex[3,4,5,6,7]
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    
    # use cross-validation mean-squared-error to determine the number of principal component 
    mse = np.full((len(numpc),1),np.nan)

    for i in range(len(numpc)):
        pca = PCA(n_components = numpc[i])
        principalComponents = pca.fit_transform(X_train_scaled)
        
        X_val_weighted = pca.transform(X_val_scaled)
        
        line_fitter = LinearRegression()
        line_fitter.fit(principalComponents, Y_train)
        
        Ypred_val = np.full((num_val,1),np.nan)
        for j in range(num_val):
            Ypred_val[j,0] = line_fitter.predict(X_val_weighted[j,:].reshape(1,-1))
                   
        mse[i,0] = mean_squared_error(Y_val.reshape(-1), Ypred_val.reshape(-1))
    
    
    argmin_numpc = numpc[np.argmin(mse)]
    
    pca = PCA(n_components = argmin_numpc)
    principalComponents = pca.fit_transform(X_train_scaled)
    
    X_test_weighted = pca.transform(X_test_scaled)
    
    line_fitter = LinearRegression()
    line_fitter.fit(principalComponents, Y_train)
        
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=line_fitter.predict(X_test_weighted[j,:].reshape(1,-1))
        
          
    return Ypred_test, argmin_numpc




# =========================================================================
#   PLS, 94 + SIC indicato 
#   Use cross-validation to select the number of components  
# =========================================================================

def Pls_regression(X,Y,numpls,num_tr_cv_te):
    # numpls (list) : # of component ex[3,4,5,6,7]
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    
    # use cross-validation mean-squared-error to determine the number of component 
    mse = np.full((len(numpls),1),np.nan)

    for i in range(len(numpls)):
        pls = PLSRegression(n_components = numpls[i])
        pls.fit(X_train_scaled, Y_train)
                
        Ypred_val = np.full((num_val,1),np.nan)
        for j in range(num_val):
            Ypred_val[j,0]=pls.predict(X_val_scaled[j,:].reshape(1,-1))          
        
        mse[i,0] = mean_squared_error(Y_val.reshape(-1), Ypred_val.reshape(-1))
    
    
    argmin_numpls = numpls[np.argmin(mse)]
    
    pls = PLSRegression(n_components = argmin_numpls)
    pls.fit(X_train_scaled, Y_train)
                
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=pls.predict(X_test_scaled[j,:].reshape(1,-1))          
              
    
    return Ypred_test, argmin_numpls

# =========================================================================
#  elastic-net, Loss : mse + penalty, 94 + dummy variable(no intersection term), hyperparameter tuning
# ========================================================================= 

def elastic_net(X,Y,num_tr_cv_te):
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
    # Split data into training and test
    X_train = X[:(num_train+num_val),:]   # train + validation
    Y_train = Y[:(num_train+num_val),:]   # train + validation
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)

    X_test_scaled = X_scaler.transform(X_test)
    
    # pre-define validation 
    test_fold =  np.concatenate((np.full((num_train),-1),np.full((num_val),0)))
    ps = PredefinedSplit(test_fold.tolist())
    
    # fit & predict 
    model = ElasticNetCV(cv=ps, max_iter=3000, n_jobs=-1, l1_ratio=[.01, .05, .1, .25, .5, .8, .95, 1], \
                         alphas = [.01, 0.05, 0.1, .25, .5, 1., 2., 3., 4.], random_state=42, selection ='random')
    model = model.fit(X_train_scaled, Y_train.reshape(-1))
    
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=model.predict(X_test_scaled[j,:].reshape(1,-1))
        
    
    return Ypred_test

# =========================================================================
#   Generalized-linear, 94 + dummy variable(no intersection term), 
#   Use cross-validation to select the number of PCA components  
# =========================================================================
# Loss ftn : MSE (not huber loss)
# We use Lasso (Not group Lass) 
# include spline series of order 2 
# number of knots = [3,5,7...] and we choose the only one that minimize cross-validation MSE 
# we set knots by using linspace(col.mean-2*col.std, col.mean+2*col.std, # knots)
# for example if we use 3 knots, the # of variables is 94(order1) + 94*3(order 2) + dummy(74) = 450 

def general_linear(X,Y,num_tr_cv_te, num_knots):
    # num_t_v (list) : # of training set & cross-val set   ex[100, 10]
    # X consists of Traing, Val and Test set
    
    import numpy as np 
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    from sklearn.linear_model import LassoCV
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
       
    mse = np.full((len(num_knots),1),np.nan)
    Ypred_test = np.full((len(num_knots),num_test,1),np.nan)
    
    for i in range(len(num_knots)):
        
        X_temp = X
        
        # 94 variables > make spline series of order 2
        for j in range(94):
            
            # make knots
            std_train = np.std(X[:num_train,j])
            mean_train = np.mean(X[:num_train,j])           
            
            knots = np.linspace(mean_train-2*std_train, mean_train+2*std_train, num_knots[i])
            
            # add (variable - knots)**2 column
            for k in knots:
                add_col = ((X[:,j]-k)**2).reshape(-1,1)
                X_temp = np.concatenate((X_temp, add_col), axis=1)
        
        print(X_temp.shape)
        
        # Split data into training and test
        X_train = X_temp[:(num_train+num_val),:]   # train + validation
        Y_train = Y[:(num_train+num_val),:]   # train + validation
        
        X_test = X_temp[(num_train+num_val):,:]
        
        # Scale Inputs for Training
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        
        X_test_scaled = X_scaler.transform(X_test)
        
        # pre-define validation 
        test_fold =  np.concatenate((np.full((num_train),-1),np.full((num_val),0)))
        ps = PredefinedSplit(test_fold.tolist())
        
        # we use cross-val to find best 'alpha'(penalty term in loss function)
        model = LassoCV(cv=ps, max_iter=3000,  alphas = [.01, 0.05, 0.1, .25, .5, 1., 2., 3., 4.], \
                        n_jobs=-1, random_state=42)
        model = model.fit(X_train_scaled, Y_train.reshape(-1))

        
        # to choose # of knots, calculate mse of validation set
        Ypred_val = np.full((num_val,1),np.nan)
        for j in range(num_val):
            Ypred_val[j,0]=model.predict(X_train_scaled[num_train+j,:].reshape(1,-1))
            
        mse[i,0] = mean_squared_error(Y[num_train:(num_train+num_val),:].reshape(-1), Ypred_val.reshape(-1))
        
        # predict test set 
        for j in range(num_test):
            Ypred_test[i,j,0]=model.predict(X_test_scaled[j,:].reshape(1,-1))
    
    
    # choose knots that minimize mse in validation
    argmin_index = np.argmin(mse)
    
    print(argmin_index)
    
    Ypred_test_final = Ypred_test[argmin_index,:,:].reshape(-1,1)
    
    return Ypred_test_final


# =========================================================================
#                   Random-forest w/o intersection term  (hyperparameter tuning)
# =========================================================================

# in random forest, no need to scale x-variable 
def Random_Forest(X,Y,num_tr_cv_te):
    
    # use intersection 
    # We can set many hyper-parameter in Random forest model. 
    # 1. the depth of the individual trees(max_depth),  
    # 2. the size of the randomly selected sub-set of predictors (max_features) 
    # 3. the number of trees(n_estimators)  
    # 4. min_samples_split   5. min_samples_leaf  6. max_samples  ..........etc....
    # 
    # Here we only consider 1(max_depth), 2(max_features), 6(max samples) for hyper-parameter tuning
    
    # for detail, refer to algorithms in details 'ML_supp' file
    # we set hyper-parameter following Table A.5
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import RandomizedSearchCV
    
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
    # Split data into training and test
    X_train = X[:(num_train+num_val),:]   # train + validation
    Y_train = Y[:(num_train+num_val),:]   # train + validation
    
    X_test = X[(num_train+num_val):,:]
    
    # pre-define validation 
    test_fold =  np.concatenate(((np.full((num_train),-1),np.full((num_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    
    
    # Set hyper-parameter candidate 
    max_depth = [10,15,25,30,35]
    max_features = ["sqrt"]
    max_samples = [0.5]
    n_estimators = [200]
    
    grid_param = {'n_estimators':n_estimators, 'max_depth':max_depth, 'max_features':max_features, 'max_samples':max_samples}     

    RFR = RandomForestRegressor(bootstrap = True, n_jobs=-1, random_state=42)
        
    RFR_grid = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_jobs=-1, cv=ps)
    RFR_grid.fit(X_train, np.ravel(Y_train))
        
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=RFR_grid.predict(X_test[j,:].reshape(1,-1))
        
    return Ypred_test





def Neural_net(X, Y, num_tr_cv_te, archi, epoch, Seed):
    # archi : # of neurons in hyden layer 
    # Use mini-batch, MSE Loss 
    # Linear > Relu > Batch-normalization > linear > Relu > BN .... > linear 
    # Adam optimizer, Learning decay, Early-stopping
    # Use GPU if available (Designed for Google Colab)
    # Use Specified Seed to use ensemble method 
    # Unlike the original paper L2 penalty term is used based on Adam

    import torch 
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np  
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')

    #seed 
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    
    num_train = num_tr_cv_te[0]
    num_val = num_tr_cv_te[1]
    num_test = num_tr_cv_te[2]
    
    # Split data into training and test
    X_train = X[:num_train,:]
    Y_train = Y[:num_train,:]
    
    X_val = X[num_train:(num_train+num_val),:]
    Y_val = Y[num_train:(num_train+num_val),:]
    
    X_test = X[(num_train+num_val):,:]
    
       
    # Scale Inputs for Training
    X_scaler = MinMaxScaler(feature_range=(-1,1))
    X_train_scaled = X_scaler.fit_transform(X_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    # from np.array > torch tensor 
    X_train_scaled = torch.tensor(X_train_scaled).to(device)    
    X_val_scaled = torch.tensor(X_val_scaled).to(device)
    X_test_scaled = torch.tensor(X_test_scaled).to(device)
    
    Y_train = torch.tensor(Y_train).to(device)
    Y_val = torch.tensor(Y_val).to(device)
        
    # dataset
    train_dataset = TensorDataset(X_train_scaled, Y_train)   
    valid_dataset = TensorDataset(X_val_scaled, Y_val)
    
    trainloader = DataLoader(train_dataset, batch_size= 8192, shuffle=True, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size= 8192, shuffle=True, drop_last=True)
    


    # define Network 
    class NN_fwd_model(nn.Module):
        
        def __init__(self, X_dim, Y_dim, archi):
            
            super(NN_fwd_model, self).__init__()
            
            n = len(archi)
            self.nn_module = torch.nn.Sequential()
            
            for i in range(n):
                if i==0:                 
                    self.nn_module.add_module('linear'+str(i+1), nn.Linear(X_dim, archi[i]))
                    self.nn_module.add_module('Relu'+str(i+1), nn.ReLU())
                    self.nn_module.add_module('BN'+str(i+1), nn.BatchNorm1d(archi[i]))
                    
                else:                  
                    self.nn_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
                    self.nn_module.add_module('Relu'+str(i+1), nn.ReLU())
                    self.nn_module.add_module('BN'+str(i+1), nn.BatchNorm1d(archi[i]))                    
            
            # for output layer
            self.lastlinear = nn.Linear(archi[-1], Y_dim)
                    
            # Using He-initilization 
            for m in self.nn_module:
                if isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            
            nn.init.kaiming_normal_(self.lastlinear.weight, nonlinearity="relu")
                    
         
        def forward(self, X_train_scaled):
           y_hat = self.nn_module(X_train_scaled)
           y_hat = self.lastlinear(y_hat)
           
           return y_hat
       
        
    model = NN_fwd_model(X_train_scaled.shape[1], Y_train.shape[1], archi)
    model = model.to(device)
    print(model)
    
    
    # define loss ftn 
    loss_ftn = torch.nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0075, weight_decay= 0.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda epoch: 0.975 ** epoch)
    
    min_val_loss = np.Inf
    epochs_no_improve = np.nan

    for i in range(epoch):       
        
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
  
        model.train()
        for (batch_X, batch_Y) in trainloader:
            
            optimizer.zero_grad()
           
            # compute the model output
            trained_y = model(batch_X.float())            
            
            # calculate loss
            
            loss = loss_ftn(trained_y, batch_Y.float())        
            # credit assignment
            loss.backward()
            
            # update model weights
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        for (batch_X_val, batch_Y_val) in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch_X_val.float())
            # calculate the loss
            loss = loss_ftn(output, batch_Y_val.float())
            # record validation loss
            valid_losses.append(loss.item())         
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)        
        
        if i % 5 ==0:
            print('the epoch number ' + str(i) + ' (train_loss) : ' + str(train_loss))
            print('the epoch number ' + str(i) + ' (valid_loss) : ' + str(valid_loss))
        
        # Early-stopping
        if valid_loss < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = valid_loss
             print('Minimum Validation Loss Updated at '+ str(i) +'th Epochs Loss Level ' + str(min_val_loss))
             torch.save(model.state_dict(), 'best_model_NN.pt')
  
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve > 25:
            print('Early stopping! at ' + str(i) + 'th Epochs' )
            break
        else:
            continue
        
    
    model.load_state_dict(torch.load('best_model_NN.pt'))
    model.eval()
    Ypred_test = np.full((num_test,1),np.nan)
    for j in range(num_test):
        Ypred_test[j,0]=model(X_test_scaled[j,:].float().unsqueeze(0))
        
        
    return Ypred_test

#%% setting X, y

X_no_inter = new_firm_data.iloc[:,2:-1]
y = new_firm_data.iloc[:,-1] 

y_true = new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,-1].to_numpy().reshape(-1,1)

y_true_pd = pd.concat([new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,:2].reset_index(drop=True), y_true.reset_index(drop=True)], axis=1)

y_true_pd.to_csv('/content/drive/MyDrive/y_true.csv', index = False)
#%% PCR 
# Use CV to determine the # of principal components 

for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function 'Pca_regression' (num_train, num_cv, num_test is input variable) 
    X_all_pca = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # train + val +test 
    y_all_pca = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
    
    numpc = [3,5,10]
    
    # the 'best' number of principal component is different for each estimation     
    Ypred_temp, argmin_numpc_temp = Pca_regression(X_all_pca, y_all_pca, numpc, num_tr_cv_te)
    
    if i==0:
        Y_predict_pca = Ypred_temp
        Y_predict_pca_numpc = [argmin_numpc_temp]
    else:
        Y_predict_pca = np.concatenate((Y_predict_pca, Ypred_temp), axis=0)
        Y_predict_pca_numpc.append(argmin_numpc_temp)

print('PCR')
print(R2OOS(y_true, Y_predict_pca))

Y_predict_pca = pd.DataFrame(Y_predict_pca, columns=['PCR'])
Y_predict_pca = pd.concat([new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,:2].reset_index(drop=True), Y_predict_pca.reset_index(drop=True)], axis=1)
Y_predict_pca.to_csv('/content/drive/MyDrive/y_pcr.csv', index = False)



#%% PLS 구동 
# Use CV to determine the # of components 

for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function 'Pls_regression' (num_train, num_cv, num_test is input variable) 
    X_all_pls = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # no CV
    y_all_pls = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
    
    
    numpls = [3,5,8,10]
    
    # the 'best' number of principal component is different for each estimation     
    Ypred_temp, argmin_numpls_temp = Pls_regression(X_all_pls, y_all_pls, numpls, num_tr_cv_te)
    
    if i==0:
        Y_predict_pls = Ypred_temp
        Y_predict_pls_numpls = [argmin_numpls_temp]
    else:
        Y_predict_pls = np.concatenate((Y_predict_pls, Ypred_temp), axis=0)
        Y_predict_pls_numpls.append(argmin_numpls_temp)


print('PLS')
print(R2OOS(y_true, Y_predict_pls))

Y_predict_pls = pd.DataFrame(Y_predict_pls, columns=['PLS'])
Y_predict_pls = pd.concat([new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,:2].reset_index(drop=True), Y_predict_pls.reset_index(drop=True)], axis=1)
Y_predict_pls.to_csv('/content/drive/MyDrive/y_pls.csv', index = False)



#%% GLM  구동 (Not huber loss)
for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function (num_train, num_cv, num_test is input variable) 
    X_all = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # no CV
    y_all = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
       
    Ypred_temp = general_linear(X_all, y_all, num_tr_cv_te)
    
    if i==0:
        Y_predict_general_linear = Ypred_temp
    else:
        Y_predict_general_linear = np.concatenate((Y_predict_general_linear, Ypred_temp), axis=0)

print('Generalized-linear')
print(R2OOS(y_true, Y_predict_general_linear.reshape(-1)))

Y_predict_general_linear = pd.DataFrame(Y_predict_general_linear, columns=['GLM'])
Y_predict_general_linear = pd.concat([new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,:2].reset_index(drop=True), Y_predict_general_linear.reset_index(drop=True)], axis=1)
Y_predict_general_linear.to_csv('/content/drive/MyDrive/y_glm.csv', index = False)

#%% Elastic-net  (Not huber loss)
for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function (num_train, num_cv, num_test is input variable) 
    X_all = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # no CV
    y_all = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
    
     
    Ypred_temp = elastic_net(X_all, y_all, num_tr_cv_te)
    
    if i==0:
        Y_predict_elastic = Ypred_temp
    else:
        Y_predict_elastic = np.concatenate((Y_predict_elastic, Ypred_temp), axis=0)

print('Elastic-net')
print(R2OOS(y_true, Y_predict_elastic.reshape(-1)))

Y_predict_elastic = pd.DataFrame(Y_predict_elastic, columns=['elastic'])
Y_predict_elastic = pd.concat([new_firm_data.iloc[-sum(date_num_firm[-OOS_len:]):,:2].reset_index(drop=True), Y_predict_elastic.reset_index(drop=True)], axis=1)
Y_predict_elastic.to_csv('/content/drive/MyDrive/y_elastic.csv', index = False)

#%% RF 구동 (진짜 진짜 오래걸림 )
for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function (num_train, num_cv, num_test is input variable) 
    X_all = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # no CV
    y_all = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
    
    
    Ypred_temp = Random_Forest(X_all, y_all, num_tr_cv_te)
    
    if i==0:
        Y_predict_rf = Ypred_temp
    else:
        Y_predict_rf = np.concatenate((Y_predict_rf, Ypred_temp), axis=0)

Y_predict_rf = pd.DataFrame(Y_predict_rf)
Y_predict_rf.to_csv('/content/drive/MyDrive/Y_predict_rf.csv', index = False)





#%% NN3 구동 
archi3 = [32,16,8]
epoch3 = 200
seed1 = 1109
seed2 = 1117
seed3 = 1123
seed4 = 1223
for i in range(OOS_num_estimate): 
    print(i)
    
    num_train_months = All_sp_len-OOS_len-CV_len+12*i
    num_cv_months = CV_len    #120 months
    num_test_months = 12   # for each estimation, predict 12months
    
    num_train = sum(date_num_firm[:num_train_months])
    num_cv = sum(date_num_firm[num_train_months:(num_train_months+num_cv_months)])
    num_test = sum(date_num_firm[(num_train_months+num_cv_months):(num_train_months+num_cv_months+num_test_months)])
    
    num_tr_cv_te = [num_train, num_cv, num_test]
    
    # train + val + test 
    # split them in function (num_train, num_cv, num_test is input variable) 
    X_all = X_no_inter.iloc[:sum(num_tr_cv_te),:].to_numpy()    # no CV
    y_all = y.iloc[:sum(num_tr_cv_te)].to_numpy().reshape(-1,1)
    
    print('First Seed')
    Ypred_temp1 = Neural_net(X_all, y_all, num_tr_cv_te , archi3, epoch3, seed1)
    print('Second Seed')
    Ypred_temp2 = Neural_net(X_all, y_all, num_tr_cv_te , archi3, epoch3, seed2)
    print('Third Seed')
    Ypred_temp3 = Neural_net(X_all, y_all, num_tr_cv_te , archi3, epoch3, seed3)  
    print('Third Seed')
    Ypred_temp4 = Neural_net(X_all, y_all, num_tr_cv_te , archi3, epoch3, seed4) 
    if i==0:
        Y_predict_NN3_1 = Ypred_temp1
        Y_predict_NN3_2 = Ypred_temp2
        Y_predict_NN3_3 = Ypred_temp3
        Y_predict_NN3_4 = Ypred_temp4
    else:
        Y_predict_NN3_1 = np.concatenate((Y_predict_NN3_1, Ypred_temp1), axis=0)
        Y_predict_NN3_2 = np.concatenate((Y_predict_NN3_2, Ypred_temp2), axis=0)
        Y_predict_NN3_3 = np.concatenate((Y_predict_NN3_3, Ypred_temp3), axis=0)
        Y_predict_NN3_4 = np.concatenate((Y_predict_NN3_4, Ypred_temp3), axis=0)

    Y_predict_NN3_1 = pd.DataFrame(Y_predict_NN3_1, columns=['NN3'])
    Y_predict_NN3_2 = pd.DataFrame(Y_predict_NN3_2, columns=['NN3'])
    Y_predict_NN3_3 = pd.DataFrame(Y_predict_NN3_3, columns=['NN3'])
    Y_predict_NN3_4 = pd.DataFrame(Y_predict_NN3_4, columns=['NN3'])
    Y_predict_NN3_1.to_csv('/content/drive/MyDrive/y_NN3_1.csv', index = False)
    Y_predict_NN3_2.to_csv('/content/drive/MyDrive/y_NN3_2.csv', index = False)
    Y_predict_NN3_3.to_csv('/content/drive/MyDrive/y_NN3_3.csv', index = False)
    Y_predict_NN3_4.to_csv('/content/drive/MyDrive/y_NN3_4.csv', index = False)
