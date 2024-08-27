import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from DataSet.EEG_all_DataLoader import EEG_all_Dataloader
from DataSet.GetSubset import get_subset
from select_model import pick_models
import metrics
from tqdm import tqdm

def Train(model, model_name, train_loader, vali_loader, N_Epochs, Discriminator=None):
    if(Discriminator != None):
        optimizer_D = torch.optim.Adam(model.parameters(), lr=1E-2, weight_decay=1E-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-2, weight_decay=1E-5)
    loss_fn = nn.MSELoss(reduction='mean')

    MSE_train_dB_epoch = torch.empty(N_Epochs)
    MSE_cv_dB_epoch = torch.empty(N_Epochs)

    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0
    for ti in range(0, N_Epochs):
        optimizer.zero_grad()
        # Training Mode
        model.train()
        running_loss = 0.0
        for i, (data, labels) in enumerate(tqdm(train_loader), 0):
            if Discriminator == None:
                tr_filter_signal = model(data)
                LOSS = loss_fn(tr_filter_signal, labels)
                running_loss = running_loss + LOSS.item()
                LOSS.backward()
            else:
                if i % 1 == 0:
                    p_t = model(data)
                    fake_y = Discriminator(p_t)
                    real_y = Discriminator(labels)

                    d_loss = 0.5 * (torch.mean((fake_y) ** 2)) + 0.5 * (torch.mean((real_y - 1) ** 2))

                    optimizer_D.zero_grad()
                    d_loss.backward()
                    optimizer_D.step()

                if i % 1 == 0:
                    p_t = model(data)

                    loss = loss_fn(p_t, labels)

                    optimizer_D.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            optimizer.step()
        MSE_train_dB_epoch[ti] = 10 * torch.log10(torch.tensor(running_loss))

        #################################
        ### Validation Sequence Batch ###
        #################################

        with torch.no_grad():
            vali_loss = 0.0
            for i, (data, labels) in enumerate(tqdm(vali_loader), 0):
                cv_filter_signal = model(data)
                LOSS = loss_fn(cv_filter_signal, labels)
                vali_loss = running_loss + LOSS.item()
            MSE_cv_dB_epoch[ti] = 10 * torch.log10(torch.tensor(vali_loss))

            if (MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):
                MSE_cv_dB_opt = MSE_cv_dB_epoch[ti]
                MSE_cv_idx_opt = ti
                path = 'model_save/' + model_name + '.pt'
                torch.save(model.state_dict(), path)

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  MSE_cv_dB_epoch[ti],
                  "[dB]")
        if (ti > 0):
            d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
            d_cv = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
            print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

        print("Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")

def Test(model, model_name, test_loader):
    # MSE LOSS Function
    loss_fn = nn.MSELoss(reduction='mean')
    path = 'model_save/' + model_name + '.pt'
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    model.eval()
    test_origin = test_loader.dataset.tensors[1]
    torch.no_grad()
    running_loss = 0.0
    for i, (data, labels) in enumerate(tqdm(test_loader), 0):
        test_denoise = model(data)

        LOSS = loss_fn(test_denoise, labels)
        test_loss = running_loss + LOSS.item()
    MSE_test_dB_avg = 10 * torch.log10(torch.tensor(test_loss))

    # Print MSE Cross Validation
    str = "MSE Test:"
    print(str, MSE_test_dB_avg, "[dB]")

    test_ssd = test_mad = test_cosineSim = test_cc = 0
    for i in range(10):
        test_ssd_index = metrics.SSD(test_origin[i, 0, :].cpu().detach().numpy(),
                                     test_denoise[i].cpu().detach().numpy())
        test_ssd = test_ssd + test_ssd_index
        test_mad = test_mad + metrics.MAD(test_origin[i, 0, :].cpu().detach().numpy(),
                                          test_denoise[i].cpu().detach().numpy())
        test_cosineSim = test_cosineSim + metrics.CosineSim(test_origin[i, 0, :].cpu().detach().numpy(),
                                                            test_denoise[i].cpu().detach().numpy())
        test_cc = test_cc + np.corrcoef(test_origin[i, 0, :].cpu().detach().numpy(),
                                        test_denoise[i].cpu().detach().numpy())[0, 1]
    print('测试集的SSD是', test_ssd / 10, 'uv')
    print('测试集的MAD是', test_mad / 10, 'uv')
    print('测试集的Cosine Sim是', test_cosineSim / 10)
    print('测试集的CC是', test_cc / 10)

torch.manual_seed(0)

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

#######################
###### 数据载入部分 #####
#######################
dataloader = EEG_all_Dataloader
dataloader = dataloader(512, [0], 1, 0)
train_loader, test_loader = get_subset(dataloader, 20)
vali_loader, test_loader = get_subset(test_loader, 10)

train_set = TensorDataset(train_loader.dataset.observations[:, :5120, 1].reshape(-1, 1, 512), train_loader.dataset.dataset[:, :5120, 1].reshape(-1, 1, 512))
val_set = TensorDataset(vali_loader.dataset.dataset.observations[:, :5120, 1].reshape(-1, 1, 512), vali_loader.dataset.dataset.dataset[:, :5120, 1].reshape(-1, 1, 512))
test_set = TensorDataset(test_loader.dataset.dataset.observations[:, :5120, 1].reshape(-1, 1, 512), test_loader.dataset.dataset.dataset[:, :5120, 1].reshape(-1, 1, 512))

train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
vali_loader = DataLoader(val_set, batch_size=10, shuffle=False)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

modelname = 'GCTNet'
D = None
if modelname == 'GCTNet':
    model, D = pick_models(modelname)
else:
    model = pick_models(modelname)
Train(model, modelname, train_loader, vali_loader, 50)
Test(model, modelname, test_loader)