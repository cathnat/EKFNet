import torch
from Model.Linear_sysmdl import SystemModel
from Pipeline_KF import Pipeline_KF
from Model.KalmanNet_nn import KalmanNetNN
from DataSet.EEG_all_DataLoader import EEG_all_Dataloader
from DataSet.GetSubset import get_subset

torch.manual_seed(0)

# if torch.cuda.is_available():
#    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#    torch.set_default_tensor_type('torch.cuda.FloatTensor')
#    print("Running on the GPU")
# else:
dev = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')
print("Running on the CPU")
print("Pipeline Start")

#######################
#### 设置物理模型参数 ####
#######################
N_CV = 10
N_T = 10
F = torch.tensor([[1.0, 0.0],[0.0, 1.0]])
H = torch.tensor([[1.0, 0.0],[0.0, 1.0]])
T = 512
T_test = 512
m1_0 = torch.tensor([[0.0], [0.0]])
m2_0 = 0 * 0 * torch.eye(2)

#######################
####### 生成噪声 #######
#######################

# 噪声协方差矩阵的生成
r2 = torch.tensor(10)
vdB = -20  # r2和q2的比例： v=q2/r2
v = 10 ** (vdB / 10)  # 值为0.01，r2和q2数值相差100倍
q2 = torch.mul(v, r2)

#######################
###### 物理建模部分 #####
#######################
# r:测量方程的噪声信号
r = torch.sqrt(r2)
# q:状态方程的噪声信号
q = torch.sqrt(q2)
sys_model = SystemModel(F, q, H, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)  # m1_0代表X0,初始测量值；m2_0为P0，初始协方差

#######################
###### 数据载入部分 #####
#######################
dataloader = EEG_all_Dataloader
dataloader = dataloader(512, [0], 1, 0)
prior_loader, test_loader = get_subset(dataloader, 10)
max_val = prior_loader.dataset.max_val
min_val = prior_loader.dataset.min_val
noiseEEG_train = torch.zeros((10, 1, 512))
EEG_train = torch.zeros((10, 1, 512))
noiseEEG_val = torch.zeros((10, 1, 512))
EEG_val = torch.zeros((10, 1, 512))
noiseEEG_test = torch.zeros((10, 1, 512))
EEG_test = torch.zeros((10, 1, 512))
for i in range(10):
    index = 512 * i
    noiseEEG_train[i] = prior_loader.dataset.observations[0, index:index + 512, 0].float()
    EEG_train[i] = prior_loader.dataset.dataset[0, index:index + 512, 0].float()
    noiseEEG_val[i] = prior_loader.dataset.observations[0, index + 5120:index + 5632, 0].float()
    EEG_val[i] = prior_loader.dataset.dataset[0, index + 5120:index + 5632, 0].float()
    noiseEEG_test[i] = prior_loader.dataset.observations[0, index + 10240:index + 10752, 0].float()
    EEG_test[i] = prior_loader.dataset.dataset[0, index + 10240:index + 10752, 0].float()

#######################
###### KalmanNet ######
#######################
print("Start Model pipeline")
# KNet管道的初始化
KNet_Pipeline = Pipeline_KF()
# KNet管道的状态空间模型设置为sys_model
KNet_Pipeline.setssModel(sys_model)
# 定义一个KNet
KNet_model = KalmanNetNN()
# 构建KNet模型
KNet_model.Build(sys_model)
KNet_Pipeline.setModel(KNet_model)
# 设置KNet模型的参数
KNet_Pipeline.setTrainingParams(n_Epochs=200, n_Batch=10, learningRate=1E-2, weightDecay=1E-5)
# 计算训练集、验证集的MES
KNet_Pipeline.NNTrain(noiseEEG_train, EEG_train, N_CV, noiseEEG_val, EEG_val)
# 计算测试集的MES
[KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(N_T, noiseEEG_test, EEG_test)