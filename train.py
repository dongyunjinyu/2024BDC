import os
import time
import random
import warnings
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from iTransformer import iTransformer
import numpy as np
warnings.filterwarnings('ignore')

seed = 2024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)

def diff_train(data):
    diff = np.diff(data, n=1, axis=0)
    diff = np.concatenate([diff[:1], diff], axis=0)
    return diff

def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Dataset(Dataset):
    def __init__(self):
        self.tot_len = data.shape[0] - 168 - 72 + 1

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + 168
        r_begin = s_end
        r_end = r_begin + 72
        # data是全局变量，不在Dataset中定义
        return data[s_begin:s_end, :, station_id], data[r_begin:r_end, :, station_id]
    def __len__(self):
        return self.tot_len * data.shape[-1]


def data_provider(args):
    data_set = Dataset()
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 使用页锁定内存
        prefetch_factor=2,  # 提前加载数据
        persistent_workers=True,  # 保持worker在epoch之间活跃
        worker_init_fn=worker_init_fn)
    return data_loader

class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        self.model = iTransformer(self.args).float().to('cuda')

    def train(self):
        train_loader = data_provider(self.args)
        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(train_loader)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        crits = {'mae': nn.L1Loss(), 'huber': nn.SmoothL1Loss(),
                 'mae2': nn.L1Loss(reduction='none'), 'huber2': nn.SmoothL1Loss(reduction='none')}
        crit = crits[self.args.loss]
        if self.args.loss[-1] == '2':
            weights = torch.linspace(1, 2, 72)  # 从 1 增加到 2
            weights = weights / 1.5  # 归一化，使得权重之和为 1
            weights = weights.view(1, -1)  # (1, 72, 1)
        best = 50

        self.model.train()
        time_start = time.time()
        i = 1
        for epoch in range(self.args.train_epochs):
            for batch_x, batch_y in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                outputs = self.model(batch_x)
                outputs = outputs[:, -72:, -2:]
                batch_y = batch_y[:, -72:, -2:]
                if self.args.loss[-1] == '2':
                    weights=weights.expand(outputs.shape[0], -1)
                    loss1 = crit(outputs[:, :, -1], batch_y[:, :, -1]) * (weights.to(self.device))
                    loss2 = crit(outputs[:, :, -2], batch_y[:, :, -2]) * (weights.to(self.device))
                    loss = loss1.mean() / 0.8 + loss2.mean() / 1.3
                else:
                    loss = crit(outputs[:, :, -1], batch_y[:, :, -1]).item() / 0.8 + crit(outputs[:, :, -2],batch_y[:, :,-2]).item() / 1.3
                if self.args.loss == 'huber' or self.args.loss == 'huber2':
                    loss = (loss - 1) * 2
                if loss < best:
                    best = loss
                    torch.save(self.model.state_dict(), path + '/' + f'checkpoint{(i // 1000) % 3}.pth')
                if i % 20 == 0:
                    speed = (time.time() - time_start) / 20
                    left_time = speed * (self.args.train_epochs * train_steps - i)
                    print(f'{i}, epoch: {epoch + 1} | loss: {loss:.2f}; left time: {left_time / 60:.2f}min')
                    crit2 = nn.MSELoss()
                    mse = crit2(outputs[:, :, -1], batch_y[:, :, -1]).item() / 8 + crit2(outputs[:, :, -2],batch_y[:, :, -2]).item() / 13
                    print(f'mse:{mse:.2f}')
                    time_start = time.time()
                loss.backward()
                model_optim.step()
                if i % 800 == 0:
                    best = 50
                    adjust_learning_rate(model_optim, i // 800 + 1, self.args)
                    torch.save(self.model.state_dict(), path + '/' + f'checkpoint{(i // 800) % 3 + 3}.pth')
                i += 1


# 读取数据，如果数据未找到可能是因为未挂载、解压缩少生成一级目录，此时将bdc_train9198/global/global/换为bdc_train9198/global/
wind = np.squeeze(np.load('/home/mw/input/bdc_train9198/global/global/wind.npy')).astype(np.float32)  # (T, S)
temp = np.squeeze(np.load('/home/mw/input/bdc_train9198/global/global/temp.npy')).astype(np.float32)  # (T, S)
data = np.load('/home/mw/input/bdc_train9198/global/global/global_data.npy').astype(np.float32)

# 异常值处理
for i in range(3850):
    mean_rounded = round(temp[:, i].mean())
    for j in range(1, 17544):
        if round(temp[j, i]) == mean_rounded:
            temp[j, i] = temp[j - 1, i]

# 特征合并
data = np.repeat(data, 3, axis=0).mean(axis=2)  # (T, 4, S)


# 定义模型参数，此处不是所有参数，dff、nhead、loss三个参数在训练时指定
class args:
    checkpoints = '/home/mw/project/best_model/'
    d_model = 256
    e_layers = 2
    dropout = 0.12
    activation = 'gelu'
    num_workers = 10
    train_epochs = 2
    batch_size = 15000
    learning_rate = 0.005
    output_attention = False


print('加载成功，开始特征工程')  # 开始构建新特征
wind_diff = diff_train(wind)[:, np.newaxis, :]  # 风速差分
temp_diff = diff_train(temp)[:, np.newaxis, :]  # 温度差分
wind_minus = (np.sqrt(np.square(data[:, 0, :]) + np.square(data[:, 1, :])) - wind)[:, np.newaxis, :]  # 风速差
temp_minus = (data[:, 2, :] - temp)[:, np.newaxis, :]  # 温度差
press_diff = diff_train(data[:, -1, :])[:, np.newaxis, :]  # 压强差分
square = np.sqrt(np.square(data[:, 0, :]) + np.square(data[:, 1, :]))[:, np.newaxis, :]  # 标量风速
qh = square * temp_minus  # 热通量
wci = ((10.45 + 10 * np.sqrt(np.abs(wind)) - wind) * (33 - temp))[:, np.newaxis, :]  # 风冷指数

# 合并所有特征
data = np.concatenate(
    [data, square, qh, wci, press_diff, temp_minus, wind_minus, temp_diff, wind_diff, temp[:, np.newaxis, :],
     wind[:, np.newaxis, :]], axis=1)
del wind_diff, temp_diff, press_diff, temp, wind, square, qh, wci, temp_minus, wind_minus  # 清除不用的变量
print('特征构建完成')
data = data[:, :, :3800]  # 训练集

#十个基模型的训练
for arg in [[4,1024,'mae'],[4,1024,'huber'],[8,512,'mae'],[8,512,'mae2'],[8,512,'huber'],
            [8,512,'huber2'],[4,512,'mae'],[4,512,'mae2'],[4,512,'huber'],[4,512,'huber2']]:

    #打印、指定参数
    print(arg)
    args.n_heads=arg[0]
    args.d_ff=arg[1]
    args.loss=arg[2]

    #固定种子
    seed = 2026
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    #实例化Exp并将dataloader传入进行训练
    exp = Exp_Long_Term_Forecast(args)
    exp.train(data_provider)