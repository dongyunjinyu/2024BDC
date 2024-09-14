import os
import numpy as np
import random
import torch
from iTransformer import iTransformer

seed = 2024
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def diff(data, axis=1):  # (N, L, 1, S)
    diff = np.diff(data, n=1, axis=axis)
    diff = np.concatenate([diff[:, :1, :, :], diff], axis=axis)
    return diff

def invoke(inputs):
    cwd = os.path.dirname(inputs)
    save_path = '/home/mw/project'
    class args:
        d_model = 256
        e_layers = 2
        output_attention = False
        dropout = 0
        activation = 'gelu'

    test_path = inputs
    temp = np.load(os.path.join(test_path, "temp_lookback.npy")).transpose(0, 1, 3, 2) # (N, L, 1, S)
    wind = np.load(os.path.join(test_path, "wind_lookback.npy")).transpose(0, 1, 3, 2) # (N, L, 1, S)
    N, L, _, S = wind.shape # 72, 168, 60
    era5 = np.load(os.path.join(test_path, "cenn_data.npy")) # (N, L, 4, 9, S)
    wind_res = np.empty((N, 72, S, 0)) # 储存每一个结果的容器
    temp_res = np.empty((N, 72, S, 0))

    data = np.repeat(era5, 3, axis=1).mean(axis=3)  # (N, L, 4, S)
    wind_minus = (np.sqrt(np.square(data[:,:, 0:1, :]) + np.square(data[:,:, 1:2, :])) - wind)
    temp_minus = (data[:,:, 2:3, :] - temp)
    press_diff = diff(data[:, :, -1, :][:,:, np.newaxis, :])
    square = np.sqrt(np.square(data[:,:, 0:1, :]) + np.square(data[:,:, 1:2, :]))
    qh = square * temp_minus
    wci = ((10.45 + 10 * np.sqrt(np.abs(wind)) - wind) * (33 - temp))
    data = np.concatenate([data, square, qh, wci, press_diff, temp_minus, wind_minus, diff(temp), diff(wind), temp, wind], axis=2)  # (N, L, 14, S)
    data = data.transpose(0, 3, 1, 2)  # (N, S, L, 14)
    data = data.reshape(N * S, L, data.shape[-1])  # (N * S, L, 14)
    data = torch.tensor(data).float().cuda()
    del temp_minus, wind_minus, press_diff, square, qh, wci, temp, wind
    x = data.permute(1, 0, 2).cpu().numpy()
    x = torch.Tensor(x).permute(1, 0, 2).cuda()
    for weights in os.listdir('best_model'):
        hyp = weights.split('-')
        args.n_heads = int(hyp[0])
        args.d_ff = int(hyp[1])
        model = iTransformer(args).cuda().eval()
        for i in range(6):
            model.load_state_dict(torch.load('best_model/' + weights + f'/checkpoint{i}.pth'))
            outputs = model(x)[:, :, -2:].detach().cpu().numpy()  # (N * S, P, 2)
            forecast = outputs.reshape(N, S, outputs.shape[1], 2).transpose(0, 2, 1, 3)  # (N, P, S, 2)
            wind_res = np.concatenate([wind_res, forecast[:, :, :, 1:]], axis=3)
            temp_res = np.concatenate([temp_res, forecast[:, :, :, :1]], axis=3)

    ###### ensemble
    temp_res = np.mean(temp_res, axis=3, keepdims=True)   # (N, P, S, 1)
    wind_res = np.mean(wind_res, axis=3, keepdims=True)   # (N, P, S, 1)
    np.save(os.path.join(save_path, "temp_predict.npy"), temp_res)
    np.save(os.path.join(save_path, "wind_predict.npy"), wind_res)
