import os
import gc
import argparse
import json
import wandb
import swanlab
import time
from datetime import datetime
from operator import itemgetter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Any, List, Tuple
from kornia.losses import SSIMLoss
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle # 导入pickle用于保存结果
import hydra
from omegaconf import DictConfig, OmegaConf
import socket
# 获取当前主机的主机名
hostname = socket.gethostname()
swanlab.login(api_key='I4Ge4KHvI92q1Bg5DkrRl')

from data.SEGC3 import reconstruct, PatchMetadataManager
from dataset.dataset_SEGC3 import Dataset_SEGC3
from dataset.dataset_MAVG import Dataset_MAVG
from dataset.dataset_BP2004 import Dataset_BP2004
from dataset.dataset_F3 import Dataset_F3
from dataset.dataset_Model94 import Dataset_Model94
from utils.meta_optimizers import MetaSGD
from models.compare_models import HaarDWT2D, curvelet_reconstruct
from utils.utils import AverageMeter, load_trained_model, seed_rng, makedirs, get_project_path, load_model, count_parameters, format_parameters_as_million, format_times_as_hms, torch_dtype, get_noise
from utils.utils import signal_to_noise as snr
from utils.utils import peak_signal_to_noise as psnr
from utils.utils import structural_similarity as ssim
from utils.loss import TVLoss, DiscriminatorLoss, LossCalculator, D3UNetLoss, SPNetLoss, loss_dict
from utils.plot import train_loss_figure, val_loss_figure, train_SNR_figre, val_SNR_figre, \
    train_PSNR_figre, val_PSNR_figre, train_SSIM_figre, val_SSIM_figre, SNR_figre, PSNR_figre, SSIM_figre
from utils.plot import plot_seismic, plot_difference, plot_mask

def train(config, model, train_loader, optimizer, scheduler, loss_list, SNR_function, PSNR_function, SSIM_function, device, epoch, run):

    # model
    model = model[0]
    if getattr(model, 'multi_exit_supervision', False):
        # 本入口仅补充固定T训练/逐出口评估；双层优化和Meta-GF仍使用原BLO入口。
        raise ValueError('多出口监督训练请使用train_DP_BLO.py；本入口支持其检查点的逐出口测试')
    model.train()

    # optimizer
    optimizer = optimizer[0]

    # Initialize AverageMeters
    metrics = {
        'SSIM': AverageMeter(),
        'total_loss': AverageMeter(),
        'SNR': AverageMeter(),
        'PSNR': AverageMeter(),
    }

    # Add meters for all potential losses (standard + extra)
    # all_potential_losses = ['L1Loss', 'SSIMLoss', 'MSELoss', 'PerceptLoss','Pocsnet_loss', 'discrepancy_loss', 'constraint_loss']
    for k in loss_list:
        metrics[k] = AverageMeter()

    # loss initialization
    # Note: loss_list[0] is used for the main loss calculator based on original code logic
    loss_objects = {name: loss_dict[name](device) for name in loss_list if name in loss_dict}
    loss_function = LossCalculator(config, loss_objects, config.loss.loss_weight, device)

    # # 用于收集所有Batch数据的列表，用于Epoch结束时计算全局指标
    # train_preds_list = []
    # train_targets_list = []

    # results
    results_dict  = {}

    min_ratio, max_ratio = 1.0, 0.0

    for step, (patch_H, patch_L, patch_name, patch_mask, patch_dict, ratio) in enumerate(train_loader):
        # print(type(patch_H), type(patch_L), type(patch_mask),len(patch_L),len(patch_L[0]),len(patch_L[1]))
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)
        max = ratio.max().item()
        min = ratio.min().item()
        if max >= max_ratio:
            max_ratio = max
        if min <= min_ratio:
            min_ratio = min

        # Input generation (DIP style input)
        batch_size = patch_H.size(0)
        out = patch_L # Default init
        state = {} # Reset state for current batch
        total_loss = 0.0

        if config.model.name == 'POCS':
            # POCS logic (adapted for batch training)
            th = config.threshold - (config.threshold / config.Epoch) * epoch
            out = model(out, patch_H, mask=patch_mask, th=th)
            # Loss calculation
            total_loss, state = loss_function(state, patch_H, out, patch_mask, model)

        elif config.model.name == 'DIPOCS':
            optimizer.zero_grad()
            net_input = get_noise(batch_size, config.hyp.input_dim, config.dip_input_type, (config.dataset.resolution_h,config.dataset.resolution_w), noise_type='u', var=1./10).to(device)
            if config.dataset.reg_noise_std > 0:
                net_input = net_input + (torch.randn_like(net_input) * config.dataset.reg_noise_std)

            # Threshold schedule
            th = config.threshold - (config.threshold / config.Epoch) * epoch

            out, pocs_out = model(net_input, patch_H, mask=patch_mask, th=th)

            # Main Loss
            mse_loss_val, state = loss_function(state, patch_H, out, patch_mask, model)

            # Reg Loss
            loss_objects_reg = {name: loss_dict[name](device) for name in loss_list[1] if name in loss_dict}
            beta = np.linspace(config.loss_weight[1], 2, config.Epoch)
            current_beta = beta[epoch] if epoch > 2000 else config.loss_weight[1]
            loss_function_reg = LossCalculator(config, loss_objects_reg, [current_beta], device)

            reg_loss_val, state = loss_function_reg(state, out, pocs_out, patch_mask, model)

            total_loss = mse_loss_val + reg_loss_val
            total_loss.backward()
            optimizer.step()

        elif config.model.name == 'POCSNet':
            # POCSNet Logic
            optimizer.zero_grad()
            # Note: loss_list passed entire list for POCSNet in original code?
            # Re-init loss function for POCSNet if needed to include all losses
            loss_objects_pocs = {name: loss_dict[name](device) for name in loss_list if name in loss_dict}
            loss_function_pocs = LossCalculator(config, loss_objects_pocs, config.loss.loss_weight, device)

            out, layers_sym = model(Phi=patch_mask, b=patch_L, x0=patch_L)
            total_loss, state = loss_function_pocs(state, patch_H, [out, layers_sym], patch_mask, model)

            total_loss.backward()
            optimizer.step()

        elif config.model.name in ['DP', 'AdaptiveDP']:
            # 保留固定T的原损失形式，仅适配POCS网络必需的观测和mask参数。
            optimizer.zero_grad()
            out, sym_loss = model(patch_L, patch_L, patch_mask)
            total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)
            total_loss.backward()
            optimizer.step()
        elif config.model.name == 'D3UNet':
            dwt_module = HaarDWT2D().to(device)
            with torch.no_grad():
                theta_gt = dwt_module(patch_H)
            optimizer.zero_grad()
            out, theta_hat = model(patch_L, patch_mask)
            total_loss, state = loss_function(state, [patch_H, theta_gt], [out, theta_hat], patch_mask, model)

            total_loss.backward()
            optimizer.step()
        elif config.model.name == 'SPNet':
            optimizer.zero_grad()
            out, sym_loss = model(patch_L, patch_mask)
            total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)

            total_loss.backward()
            optimizer.step()
        else:
             # Default/DIP Generic Logic
             optimizer.zero_grad()
             if config.model.name == 'DIP':
                 out = model(net_input)
             else:
                 out = model(patch_L)

             total_loss, state = loss_function(state, patch_H, out, patch_mask, model)
             total_loss.backward()
             optimizer.step()

        # Update Metrics
        with torch.no_grad():
            cur_ssim = SSIM_function(patch_H, out)
            cur_snr = SNR_function(patch_H, out)
            cur_psnr = PSNR_function(out, patch_H, max_val=1.0)

            metrics['SSIM'].update(cur_ssim, batch_size)
            metrics['SNR'].update(cur_snr, batch_size)
            metrics['PSNR'].update(cur_psnr, batch_size)
            metrics['total_loss'].update(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss, batch_size)

            # Update dynamic losses from state
            for key in state:
                if key in metrics:
                    val = state[key]
                    val_float = val.item() if isinstance(val, torch.Tensor) else val
                    metrics[key].update(val_float, batch_size)

            # # 收集数据用于全局计算，转移到CPU防止显存爆炸
            # train_preds_list.append(out.detach().cpu())
            # train_targets_list.append(patch_H.detach().cpu())

    scheduler.step()
    results_dict = {k: v.avg for k, v in metrics.items() if v.count > 0}

    # # 计算全局 SNR 和 PSNR
    # global_preds = torch.cat(train_preds_list, dim=0)
    # global_targets = torch.cat(train_targets_list, dim=0)

    # # 假设 SNR_function 和 PSNR_function 可以处理 CPU Tensor
    # global_snr = SNR_function(global_targets, global_preds)
    # global_psnr = PSNR_function(global_preds, global_targets, max_val=1.0)
    # results_dict['SNR'] = global_snr
    # results_dict['PSNR'] = global_psnr

    return results_dict, min_ratio, max_ratio


def val(config, model, val_data_loader, loss_list, SNR_function, PSNR_function, SSIM_function, state, device):

    model.eval()

    # loss
    loss_objects = {name: loss_dict[name](device) for name in loss_list if name in loss_dict}
    loss_function = LossCalculator(config, loss_objects, config.loss.loss_weight, device)

    # Metrics Meters
    metrics = {
        'total_loss': AverageMeter(),
        'SSIM': AverageMeter(),
        'SNR': AverageMeter(),
        'PSNR': AverageMeter(),
    }
    # Track individual losses
    # known_losses = ['L1Loss', 'SSIMLoss', 'MSELoss', 'PerceptLoss','Pocsnet_loss', 'discrepancy_loss', 'constraint_loss']
    for k in loss_list: metrics[k] = AverageMeter()

    # val_preds_list = []
    # val_targets_list = []

    results_dict = {}
    min_ratio, max_ratio = 1.0, 0.0

    for step, (patch_H, patch_L, patch_name, patch_mask, patch_dict, ratio) in enumerate(val_data_loader):
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)

        max = ratio.max().item()
        min = ratio.min().item()
        if max >= max_ratio:
            max_ratio = max
        if min <= min_ratio:
            min_ratio = min

        batch_size = patch_H.size(0)

        with torch.no_grad():
            if config.model.name == 'POCSNet':
                out, layers_sym = model(Phi=patch_mask, b=patch_L, x0=patch_L)
                total_loss, state = loss_function(state, patch_H, [out, layers_sym], patch_mask, model)
            elif config.model.name == 'D3UNet':
                dwt_module = HaarDWT2D().to(device)
                with torch.no_grad():
                    theta_gt = dwt_module(patch_H)
                out, theta_hat = model(patch_L, patch_mask)
                total_loss, state = loss_function(state, [patch_H, theta_gt], [out, theta_hat], patch_mask, model)
            elif config.model.name == 'SPNet':
                out, sym_loss = model(patch_L, patch_mask)
                total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)
            elif config.model.name in ['DP', 'AdaptiveDP']:
                out, sym_loss = model(patch_L, patch_L, patch_mask)
                total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)
            else:
                out = model(patch_L)
                total_loss, state = loss_function(state, patch_H, out, patch_mask, model)

            # Record losses
            # 仅验证/测试报告指定出口，L1Loss_T为未乘训练权重的MAE。
            for k, depth in enumerate(getattr(model, 'active_eval_exit_depths', ())):
                prediction = model.eval_outputs[:, k]
                for key, value in [(f'SNR_T{depth}', SNR_function(patch_H, prediction)),
                                   (f'PSNR_T{depth}', PSNR_function(prediction, patch_H, max_val=1.0)),
                                   (f'L1Loss_T{depth}', (prediction - patch_H).abs().mean())]:
                    if key not in metrics:
                        metrics[key] = AverageMeter()
                    metrics[key].update(value.item() if isinstance(value, torch.Tensor) else value, batch_size)
            metrics['total_loss'].update(total_loss.item(), batch_size)
            for k, v in state.items():
                if k in metrics:
                     metrics[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)

            cur_ssim = SSIM_function(patch_H, out)
            cur_snr = SNR_function(patch_H, out)
            cur_psnr = PSNR_function(out, patch_H, max_val=1.0)

            metrics['SSIM'].update(cur_ssim, batch_size)
            metrics['SNR'].update(cur_snr, batch_size)
            metrics['PSNR'].update(cur_psnr, batch_size)
            # Record SSIM (average)
            metrics['SSIM'].update(cur_ssim, batch_size)

            # # Collect for global metrics
            # val_preds_list.append(out.detach().cpu())
            # val_targets_list.append(patch_H.detach().cpu())


    # # Calculate Global Metrics
    # global_preds = torch.cat(val_preds_list, dim=0)
    # global_targets = torch.cat(val_targets_list, dim=0)

    # global_snr = SNR_function(global_targets, global_preds)
    # global_psnr = PSNR_function(global_preds, global_targets, max_val=1.0)

    results_dict = {k: v.avg for k, v in metrics.items() if v.count > 0}
    # results_dict['SNR'] = global_snr
    # results_dict['PSNR'] = global_psnr

    return results_dict, min_ratio, max_ratio

def test(config, model, test_data_loader, loss_list, SNR_function, PSNR_function, SSIM_function, state, device, save_exit_depth=None, save_common=True):
    model.eval()

    # Initialize AverageMeters
    metrics = {
        'SSIM': AverageMeter(),
        'total_loss': AverageMeter(),
    }

    # Add meters for all potential losses (standard + extra)
    # all_potential_losses = ['L1Loss', 'SSIMLoss', 'MSELoss', 'PerceptLoss','Pocsnet_loss', 'discrepancy_loss', 'constraint_loss']
    for k in loss_list:
        metrics[k] = AverageMeter()

    # loss
    loss_objects = {name: loss_dict[name](device) for name in loss_list if name in loss_dict}
    loss_function = LossCalculator(config, loss_objects, config.loss.test_loss_weight, device)

    min_ratio, max_ratio = 1.0, 0.0

    # 用于收集所有切片的字典 {patch_name: output_tensor_cpu}
    results_dict = {}

    # 全局SNR/PSNR累加能量和误差，不保留全测试集GT/预测副本；数学口径不变。
    signal_energy = 0.0
    error_energy = 0.0
    pixel_count = 0

    pbar = tqdm(total=len(test_data_loader), ncols=200)
    for step, (patch_H, patch_L, patch_name, patch_mask, patch_dict, ratio) in enumerate(test_data_loader):
        # print(type(patch_H), type(patch_L), type(patch_name), type(patch_mask))
        if isinstance(patch_L, list):
                patch_L = patch_L[0] # 取出mask后的数据
        patch_H, patch_L, patch_mask = patch_H.to(device), patch_L.to(device), patch_mask.to(device)

        num = step + 1
        max_val = ratio.max().item()
        min_val = ratio.min().item()
        if max_val >= max_ratio:
            max_ratio = max_val
        if min_val <= min_ratio:
            min_ratio = min_val

        batch_size = patch_H.size(0)

        with torch.no_grad():
            if config.model.name == 'POCSNet':
                out, layers_sym = model(Phi=patch_mask, b=patch_L, x0=patch_L)
                total_loss, state = loss_function(state, patch_H, out, patch_mask, model)
            elif config.model.name == 'D3UNet':
                dwt_module = HaarDWT2D().to(device)
                with torch.no_grad():
                    theta_gt = dwt_module(patch_H)
                out, theta_hat = model(patch_L, patch_mask)
                total_loss, state = loss_function(state, [patch_H, theta_gt], [out, theta_hat], patch_mask, model)
            elif config.model.name == 'SPNet':
                out, sym_loss = model(patch_L, patch_mask)
                total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)
            elif config.model.name == 'Curvelet':
                out = curvelet_reconstruct(patch_L, patch_mask, method=config.model.method, decay=config.model.decay, nbscales=config.model.nbscales, nbangles_coarse=config.model.nbangles_coarse, niter=config.model.niter, eps=config.model.eps, thresh_max=config.model.thresh_max, thresh_min=config.model.thresh_min)
                total_loss, state = loss_function(state, patch_H, out, patch_mask, model)
            elif config.model.name in ['DP', 'AdaptiveDP']:
                out, sym_loss = model(patch_L, patch_L, patch_mask)
                total_loss, state = loss_function(state, patch_H, [out, sym_loss], patch_mask, model)
            else:
                out = model(patch_L)
                total_loss, state = loss_function(state, patch_H, out, patch_mask, model)

            # 非最后一轮只留出口数组，公共输入和最终输出均不跨batch累积。
            if save_common:
                # 计算插补数据 (Imputed Data)
                # Imputed = Mask * GT + (1 - Mask) * Output
                imputed_data = patch_mask * patch_H + (1 - patch_mask) * out

                # 转换为 Numpy，从 GPU 移到 CPU
                patch_H_np = patch_H.cpu().detach().numpy()
                patch_L_np = patch_L.cpu().detach().numpy()
                patch_mask_np = patch_mask.cpu().detach().numpy()
                out_np = out.cpu().detach().numpy()
                imputed_np = imputed_data.cpu().detach().numpy()

                # [中文标注] 新增：将详细结果存入 results_dict，Key 为 patch_name
                # 根据要求：根据patch_name按照对应关系加入patch_H, patch_L, patch_mask, test_output, imputed_data
                for b in range(batch_size):
                    name = patch_name[b]
                    results_dict[name] = {
                        'patch_H': patch_H_np[b],        # 原始高质图
                        'patch_L': patch_L_np[b],        # 输入低质图
                        'patch_mask': patch_mask_np[b],  # 采样Mask
                        'test_output': out_np[b],        # 模型直接输出
                        'imputed_data': imputed_np[b]    # 最终重建结果 (保留观测值)
                    }
            else:
                for name in patch_name:
                    results_dict[name] = {}

            for k, depth in enumerate(getattr(model, 'active_eval_exit_depths', ())):
                if save_exit_depth is not None and depth != save_exit_depth:
                    continue
                prediction = model.eval_outputs[:, k]
                if config.save_test_plot and config.hyp.save_exit_results:
                    if save_exit_depth is not None and save_common:
                        # auto最后出口等于原最终输出，两个字段引用同一数组，避免双份存储。
                        for name in patch_name:
                            results_dict[name][f'imputed_data_exit_T{depth}'] = results_dict[name]['imputed_data']
                    else:
                        exit_imputed = (patch_mask * patch_H + (1 - patch_mask) * prediction).cpu().numpy()
                        for b, name in enumerate(patch_name):
                            results_dict[name][f'imputed_data_exit_T{depth}'] = exit_imputed[b].copy()
                for key, value in [(f'SNR_T{depth}', SNR_function(patch_H, prediction)),
                                   (f'PSNR_T{depth}', PSNR_function(prediction, patch_H, max_val=1.0)),
                                   (f'L1Loss_T{depth}', (prediction - patch_H).abs().mean())]:
                    if key not in metrics:
                        metrics[key] = AverageMeter()
                    metrics[key].update(value.item() if isinstance(value, torch.Tensor) else value, batch_size)

            # Record losses
            metrics['total_loss'].update(total_loss.item(), batch_size)
            for k, v in state.items():
                if k in metrics:
                     metrics[k].update(v.item() if isinstance(v, torch.Tensor) else v, batch_size)

            # Record SSIM (average)
            metrics['SSIM'].update(SSIM_function(patch_H, out), batch_size)

            # 只保留当前batch的临时数组，float64累计减少求和误差，不改变网络精度。
            target_np = patch_H.detach().cpu().numpy()
            prediction_np = out.detach().cpu().numpy()
            signal_energy += np.square(target_np, dtype=np.float64).sum()
            error_energy += np.square(prediction_np - target_np, dtype=np.float64).sum()
            pixel_count += target_np.size

        pbar.set_description(' '.join(['Step: [{0}/{1}]  '.format(step, len(test_data_loader))] ))
        pbar.update(1)
    pbar.close()

    # 与原工具函数一致：真值能量/总误差，PSNR使用全局MSE，而不是batch指标均值。
    global_snr = 10 * np.log10(signal_energy / (error_energy + 1e-9))
    global_psnr = float('inf') if error_energy == 0 else 10 * np.log10(pixel_count / error_energy)

    results_dict.update({k: v.avg for k, v in metrics.items() if v.count > 0})
    results_dict['SNR'] = global_snr
    results_dict['PSNR'] = global_psnr
    return results_dict, min_ratio, max_ratio

@hydra.main(config_path="../DAP/config", config_name="train", version_base="1.1")
def main(config: DictConfig): # 将 config 参数改为 cfg: DictConfig
    # Base setting
    seed_rng(config.hyp.seed)
    print("\nconfig:", OmegaConf.to_yaml(config)) # 使用 OmegaConf.to_yaml() 打印配置，更符合 Hydra 风格
    print("Exp PID = {}".format(os.getpid()))
    device = torch.device('cuda:' + str(config.hyp.gpu) if torch.cuda.is_available() else 'cpu')
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config.save = os.path.join(config.save, time_now)
    if config.hyp.online_vis == 'wandb':
        run = wandb.init(
                        project="DAP",
                        name='{}'.format('{}_{}'.format(config.save.split('/')[-2], config.save.split('/')[-1])),
                        config=OmegaConf.to_container(config, resolve=True),
                        settings=wandb.Settings(init_timeout=300))
    elif config.hyp.online_vis == 'swanlab':
        run = swanlab.init(
                        project="DAP",
                        experiment_name='{}'.format('{}_{}'.format(config.save.split('/')[-2], config.save.split('/')[-1])),
                        config=OmegaConf.to_container(config, resolve=True),)
    else:
        run = None

    # Path
    root_path = get_project_path()
    if hostname == "ubuntu":
        config.dataset.data_dir = os.path.join(root_path.replace('cxg_jbs/code/DAP', '14.6T/jbs_data'), config.dataset.data_dir)
        config.save = os.path.join(root_path.replace('cxg_jbs/code/DAP', '14.6T/jbs_data'), config.save)
        config.pt_path = os.path.join(root_path.replace('cxg_jbs/code/DAP', '14.6T/jbs_data'), config.pt_path)
        root_path = root_path.replace('cxg_jbs/code/DAP', '14.6T/jbs_data')
        root_path = os.path.join(root_path, 'Hint_seismic/data')
    elif hostname == "xjtu-Sun-04":
        print("root_path:", root_path)
        config.dataset.data_dir = os.path.join(root_path.replace('code/DAP', 'data'), config.dataset.data_dir)
        config.save = os.path.join(root_path.replace('code/DAP', 'code'), config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
        root_path = root_path.replace('code/DAP', 'data')
    elif hostname == "xjtu-Sun-08":
        config.dataset.data_dir = os.path.join(root_path.replace('demo', 'data'), config.dataset.data_dir)
        config.save = os.path.join(root_path, config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
        root_path = root_path.replace('demo', 'data')
    elif hostname == "xjtu-SYJ-03":
        config.dataset.data_dir = os.path.join(root_path.replace('code/DAP', 'data'), config.dataset.data_dir)
        config.save = os.path.join(root_path, config.save)
        config.pt_path = os.path.join(root_path, config.pt_path)
        root_path = root_path.replace('code/DAP', 'data')
    elif hostname == "PC-20201112YXPH":
        config.dataset.data_dir = os.path.join('H:\seismic data', config.dataset.data_dir)
        config.save = os.path.join('G:', config.save)
        config.pt_path = os.path.join('G:/DAP/results',  config.pt_path)
        root_path = 'H:/seismic data/Hint_seismic/data'
        print("config.pt_path:", config.pt_path)
    else:
        print("hostname:", hostname, "Unkown")
    print("data_path ", os.path.join(root_path, f'{config.dataset.name}'))
    print("\nconfig:", OmegaConf.to_yaml(config)) # 使用 OmegaConf.to_yaml() 打印配置，更符合 Hydra 风格

    # data
    if config.dataset.name == 'SEGC3':
        train_dataset = Dataset_SEGC3(config, os.path.join(config.dataset.data_dir, config.dataset.train_data_dir), config.dataset.sampler, config.hyp.mode)
        val_dataset = Dataset_SEGC3(config, os.path.join(config.dataset.data_dir, config.dataset.val_data_dir), config.dataset.sampler, 'val')
        test_dataset = Dataset_SEGC3(config, os.path.join(config.dataset.data_dir, config.dataset.test_data_dir), config.dataset.sampler, 'test')
    elif config.dataset.name == 'MAVG':
        train_dataset = Dataset_MAVG(config, os.path.join(config.dataset.data_dir, config.dataset.train_data_dir), config.dataset.sampler, config.hyp.mode)
        val_dataset = Dataset_MAVG(config, os.path.join(config.dataset.data_dir, config.dataset.val_data_dir), config.dataset.sampler, 'val')
        test_dataset = Dataset_MAVG(config, os.path.join(config.dataset.data_dir, config.dataset.test_data_dir), config.dataset.sampler, 'test')
    elif config.dataset.name == 'F3':
        train_dataset = Dataset_F3(config, os.path.join(config.dataset.data_dir, config.dataset.train_data_dir), config.dataset.sampler, config.hyp.mode)
        val_dataset = Dataset_F3(config, os.path.join(config.dataset.data_dir, config.dataset.val_data_dir), config.dataset.sampler, 'val')
        test_dataset = Dataset_F3(config, os.path.join(config.dataset.data_dir, config.dataset.test_data_dir), config.dataset.sampler, 'test')
    elif config.dataset.name == 'BP2004':
        train_dataset = Dataset_BP2004(config, os.path.join(config.dataset.data_dir, config.dataset.train_data_dir), config.dataset.sampler, config.hyp.mode)
        val_dataset = Dataset_BP2004(config, os.path.join(config.dataset.data_dir, config.dataset.val_data_dir), config.dataset.sampler, 'val')
        test_dataset = Dataset_BP2004(config, os.path.join(config.dataset.data_dir, config.dataset.test_data_dir), config.dataset.sampler, 'test')
    elif config.dataset.name == 'Model94':
        train_dataset = Dataset_Model94(config, os.path.join(config.dataset.data_dir, config.dataset.train_data_dir), config.dataset.sampler, config.hyp.mode)
        val_dataset = Dataset_Model94(config, os.path.join(config.dataset.data_dir, config.dataset.val_data_dir), config.dataset.sampler, 'val')
        test_dataset = Dataset_Model94(config, os.path.join(config.dataset.data_dir, config.dataset.test_data_dir), config.dataset.sampler, 'test')

    train_data_loader = DataLoader(train_dataset, config.hyp.batch_train, shuffle=True, drop_last=True, pin_memory=True, num_workers=config.hyp.num_workers)
    val_data_loader = DataLoader(val_dataset, config.hyp.batch_val, shuffle=False, drop_last=False, num_workers=config.hyp.num_workers)
    test_data_loader = DataLoader(test_dataset, config.hyp.batch_test, shuffle=False, drop_last=False, num_workers=config.hyp.num_workers)

    # Model initialization (Moved outside loop)
    model = load_model(config, device=device)
    print("\nmodel", model)
    model_parameters = list(model.parameters())
    model_params_count = count_parameters(model)
    print("params_count", format_parameters_as_million(model_params_count))
    state = {}

    # Optimizers & scheduler (Moved outside loop)
    if config.hyp.optimizer == 'ADAM':
        optimizer = optim.Adam(model_parameters, lr=config.hyp.lr, weight_decay=config.hyp.wd)
    elif config.hyp.optimizer == 'SGD':
        optimizer = optim.SGD(model_parameters, lr=config.hyp.lr, weight_decay=config.hyp.wd)
    else:
        optimizer = optim.Adam(model_parameters, lr=config.hyp.lr, weight_decay=config.hyp.wd)

    if config.hyp.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.hyp.step_size, gamma=config.hyp.gamma, last_epoch=-1)
    elif config.hyp.scheduler == 'CosineAnnealingLR':
        scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyp.Epoch)

    # Pack for function call
    model_pkg = [model]
    optimizer_pkg = [optimizer]

    # TRAIN
    start_time_all = time.time()

    # 初始化最佳SNR
    best_val_snr = -float('inf')
    best_model_path = ''

    if config.hyp.mode != 'test':
        pbar = tqdm(total=config.hyp.Epoch, ncols=200)
        for epoch in range(config.hyp.Epoch):
            start_time_epoch = time.time()

            # Train one epoch
            results_dict_train, min_ratio, max_ratio = train(config, model_pkg, train_data_loader, optimizer_pkg, scheduler, config.loss.loss_list, snr, psnr, ssim, device, epoch, run)

            # Log to wandb/swanlab
            if run is not None:
                # Create log dict with prefix
                log_dict = {f"train_{k}": f"{v:.7f}" for k, v in results_dict_train.items()}
                run.log(log_dict)

            # Validation Step
            # print(f"Running Validation for Epoch {epoch}...")
            results_dict_val, min_ratio, max_ratio = val(config, model, val_data_loader, config.loss.loss_list, snr, psnr, ssim, state, device)
            if getattr(model, 'active_eval_exit_depths', ()):
                print('同一网络各出口验证指标:', {k: v for k, v in results_dict_val.items()
                                              if k.startswith(('SNR_T', 'PSNR_T', 'L1Loss_T'))})

            current_val_snr = results_dict_val['SNR']
            if torch.cuda.is_available() and epoch ==0:
                allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)  # 实际使用的显存
                reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)     # PyTorch缓存的显存(分配给进程但未被tensor占用的那部分)
                # 你可以根据需要取消注释下面的print直接打印，或者选择把它拼接到下方的 pbar.set_description 里面
                print(f"\n[当前 Epoch {epoch+1} 运行显存] 实际占用 (Allocated): {allocated_mb:.2f} MB | 总保留 (Reserved): {reserved_mb:.2f} MB")
            # Log to wandb/swanlab
            if run is not None:
                # Create log dict with prefix
                log_dict = {f"val_{k}": f"{v:.7f}" for k, v in results_dict_val.items()}
                if run is not None:
                    run.log(log_dict)
                    if epoch ==0:
                        run.log({'Allocated(MB)': f'{allocated_mb:.2f}', 'Reserved(MB)': f'{reserved_mb:.2f}'})
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1} | Loss: {results_dict_train['total_loss']:.4f} | train_SNR: {results_dict_train['SNR']:.4f} | train_PSNR: {results_dict_train['PSNR']:.4f} | train_SSIM: {results_dict_train['SSIM']:.4f} | val_SNR: {results_dict_val['SNR']:.4f} | val_SSIM: {results_dict_val['SSIM']:.4f}| val_PSNR: {results_dict_val['PSNR']:.4f} ")
            # print(f"Validation SNR: {current_val_snr:.4f} (Best: {best_val_snr:.4f})")

            # Save Best Model
            if current_val_snr > best_val_snr:
                best_val_snr = current_val_snr
                save_dir = config.save
                if not os.path.exists(save_dir):
                    makedirs(save_dir)
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                # print(f"Saved new best model to {best_model_path}")

            end_time_epoch = time.time()
            h, m, s = format_times_as_hms(end_time_epoch - start_time_epoch)
        # print(f"\nEpoch {epoch} finished in {h}h {m}m {s:.2f}s")
        pbar.close()
        print(f"Saved new best model to {best_model_path}")
        print("Training Finished!")
    else:
        best_model_path = os.path.join(config.pt_path, "best_model.pth")

    # Run Testing and Save Reconstruction Data
    print("Running Final Test...")
    try:
        if config.model.name == 'Curvelet':
            trained_model = model
        else:
            trained_model = load_trained_model(best_model_path, model, device)
    except Exception as e:
        print(e)
        trained_model = model
    # 仅auto且开启出口保存时串行测试；无出口的普通模型和显式列表保持单轮。
    save_exit_passes = [None]
    if config.save_test_plot and config.hyp.save_exit_results and config.hyp.eval_exit_depths == 'auto':
        depths = tuple(getattr(trained_model, 'eval_exit_depths', ()) or ())
        if getattr(trained_model, 'fix_iter', None) is not None:
            depths = tuple(t for t in depths if t <= trained_model.fix_iter)
        if depths:
            save_exit_passes = list(depths)
    test_summary = {}
    for pass_index, save_exit_depth in enumerate(save_exit_passes):
        save_common = pass_index == len(save_exit_passes) - 1
        if save_exit_depth is not None:
            # 每轮相同种子，确保各出口比较的是相同缺失/噪声输入。
            seed_rng(config.hyp.seed)
            print(f"Testing/saving exit T{save_exit_depth} ({pass_index + 1}/{len(save_exit_passes)})")
        results_dict_test, min_ratio, max_ratio  = test(
            config, trained_model, test_data_loader, config.loss.test_loss_list, snr, psnr, ssim, state, device,
            save_exit_depth=save_exit_depth, save_common=save_common
        )

        # 每轮只合并小型标量指标；最后统一报告全部出口，不跨轮保留patch字典。
        test_summary.update({k: v for k, v in results_dict_test.items() if not isinstance(v, dict)})
        if save_common:
            results_dict_test.update(test_summary)
            # Log to wandb/swanlab
            # 补充逐出口三项指标；patch字典不能混入标量日志。
            test_log = {f"test_{k}": f"{v:.7f}" for k, v in results_dict_test.items()
                        if k in ['SNR', 'PSNR', 'SSIM', 'L1Loss', 'SSIMLoss', 'MSELoss']
                        or k.startswith(('SNR_T', 'PSNR_T', 'L1Loss_T'))}
            if run is not None:
                # Create log dict with prefix
                run.log(test_log)
            print(test_log)
        else:
            for key in test_summary:
                results_dict_test.pop(key, None)

        save_path = os.path.join(config.pt_path, 'test_'+time_now)
        print("test_save_path:", save_path)
        if config.save_test_plot:
            # 保存结果到pkl文件，用于后续重建
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 生成文件名，包含模型和时间戳
            exit_suffix = f"_exit_T{save_exit_depth}" if save_exit_depth is not None else ""
            pkl_filename = f"{config.dataset.name}_test_results_{config.model.name}{exit_suffix}.pkl"
            pkl_full_path = os.path.join(save_path, pkl_filename)
            print(f"Saving test results dictionary to {pkl_full_path}...")
            with open(pkl_full_path, 'wb') as f:
                pickle.dump(results_dict_test, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Results saved.")

            path_image = os.path.join(save_path, "image/")
            path_difference = os.path.join(save_path, "difference/")
            makedirs(path_image)
            makedirs(path_difference)
            metadata_path = os.path.join(config.dataset.data_dir, config.dataset.test_data_dir)

            data_path = os.path.join(root_path, f'{config.dataset.name}')
            print("data_path:", data_path)
            metadata_manager = PatchMetadataManager()
            metadata_manager.load_metadata(metadata_path)
            saved_exit_depths = ([save_exit_depth] if save_exit_depth is not None else
                                 list(getattr(trained_model, 'active_eval_exit_depths', ())) if config.hyp.save_exit_results else None)
            if config.dataset.name == 'MAVG':
                from data.MAVO import reconstruct
                reconstruct(data_path, metadata_manager, results_dict_test, patch_sizetime=256, patch_sizetrace=112, reconstruct_strategy=config.hyp.reconstruct_strategy, do_reconstruct_denorm=True, save_path=save_path, eval_exit_depths=saved_exit_depths, save_common=save_common)
            elif config.dataset.name == 'SEGC3':
                from data.SEGC3 import reconstruct
                reconstruct(data_path, metadata_manager, results_dict_test, patch_sizetime=128, patch_sizetrace=128, reconstruct_strategy=config.hyp.reconstruct_strategy, do_reconstruct_denorm=True, shot_n=config.hyp.shot_n, xline_idx=config.hyp.xline_idx, yline_idx=config.hyp.yline_idx, save_path=save_path, eval_exit_depths=saved_exit_depths, save_common=save_common)
            elif config.dataset.name == 'Model94':
                from data.MODEL94 import reconstruct
                reconstruct(data_path, metadata_manager, results_dict_test, patch_sizetime=128, patch_sizetrace=128, reconstruct_strategy=config.hyp.reconstruct_strategy, do_reconstruct_denorm=True, Cutting_time = [0, 1000], save_path=save_path, eval_exit_depths=saved_exit_depths, save_common=save_common)
            elif config.dataset.name == 'F3':
                from data.F3_Netherlands import reconstruct
                reconstruct(data_path, metadata_manager, results_dict_test, patch_sizetime=128, patch_sizetrace=128, reconstruct_strategy=config.hyp.reconstruct_strategy, do_reconstruct_denorm=True, save_path=save_path, eval_exit_depths=saved_exit_depths, save_common=save_common)

            # 直接使用内存字典，避免重读pkl产生第二份结果；本轮成功后再删除文件。
            if os.path.exists(pkl_full_path) and config.model.name != 'Curvelet':
                os.remove(pkl_full_path)
                print(f"Deleted temporary pkl file: {pkl_full_path}")
            del metadata_manager
        if save_exit_depth is not None:
            results_dict_test.clear()
            gc.collect()
    # Finish the run and upload any remaining data.
    if run is not None:
        run.finish()
if __name__ == '__main__':
    main()


