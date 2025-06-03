import jittor as jt
from jittor.dataset import DataLoader
from jittor.optim import LambdaLR, Adam

import tqdm
import time
import os
import json
import math

from core.load_data import get_dataset
from core.diffusion import Diffusion
from core.unet import UNet
from func import logger


default_config = {
    "data_root": "../../data",
    "dataset_name": "cifar10",
    
    "num_epochs": 500,
    "current_epoch": 0,
    "batch_size": 128,
    "num_workers": 12,
    "T": 1000,
    "lr": 1e-4,
    "max_norm": 1.0,
    "warmup": 5000,
    "beta_1": 0.0001,
    "beta_T": 0.02,

    "device": "cuda",

    "image_channels": 3,
    "n_channels": 128,
    "ch_mults": (1, 2, 2, 2),
    "is_attn": (False, True, True, False),
    "n_blocks": 2,

    "log": True,
    "log_root": None,
    "checkpoint": 5,
    "only_checkpoint_max": True,
}


class DDPM:
    """
    DDPM训练，使用UNet预测网络，DDPM物理概率模型，Adam优化器
    """
    def __init__(
        self,
        config: dict | str = default_config,
        model_path: str | None = None,
    ):
        self.config = json.load(open(config)) if isinstance(config, str) else config
        self.extract_config()
        jt.flags.use_cuda = True if self.device == "cuda" else False
        jt.flags.log_silent = True
        # ---------------
        # 模型定义
        # ---------------
        self.denoise = Diffusion(
            eps_model=UNet(
                image_channels=self.image_channels,
                n_channels=self.n_channels,
                ch_mults=self.ch_mults,
                is_attn=self.is_attn,
                n_blocks=self.n_blocks
            ),
            T=self.T,
            beta_1=self.beta_1,
            beta_T=self.beta_T
        )
        self.optimizer = Adam(
            self.denoise.eps_model.parameters(),
            lr=self.lr
        )
        self.lr_schedule = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, self.warmup))
        )
        if model_path is not None:
            state_dict = jt.load(model_path)
            self.denoise.eps_model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.logger.info(f"Model loaded from {model_path}")

    def train(self):
        """
        训练
        """
        dataset = get_dataset(self.data_root, self.dataset_name)
        self.logger.info(f"Dataset {self.dataset_name} loaded: {len(dataset)}")
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            buffer_size=2*1024*1024*1024
        )
        # log数据
        self.denoise.eps_model.train()
        epoch_list = []
        epoch_times_list = []
        epoch_loss_list = []
        steps = 0
        
        while self.current_epoch < self.num_epochs:
            self.current_epoch += 1
            batch_len = math.ceil(len(dataset) / self.batch_size)
            progress_bar = tqdm.tqdm(
                train_loader,
                total=batch_len,
                desc=f"Epoch {self.current_epoch}"
            )
            # log数据
            total_loss = 0.0
            step_list = []
            step_loss_list = []
            epoch_start = time.time()
            
            for i, (x, _) in enumerate(progress_bar):
                self.optimizer.zero_grad()
                loss = self.denoise.loss(x)
                self.optimizer.backward(loss)
                if self.max_norm > 0:
                    self.optimizer.clip_grad_norm(max_norm=self.max_norm)
                self.optimizer.step(loss)
                self.lr_schedule.step()
                # logging
                step_loss = loss.item()
                step_list.append(steps)
                step_loss_list.append(step_loss)
                total_loss += step_loss
                self.logger.info(f"Epoch {self.current_epoch} Step {i+1} - Loss: {step_loss:.4f}")
                progress_bar.set_postfix(loss=total_loss / (i+1))
                steps += 1
            
            # logging
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / batch_len
            self.loss = avg_loss
            self.min_loss = min(self.min_loss, avg_loss) if self.min_loss is not None else avg_loss
            epoch_loss_list.append(avg_loss)
            epoch_times_list.append(epoch_time)
            epoch_list.append(self.current_epoch)
            self.logger.info(f"Epoch {self.current_epoch} completed - time cost: {epoch_time:.2f}s")
            self.logger.info(f"Average loss: {avg_loss:.4f}\n")
            if len(epoch_list) >= 20:
                self.logger.plot_loss(epoch_list[-20:], epoch_loss_list[-20:], filename=f"epoch_{epoch_list[-1]//20*20}_loss.png")
            self.logger.plot_loss(step_list, step_loss_list, filename=f"step_loss.png" if self.only_checkpoint_max else None)
            self.logger.plot_loss(epoch_list, epoch_loss_list, "epoch_loss.png")
            self.logger.plot_epoch_time(epoch_list, epoch_times_list, "epoch_time.png")
            if self.loss == self.min_loss:
                self.save_model(pkl_name=f"eps_best.pkl", config_name=f"eps_best.json")
            if self.checkpoint > 0 and self.current_epoch % self.checkpoint == 0:
                self.save_model(
                    pkl_name=f"eps_current.pkl" if self.only_checkpoint_max else None,
                    config_name=f"eps_current.json" if self.only_checkpoint_max else None
                )
            
    def save_model(
        self, 
        save_root: str | None = None,
        pkl_name: str | None = None,
        config_name: str | None = None
    ):
        """
        保存模型
        Args:
            save_path: 保存路径，默认为checkpoints/{self.timestamp}
        """
        if save_root is None:
            save_root = f"checkpoints/{self.timestamp}"
        if os.path.exists(save_root) == False:
            os.makedirs(save_root)
        if pkl_name is None:
            pkl_name = f"eps_{self.current_epoch}_epoch.pkl"
        if config_name is None:
            config_name = f"eps_{self.current_epoch}_epoch.json"
        save_path = os.path.join(save_root, pkl_name)
        self.unextract_config()
        json.dump(self.config, open(os.path.join(save_root, config_name), "w"), indent=4)
        jt.save({
            'model_state_dict': self.denoise.eps_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")
        
    def rand_sample_x0(
        self,
        batch_size: int = 16,
        show_denoised: int = 16
    ):
        """
        随机采样生成x0
        Args:
            num_samples: 采样数量
            batch_size: 批大小
            show_denoised: 展示降噪结果数量
        Return:
            xs[b, t, c, h, w]
        """
        self.denoise.eps_model.eval()
        with jt.no_grad():
            self.logger.info(f"Sampling {batch_size} x0...")
            time_start = time.time()
            xs = [] # 降噪结果
            x_t = jt.randn(batch_size, self.image_channels, 32, 32)
            for t in reversed(range(0, self.T)):
                bt = jt.full((batch_size,), t, dtype=jt.int32)
                x_t = self.denoise.p_sample(x_t, bt)
                if show_denoised > 0 and t % (self.T // show_denoised) == 0:
                    xs.append(((x_t + 1) / 2 * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu())
            time_end = time.time()
            self.logger.info(f"Time cost: {time_end - time_start:.2f}s")
            return xs
    
    def rand_sample_x0_ddim(
        self,
        batch_size: int = 16,
        show_denoised: int = 16,
        num_steps: int = 50,
        eta: float = 1.0
    ):
        """
        DDIM 采样循环
        Args:
            num_samples: 采样数量
            batch_size: 批大小
            show_denoised: 展示降噪结果数量
            num_steps: 采样步数 (远小于总步数 T)
            eta: 随机性参数 (0=确定性)
        """
        self.denoise.eps_model.eval()
        step_size = (self.T - 1) // (num_steps - 1)
        time_steps = list(range(0, self.T - 1, step_size))[:num_steps] + [self.T - 1]
        with jt.no_grad():
            self.logger.info(f"Sampling {batch_size} x0 with {num_steps} steps...")
            time_start = time.time()
            xs = [] # 降噪结果
            x_t = jt.randn(batch_size, self.image_channels, 32, 32)
            t_next = jt.full((batch_size,), time_steps.pop(), dtype=jt.int32)
            while len(time_steps) > 0:
                t_prev = jt.full((batch_size,), time_steps.pop(), dtype=jt.int32)
                x_t = self.denoise.p_sample_ddim(x_t, t_next, t_prev, eta)
                t_next = t_prev
                if show_denoised > 0 and len(time_steps) % (num_steps // show_denoised) == 0:
                    xs.append(((x_t + 1) / 2 * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu())
            time_end = time.time()
            self.logger.info(f"Time cost: {time_end - time_start:.2f}s")
            return xs

    def extract_config(self):
        """
        生效配置self.config
        """
        # ---------------
        # 日志配置
        # ---------------
        self.logger = logger.Logger(self.config['log_root'])
        self.logger.is_logging = self.config['log']  # 是否开启日志
        self.checkpoint = self.config["checkpoint"] # 设置检查点，每隔checkpoint个epoch保存一次模型，0时为不保存模型
        self.only_checkpoint_max = self.config["only_checkpoint_max"] # 是否只保存最大模型
        self.loss = self.config.get("loss", None) # 记录训练损失
        self.min_loss = self.config.get("min_loss", None) # 记录最小训练损失
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")  # 时间戳
        # ---------------
        # 数据处理参数
        # ---------------
        self.data_root = self.config['data_root']  # 数据集位置
        self.dataset_name = self.config['dataset_name']  # 数据集名称
        self.batch_size = self.config['batch_size']  # 批大小
        self.num_workers = self.config['num_workers']  # 线程数
        # ---------------
        # 训练参数
        # ---------------
        self.device = self.config['device']
        self.T = self.config['T']  # 扩散步数
        self.lr = self.config['lr']  # 学习率
        self.num_epochs = self.config['num_epochs']  # epoch数
        self.current_epoch = self.config['current_epoch'] # 当前epoch
        self.max_norm = self.config['max_norm'] # 最大梯度范数
        self.warmup = self.config['warmup'] # 预热步数
        self.beta_1 = self.config['beta_1']
        self.beta_T = self.config['beta_T']
        # UNet参数
        self.image_channels = self.config['image_channels']
        self.n_channels = self.config['n_channels']
        self.ch_mults = self.config['ch_mults']
        self.is_attn = self.config['is_attn']
        self.n_blocks = self.config['n_blocks']
    
    def unextract_config(self):
        """
        保存配置到self.config
        """
        self.config.update({
            "data_root": self.data_root,
            "dataset_name": self.dataset_name,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "T": self.T,
            "lr": self.lr,
            "beta_1": self.beta_1,
            "beta_T": self.beta_T,
            "num_epochs": self.num_epochs,
            "current_epoch": self.current_epoch,
            "max_norm": self.max_norm,
            "warmup": self.warmup,
            "image_channels": self.image_channels,
            "n_channels": self.n_channels,
            "ch_mults": self.ch_mults,
            "is_attn": self.is_attn,
            "n_blocks": self.n_blocks,
            "log": self.logger.is_logging,
            "checkpoint": self.checkpoint,
            "only_checkpoint_max": self.only_checkpoint_max,
            "loss": self.loss if self.loss is not None else "None",
            "min_loss": self.min_loss if self.min_loss is not None else "None",
        })
