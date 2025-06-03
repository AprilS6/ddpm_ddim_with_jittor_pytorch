import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class Diffusion:
    """
    DDPM的物理概率模型，定义扩散的物理过程并提供噪声/去噪公式q和p
    Params:
        beta: 超参数，定义加噪调度。这里使用线性调度，另有余弦调度（适合更高分辨率图像，
            防止后期信息丢失太快）和平方调度（适合快速训练但效果一般）
    """
    def __init__(
            self,
            eps_model: nn.Module,
            T: int,
            device: torch.device | str,
            beta_1: float = 0.0001,
            beta_T: float = 0.02
        ):
        """
        Args:
            eps_model: epsilon_theta. UNet去噪模型
            T: 总时间步，[0, T-1]
            device: 训练用硬件
        """
        self.eps_model = eps_model
        self.beta = torch.linspace(beta_1, beta_T, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).clamp(min=1e-10)
        self.T = T
        # -------------------------------------------------------------------
        # sigma2是alpha和alpha_bar的表达式，但ddpm论文提到直接使用beta拥有
        # 相似的效果，还有论文是学习方差而非使用固定方差
        # -------------------------------------------------------------------
        self.sigma2 = self.beta
        
    def q_xt_x0(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        即重参数化后得到的公式q(x_t|x_0)，返回的是x_t对应高斯分布的均值和方差
        Args:
            x_0: 初始数据（时间步为0的数据）
            t: 经过t步加噪
        Return:
            mean: x_t的均值
            var: x_t的方差
        """
        alpha_bar = torch.gather(self.alpha_bar, index=t, dim=0)[:, None, None, None]
        mean = (alpha_bar ** 0.5) * x_0
        var = 1.0 - alpha_bar
        return mean, var
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None):
        """
        由q(x_t|x_0)得出x_t对应高斯分布后，通过eps采样得到x_t
        Args:
            x_0: 初始数据（时间步为0的数据）
            t: 经过t步加噪
            eps: 形状与x_0相同的采样，默认为标准高斯分布随机采样
        Return:
            x_t: t步加噪数据（时间步为t的数据）
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        mean, var = self.q_xt_x0(x_0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        使用eps_model预测噪声并去噪，即p_theta(x_{t-1}|x_t)
        Args:
            x_t: 待去噪数据（时间步为t的数据）
            t: 当前数据x_t的时间步
        Return:
            x_{t-1} 去噪结果
        """
        # 求均值mu，其为eps_theta, x_t, t的关系式
        eps_theta = self.eps_model(x_t, t)
        alpha_bar = torch.gather(self.alpha_bar, index=t, dim=0)[:, None, None, None]
        alpha = torch.gather(self.alpha, index=t, dim=0)[:, None, None, None]
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (x_t - eps_coef * eps_theta)
        # 方差sigma2为固定值sigma2_t
        var = torch.gather(self.sigma2, index=t, dim=0)[:, None, None, None]
        eps = torch.randn(x_t.shape, device=x_t.device)
        return mean + (var ** 0.5) * eps

    def p_sample_ddim(self, x_t: torch.Tensor, t: torch.Tensor, prev_t: torch.Tensor, eta: float = 0.0):
        """
        DDIM采样
        Args:
            x_t: 当前时刻t的噪声图像 -- (b, c, h, w)
            t: 当前时间步 -- (b)
            prev_t: 下一个时间步 -- (b)
            eta: 控制随机性的参数，默认为0
        """
        eps_theta = self.eps_model(x_t, t)
        alpha_bar_t = self.alpha_bar.gather(0, t).reshape((-1, 1, 1, 1))
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t + 1e-10)
        alpha_bar_prev = self.alpha_bar.gather(0, prev_t).reshape((-1, 1, 1, 1))
        if eta > 0:
            term1 = (1 - alpha_bar_prev) / (1 - alpha_bar_t + 1e-10)
            term2 = 1 - alpha_bar_t / (alpha_bar_prev + 1e-10)
            sigma_t = eta * torch.sqrt(term1 * term2)
        else:
            sigma_t = torch.zeros_like(alpha_bar_prev)
        coef_eps = torch.sqrt(1 - alpha_bar_prev - sigma_t**2)
        mean = torch.sqrt(alpha_bar_prev) * x0_pred + coef_eps * eps_theta
        if eta > 0:
            noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)
            return mean + sigma_t * noise
        else:
            return mean
    
    def loss(self, x_0: torch.Tensor, noise: torch.Tensor | None = None):
        """
        损失函数，此处设置为epsilon与epsilon_theta的MSE
        Args:
            x0: 来自训练数据的干净的图片
            noise: 加噪过程噪声epsilon~N(0, I)
        Return:
            loss: 真实噪声和预测噪声之间的损失
        """
        batch_size = x_0.shape[0]
        # 随机抽样t，加噪声noise得到x_t，通过eps_model预测噪声为eps_theta
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps=noise)
        eps_theta = self.eps_model(x_t, t)
        return F.mse_loss(noise, eps_theta)