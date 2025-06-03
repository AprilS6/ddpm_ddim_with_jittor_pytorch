import math
import torch
from torch import nn
from typing import Tuple, List

class TimeEmbedding(nn.Module):
    """
    t(b,) -pe-> emb_t(b, c//4) -mlp-> emb_t(b, c)
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.n_channels // 4, self.n_channels),
            nn.SiLU(),
            nn.Linear(self.n_channels, self.n_channels)
        )

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8 # dim为Positional Embedding的维度
        emb = torch.exp(torch.arange(half_dim, device=t.device) * (- math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.mlp(emb) # .clamp(emb, -1.0 + 1e-6, 1.0 - 1e-6)
        return emb

class ResidualBlock(nn.Module):
    """
    x(b, c, h, w) -merge_temb(b, c)-> x_t(b, c', h, w) -merge_x'(b, c', h, w)-> x_t'(b, c', h, w)
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_channels: int,
            n_groups: int = 32,
            dropout: float = 0.1
        ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)) # c->c'
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)) # 特征整理

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity() # c->c'，对齐
        # 传入的时间步特征已经提取，这里只需要对齐维度即可，相当于MLP后缀
        self.time_act = nn.SiLU()
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x))) # c->c'
        h += self.time_emb(self.time_act(t))[:, :, None, None] # (b, c)->(b, c', 1, 1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h)))) # (b, c', h, w)->(b, c', h, w)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    x(b, c, h, w) -flatt-> x(b, h*w, c) -proj_view-> qkv(b, h*w, n_heads, 3*d_k) -chunk-> q,k,v(b, h*w, n_heads, d_k)
     -attn-> (b, h*w, h*w, n_heads),v -merge_reshape-> (b, h*w, n_heads*d_k) -output-> (b, h*w, c)
     -merge_x(b, h*w, c)-> x'(b, h*w, c) -reshape-> x(b, c, h, w)
    """
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int | None = None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels # 默认q k v维度d_k=c
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.proj = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        _ = t
        # 展平（合并H和W）
        batch_size, n_channels, height, width = x.shape
        x = self.norm(x)
        x= x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # 对齐（方便切分）
        qkv = self.proj(x).view(batch_size, -1, self.n_heads, 3 * self.d_k) # (b, h*w, n_heads, 3*d_k)
        # 切分为3个(b, h*w, n_heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 求token间注意力权重，并softmax归一化
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        # 用注意力权重求加权和，得到最终每个token权重
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k) # reshape
        res = self.output(res) # n_heads*c->c
        res += x # 残差连接
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """
    整合ResidualBlock和AttentionBlock
    x(b, c, h, w) -res-> x_t(b, c', h, w) -attn-> x_t'(b, c', h, w)
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    整合ResidualBlock和AttentionBlock
    x(b, c, h, w) -res-> x_t(b, c', h, w) -attn-> x_t'(b, c', h, w)
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int, n_blocks: int = 1):
        super().__init__()
        blks = []
        for _ in range(n_blocks):
            blks.append(ResidualBlock(n_channels, n_channels, time_channels))
            blks.append(AttentionBlock(n_channels))
        self.blks = nn.ModuleList(blks)
        self.res = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for m in self.blks:
            x = m(x, t)
        x = self.res(x, t)
        return x


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(
            self, 
            image_channels: int = 3, 
            n_channels: int = 128,
            ch_mults: Tuple[int, ...] | List[int] = (1, 2, 2, 2),
            is_attn: Tuple[bool, ...] | List[bool] = (False, True, True, False),
            n_blocks: int = 2
        ):
        """
        Params:
            image_channels: 输入图像的通道数
            ch_mults: 通道倍数
            is_attn: 每一层是否使用注意力机制
            n_blocks: 每一层的ResidualBlock个数
        """
        super().__init__()
        n_resolutions = len(ch_mults)
        time_channels = n_channels * 4
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(time_channels)

        # --------------------------------------------------------------
        # DownBlock: ResAtnBlock整理特征，此时H,W不变，通道数可变
        # Downsample: 用step为2的卷积核将H,W压缩一半，通道数不变
        # --------------------------------------------------------------
        down = []
        in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, time_channels, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(in_channels, time_channels)
        # --------------------------------------------------------------
        # UpBlock: 比DownBlock多一个UpBlock负责与skip block连接补全分辨率特征
        # Upsample: 用step为2的转置卷积核将H,W扩大一倍，通道数不变
        # --------------------------------------------------------------
        up = []
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels // ch_mults[i]
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels + in_channels, in_channels, time_channels, is_attn[i]))
            up.append(UpBlock(in_channels + out_channels, out_channels, time_channels, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(32, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(n_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)
        skips = [x]
        for m in self.down:
            x = m(x, t)
            skips.append(x)
        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, Upsample) == False:
                x = torch.cat((x, skips.pop()), dim=1)
            x = m(x, t)
        return self.final(self.act(self.norm(x)))