import math
import jittor as jt
from jittor import nn
from typing import Tuple, List

class TimeEmbedding(nn.Module):
    """
    t(b,) -pe-> emb_t(b, c//4) -mlp-> emb_t(b, c)
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(n_channels // 4, self.n_channels),
            nn.SiLU(),
            nn.Linear(self.n_channels, self.n_channels)
        )
    
    def execute(self, t: jt.Var) -> jt.Var:
        # 固定编码部分
        half_dim = self.n_channels // 8 # dim为Positional Embedding的维度
        emb = jt.exp(jt.arange(half_dim) * (- math.log(10000) / (half_dim - 1))) # 正余弦位置编码（提取周期性特征）
        emb = t[:, None] * emb[None, :] # 编码
        emb = jt.cat((emb.sin(), emb.cos()), dim=1) # 将cos部分和sin部分展平，得到位置编码
        emb = self.mlp(emb).clamp(min_v=-1.0 + 1e-6, max_v=1.0 - 1e-6) # MLP进一步提取可学习特征，以及调整维度
        return emb


class ResBlock(nn.Module):
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels!= out_channels else nn.Identity()
        self.time_act = nn.SiLU()
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x: jt.Var, emb_t: jt.Var):
        h = self.conv1(self.act1(self.norm1(x))) # 提取特征 -- (b, c, h, w)->(b, c', h, w)
        h += self.time_emb(self.time_act(emb_t))[:, :, None, None] # 并入时间编码 -- (b, c)->(b, c', 1, 1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h)))) # 整合特征 -- (b, c', h, w)-->(b, c', h, w)
        return h + self.shortcut(x) # 防止上层信息丢失
    

class AtnBlock(nn.Module):
    """
    x(b, c, h, w) -flatt-> x(b, h*w, c) -proj_view-> qkv(b, h*w, n_heads, 3*k_dim) -chunk-> q,k,v(b, h*w, n_heads, k_dim)
     -attn-> (b, h*w, h*w, n_heads),v -merge_reshape-> (b, h*w, n_heads*k_dim) -output-> (b, h*w, c)
     -merge_x(b, h*w, c)-> x'(b, h*w, c) -reshape-> x(b, c, h, w)
    """
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        k_dim: int | None = None,
        n_groups: int = 32
    ):
        super().__init__()
        if k_dim is None:
            k_dim = n_channels # q, k, v的维度
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.proj = nn.Linear(n_channels, n_heads * k_dim * 3)
        self.output = nn.Linear(n_heads * k_dim, n_channels)
        self.scale = k_dim ** -0.5
        self.n_heads = n_heads
        self.k_dim = k_dim

    def execute(self, x: jt.Var, emb_t: jt.Var | None = None):
        _ = emb_t
        # q, k, v提取
        batch_size, n_channels, height, width = x.shape
        x = self.norm(x) # 归一化 -- (b, c, h, w)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1) # 展平，合并hw -- (b, h*w, c)
        qkv = self.proj(x).view(batch_size, -1, self.n_heads, 3 * self.k_dim) # 维度对齐，方便切分qkv -- (b, h*w, n_heads, 3*k_dim)
        q, k, v = jt.chunk(qkv, 3, dim=-1) # 切分，得到每个元素的qkv -- (b, h*w, n_heads, k_dim)
        # 注意力权重计算
        q = q.permute(0, 2, 1, 3) # (b, n_heads, i, k_dim)
        k = k.permute(0, 2, 3, 1) # (b, n_heads, k_dim, j)
        v = v.permute(0, 2, 1, 3) # (b, n_heads, j, k_dim)
        attn = jt.matmul(q, k) * self.scale # 权重矩阵q*k -- (b, n_heads, i, j)
        attn = attn.softmax(dim=-1) # 归一化 -- (b, n_heads, i, j)
        # 查询影响结果
        res = jt.matmul(attn, v) # attn*v -- (b, n_heads, i, k_dim)
        res = res.permute(0, 2, 1, 3).view(batch_size, -1, self.n_heads * self.k_dim) # 展平 -- (b, h*w, n_heads*k_dim)
        res = self.output(res) # (b, h*w, c)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width) # (b, c, h, w)
        return res
    

class DownBlock(nn.Module):
    """
    整合ResidualBlock和AttentionBlock
    x(b, c, h, w) -res-> x_t(b, c', h, w) -attn-> x_t'(b, c', h, w)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        n_groups: int = 32
    ):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_channels, n_groups)
        self.attn = AtnBlock(out_channels, n_groups=n_groups) if has_attn else nn.Identity()

    def execute(self, x: jt.Var, emb_t: jt.Var):
        x = self.res(x, emb_t)
        x = self.attn(x)
        return x
    

class UpBlock(nn.Module):
    """
    整合ResidualBlock和AttentionBlock
    x(b, c, h, w) -res-> x_t(b, c', h, w) -attn-> x_t'(b, c', h, w)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        n_groups: int = 32
    ):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_channels, n_groups)
        self.attn = AtnBlock(out_channels, n_groups=n_groups) if has_attn else nn.Identity()

    def execute(self, x: jt.Var, emb_t: jt.Var):
        x = self.res(x, emb_t)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """
    整合ResidualBlock和AttentionBlock
    x(b, c, h, w) -res-> x_t(b, c, h, w) -attn-> x_t'(b, c, h, w)
    """
    def __init__(
        self,
        n_channels: int,
        time_channels: int,
        n_blocks: int = 1,
        n_groups: int = 32
    ):
        super().__init__()
        blks = []
        for _ in range(n_blocks):
            blks.append(ResBlock(n_channels, n_channels, time_channels, n_groups=n_groups))
            blks.append(AtnBlock(n_channels, n_groups=n_groups))
        blks.append(ResBlock(n_channels, n_channels, time_channels, n_groups=n_groups))
        self.blks = nn.ModuleList(blks)
    
    def execute(self, x: jt.Var, emb_t: jt.Var):
        for blk in self.blks:
            x = blk(x, emb_t)
        return x

class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=1)
    
    def execute(self, x: jt.Var, emb_t: jt.Var | None = None):
        _ = emb_t
        return self.conv(x)
    

class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, 4, stride=2, padding=1)
        
    def execute(self, x: jt.Var, emb_t: jt.Var | None = None):
        _ = emb_t
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 128,
        ch_mults: Tuple[int, ...] | List[int] = (1, 2, 2, 2),
        is_attn: Tuple[bool, ...] | List[bool] = (False, True, True, False),
        n_blocks: int = 2,
        n_groups: int = 32
    ):
        super().__init__()
        n_resolutions = len(ch_mults)
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels)
        self.img_proj = nn.Conv2d(image_channels, n_channels, 3, padding=1)
        # --------------------------------------------------------------
        # DownBlock: ResAtnBlock整理特征，此时H,W不变，通道数可变
        # Downsample: 用step为2的卷积核将H,W压缩一半，通道数不变
        # n_resolutions层，每层先做n_blocks个ResAtnBlock，第一个ResAtnBlock通道数乘以ch_mults[i]，
        # 再Downsample，最后一层不做Downsample
        # --------------------------------------------------------------
        down = []
        in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, time_channels, is_attn[i], n_groups=n_groups))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))
        self.down = nn.ModuleList(down)
        # --------------------------------------------------------------
        # MidBlock: 多层ResAtnBlock，此时H,W不变，通道数不变
        # --------------------------------------------------------------
        self.middle = MidBlock(in_channels, time_channels, n_groups=n_groups)
        # --------------------------------------------------------------
        # UpBlock: 比DownBlock多一个UpBlock负责与skip block连接补全分辨率特征
        # Upsample: 用step为2的转置卷积核将H,W扩大一倍，通道数不变
        # n_resolutions层，每层先做n_blocks+1个ResAtnBlock，
        # 前n_blocks个与对应层倒数n-1个DownBlock/DownSample的输入进行skip connection，
        # 最后一个与对应层第一个ResAtnBlock的输入进行skip connection，并将通道数缩减为原来的1/ch_mults[i]
        # up部分的UpBlock与down部分的DownBlock/DownSample是栈的FILO对应关系
        # --------------------------------------------------------------
        up = []
        for i in reversed(range(n_resolutions)):
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels + in_channels, in_channels, time_channels, is_attn[i], n_groups=n_groups))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels + out_channels, out_channels, time_channels, is_attn[i], n_groups=n_groups))
            in_channels = out_channels
            if i > 0:
                up.append(UpSample(in_channels))
        self.up = nn.ModuleList(up)
        
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(n_channels, image_channels, 3, padding=1)
        
    
    def execute(self, x: jt.Var, t: jt.Var):
        t = self.time_emb(t)
        x = self.img_proj(x)
        skips = [x] # 第一次DownBlock的输入
        for m in self.down:
            x = m(x, t)
            skips.append(x) # 每一个块的输入(包括DownSample)
        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, UpSample) == False:
                x = jt.concat([x, skips.pop()], dim=1) # 除UpSample外其它块都做skip connection
            x = m(x, t)
        return self.final(self.act(self.norm(x)))