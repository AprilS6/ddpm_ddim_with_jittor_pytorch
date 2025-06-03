import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import time
from tqdm import tqdm

def main():
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置批处理大小（根据GPU内存调整）
    batch_size = 256  # 建议值：16GB显存用256，8GB显存用128
    
    # 加载生成数据
    start_time = time.time()
    print("Loading generated data...")
    gen_data = np.load('jittor_ver/samples/ddpm_steps.npy')
    gen_data = torch.from_numpy(gen_data).permute(0, 3, 1, 2)
    gen_data = gen_data.to(torch.uint8).to(device)
    gen_loader = torch.utils.data.DataLoader(gen_data, batch_size=batch_size)
    print(f"Generated data loaded in {time.time()-start_time:.2f}s, shape: {gen_data.shape}")

    # 加载CIFAR-10测试集
    start_time = time.time()
    print("Loading CIFAR-10 data...")
    cifar10_test = CIFAR10(root='../../data', train=False, download=True, transform=ToTensor())
    cifar_images = torch.stack([img for img, _ in cifar10_test][:1000])
    cifar_images = (cifar_images * 255).to(torch.uint8).to(device)
    cifar_loader = torch.utils.data.DataLoader(cifar_images, batch_size=batch_size)
    print(f"CIFAR-10 data loaded in {time.time()-start_time:.2f}s, shape: {cifar_images.shape}")

    # 初始化FID计算器并移动到GPU
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(device)
    
    # GPU加速的真实数据特征提取
    print("Extracting real features on GPU...")
    with torch.no_grad():
        for batch in tqdm(cifar_loader, desc="Processing real images"):
            fid.update(batch.to(device), real=True)
    
    # GPU加速的生成数据特征提取
    print("Extracting generated features on GPU...")
    with torch.no_grad():
        for batch in tqdm(gen_loader, desc="Processing generated images"):
            fid.update(batch.to(device), real=False)

    # 计算FID分数
    start_time = time.time()
    result = fid.compute()
    print(f"FID computation took {time.time()-start_time:.2f}s")
    print(f"FID score: {result.item():.4f}")
    
    
if __name__ == '__main__':
    main()