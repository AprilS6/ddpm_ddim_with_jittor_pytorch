"""
训练脚本:
    训练新的模型，或加载预训练模型继续训练。选项可以指定部分参数，完整参数可以通过json文件指定。
    所有选项均有默认参数，可以直接运行脚本，会以当前时间戳从以默认配置训练一个新模型。
命令行：
    python -m script.train [--<option>=<value>]
选项:
    --model_path: 预训练模型路径（可加载预训练模型继续训练）
    --config_path: 配置文件路径（可以直接指定配置文件，优先级低于命令行参数）
    --data_root: 数据集根目录（默认为../../data/）
    --dataset_name: 数据集名称（默认为cifar10）
    --num_epochs: 训练轮数（默认为500）
    --batch_size: 批大小（默认为128）
    --num_workers: 线程数（默认为4）
    --T: 时间步数（默认为1000）
    --lr: 学习率（默认为1e-4）
    --device: 设备类型（优先为cuda）
    --log: 是否记录日志（默认为True）
    --log_root: 日志根目录（默认为logs/）
    --checkpoint: 检查点间隔，epoch到检查点后备份一次模型（默认为5）
    --only_checkpoint_max: 只保留最新的检查点（默认为True）
"""
import argparse
import json

from script.model import DDPM, default_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--T', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device', type=str)
    parser.add_argument('--log', type=bool)
    parser.add_argument('--log_root', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--only_checkpoint_max', type=bool)
    args = parser.parse_args()
    config = default_config.copy() if args.config_path is None else json.load(open(args.config_path))
    config.update([(k, v) for k, v in vars(args).items() if v is not None])
    model = DDPM(config, args.model_path)
    model.train()
    
if __name__ == '__main__':
    main()