import matplotlib.pyplot as plt
import logging
import time
import os

class Logger:
    def __init__(self, root: str | None = None):
        """
        Args:
            root: 日志根目录，默认为logs/{timestamp}
        """
        if root is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            root = f"logs/{timestamp}"
        if not os.path.exists(root):
            os.makedirs(root)
        self.root = root
        self.is_logging  = True
        log_file = os.path.join(self.root, "log.txt")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # 文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
    def info(self, message: str):
        if self.is_logging:
            self.logger.info(message)
    
    def setOn(self):
        """
        开启日志记录
        """
        self.is_logging = True
            
    def setOff(self):
        """
        关闭日志记录
        """
        self.is_logging = False
        
    def plot_loss(self, epoch_list: list, loss_list: list, filename: str | None = None):
        """
        绘制loss曲线
        Args:
            epoch_list: 对应的epoch
            loss_list: 需要绘制的点集
            filename: 保存的文件名，默认为loss_{epoch_list[-1]}_epoch.png
        """
        if self.is_logging == False:
            return
        if filename is None:
            filename = f"loss_{epoch_list[-1]}_epoch.png"
        save_path = os.path.join(self.root, filename)
        plt.plot(epoch_list, loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (Avg Loss= {sum(loss_list)/len(loss_list):.4f})')
        plt.savefig(save_path)
        plt.close()
        
    def plot_epoch_time(self, epoch_list: list, epoch_time_list: list, filename: str | None = None):
        """
        绘制epoch时间曲线
        Args:
            epoch_list: 对应的epoch
            epoch_time_list: 对应的epoch时间
            filename: 保存的文件名，默认为epoch_time.png
        """
        if self.is_logging == False:
            return
        if filename is None:
            filename = f"epoch_time.png"
        save_path = os.path.join(self.root, filename)
        plt.plot(epoch_list, epoch_time_list)
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title(f'Epoch Time (Avg Time= {sum(epoch_time_list)/len(epoch_time_list):.4f}s)')
        plt.savefig(save_path)
        plt.close()