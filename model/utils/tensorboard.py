#models/utils.py

from tensorboardX import SummaryWriter

def get_tensorboard_writer(log_dir='../logs/fit'):
    writer = SummaryWriter(log_dir)
    return writer 





