from tensorboardX import SummaryWriter
def get_tensorboard_writer(log_dir='./model/logs/fit'):
    writer = SummaryWriter(log_dir)
    return writer 