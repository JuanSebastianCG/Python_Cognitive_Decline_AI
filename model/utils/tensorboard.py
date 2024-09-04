#models/utils.py

from tensorboardX import SummaryWriter

def get_tensorboard_writer(log_dir='../logs/fit'):
    """
    Creates and returns a TensorBoard SummaryWriter object.

    Parameters:
    log_dir (str): The directory where the TensorBoard logs will be saved. 
                   Default is '../logs/fit'.

    Returns:
    SummaryWriter: A TensorBoard SummaryWriter object for logging.
    """
    writer = SummaryWriter(log_dir)
    return writer





