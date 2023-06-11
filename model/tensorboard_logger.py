import tensorboardX
import numpy as np


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tensorboardX.SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, scalar_value=value, global_step =step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args::images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """
        for i, j in enumerate(images):
            img = ((j*0.5+0.5)*255).round().astype('uint8')
            if len(img.shape) == 3:
                pass
                # print('1', img.shape)
            else:
                # print('2_0', img.shape)
                img = img[np.newaxis, :, :]
                # print('2_1',img.shape)
            # print('3', img.shape)
            self.writer.add_image('{}--{}'.format(tag,i), img, global_step=step)


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram('{}'.format(tag), values=values, bins=bins, global_step=step)
