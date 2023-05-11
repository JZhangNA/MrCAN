import torch
import torch.nn.functional as F

class context_embedding(torch.jit.ScriptModule):
    def __init__(self, in_channels=1, embedding_size=256, k=5, dilation=1):
        super(self.__class__, self).__init__()

        self.__padding = (k - 1) * dilation

        self.causal_convolution = torch.nn.Conv1d(in_channels=in_channels, out_channels=embedding_size, kernel_size=k)

    def forward(self, x):
        """
        :param x = [batch_size, d_model=in_channels, time_step]
        :return:
            x = [batch_size, embedding_size, time_step]
        """
        x = F.pad(x, (self.__padding, 0))
        x = self.causal_convolution(x)
        return F.tanh(x)