import torch
from torch.autograd import Variable
import numpy as np

if __name__ == '__main__':

    num_threads = torch.get_num_threads()
    print("num_threads {}\n".format(num_threads))

    x = Variable(torch.randn(1, 1), requires_grad=True)
    with torch.autograd.profiler.profile() as prof:
        y = x ** 2
        y.backward()
    # NOTE: some columns were removed for brevity
    print(prof)




