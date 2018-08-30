import torch
import torchvision
import numpy as np
import torch.nn as nn
import visdom
from torch.utils.data import DataLoader
from config import VISDOM_SERVER, VISDOM_PORT
from snippets.modules import MLP
from snippets.scaffold import TrainLoop, TestLoop, sort_gpu_index
import matplotlib.pyplot as plt
import PIL.Image

print("available devices count:", torch.cuda.device_count())

transform = lambda __x: torch.from_numpy(np.asarray(__x, dtype=np.float32))
mnist_train_data = torchvision.datasets.MNIST("/tmp/mnist_data", train=True, download=True, transform=transform)
mnist_test_data = torchvision.datasets.MNIST("/tmp/mnist_data", train=False, download=True, transform=transform)


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._mlp = MLP(input_size=np.prod(input_size), net_sizes=[100, 100, 10], activation="leakyrelu")

    def forward(self, __x):
        __x = __x.view(__x.size()[0], -1)
        __x = __x / 255
        __x = self._mlp(__x)
        return __x


torch.cuda.set_device(sort_gpu_index()[0])
mnist_train_dataloader = DataLoader(mnist_train_data,
                                    batch_size=1024,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=16,
                                    pin_memory=True
                                    )
mnist_test_dataloader = DataLoader(mnist_test_data,
                                   batch_size=1024,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=16,
                                   pin_memory=True
                                   )
model = Model(mnist_train_data[0][0].size())
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env="Snippets Mnist Example")
with TrainLoop(max_epochs=10, disp_step_freq=100, disp_epoch_freq=None).with_context() as loop:
    for epoch in loop.iter_epochs():
        for _, (x, y) in loop.iter_steps(mnist_train_dataloader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loop.submit_metric("train_loss", loss.detach().cpu().numpy())
        with TestLoop(print_fn=None).with_context() as test_loop:
            for _, (x, y) in test_loop.iter_steps(mnist_test_dataloader):
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                logits = model(x)
                y_pred = torch.argmax(logits, dim=-1)
                test_loop.submit_data("correct_count", np.asscalar(torch.sum(y_pred == y).cpu().numpy()))
                test_loop.submit_data("total_count", np.asscalar(np.prod(y.size())))
                loop.submit_metric("test_loss", loss_fn(logits, y).item())
            correct_count = test_loop.get_data("correct_count", step="all")
            total_count = test_loop.get_data("total_count", step="all")
            loop.submit_metric("test_accuracy",
                               sum(correct_count) / sum(total_count))
            vis.line(Y=np.asarray([loop.get_metric("test_accuracy", step="last")]),
                     X=np.asarray([epoch]),
                     update="append" if epoch > 1 else False,
                     win="MNIST by MLP Test Accuracy",
                     opts=dict(title="MNIST by MLP Test Accuracy"))

fig = plt.figure(figsize=(8, 8), dpi=326)
with torch.no_grad():
    for i in range(9):
        x, y = mnist_test_data[i]
        img = PIL.Image.fromarray(x.numpy())
        ax = plt.subplot(int(f"33{i+1}"))
        plt.imshow(img)
        y_pred = np.argmax(model(torch.unsqueeze(x, 0).cuda()))
        plt.title(f"y_true:{y}, y_pred:{y_pred}")
        ax.set_xticks([])
        ax.set_yticks([])
vis.matplot(fig, opts=dict(title="MLP classification examples"), win="MLP-classification-examples")
plt.close("all")
