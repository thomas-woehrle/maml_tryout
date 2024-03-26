import torch
from models import OmniglotModel
from tasks import OmniglotTask

n = 5
k = 1

model = OmniglotModel(n)
ckpt_path = './checkpoints/2024-03-26-14-06-58/ep99_loss253.07886791229248.pt'

ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model_state_dict'])

task = OmniglotTask('test', n)
x, y = task.sample(k)

y = model.forward(x)

print(y)
