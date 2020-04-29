import torch
import torchvision
import torch.nn as nn

model_ft = torchvision.models.mobilenet_v1()
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, 2, bias=True)
model_ft.load_state_dict(torch.load('/home/well/0.94118mobilenet.pt'))
model_ft.eval()

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model_ft, example)
traced_script_module.save("./model.pt")
print("Finished Transformation")

