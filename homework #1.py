import cv2
import torch

img1 = cv2.imread('./1.jpg',cv2.COLOR_BGR2RGB)
img2 = cv2.imread('./2.jpg',cv2.COLOR_BGR2RGB)
img3 = cv2.imread('./3.jpg',cv2.COLOR_BGR2RGB)

print(img1.shape, img2.shape, img3.shape)

class TensorManipulator:
    def __init__(self, img1, img2, img3):
        self.tensor_list = []
        for img in [img1, img2, img3]:
            tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            self.tensor_list.append(tensor)

    def concatenation(self):
        self.concat_tensor = torch.stack(self.tensor_list, dim=0)
        return self.concat_tensor
    def flatten(self,tensor):
        self.flatten_tensor = tensor.view(tensor.shape[0], -1)
        return self.flatten_tensor

    def average(self, tensor):
        return tensor.mean(dim=1)


obj = TensorManipulator(img1, img2, img3)
out = obj.concatenation()
out_flt = obj.flatten(out)
out_avg = obj.average(out_flt)
print(f"{out.shape}\n{out_flt.shape}\n{out_avg}\n")


