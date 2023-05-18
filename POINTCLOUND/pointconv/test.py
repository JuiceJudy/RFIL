import numpy as np
import torch

# x = torch.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6]])
# y = torch.tensor([[1,2,3],[4,5,6]]).view(1, 6)

# print(x.shape, x)
# print(y.shape, y)
# print(x-y)



# mlp1 = [64, 64, 128]
# pruning_rate = [0.2,0.3,0.4,0.5,0.6]

# mlp1 = [int(x*(1.-y)) for x,y in zip(mlp1,pruning_rate[0:3])]
# print(mlp1)

# 需要修改的各层下表 ;(
# sa1_conv   = [0,2,4]
# sa1_bn     = [6,11,16]
# sa1_linear = 42
# sa2_conv   = [x+70 for x in sa1_conv]
# sa2_bn     = [x+70 for x in sa1_bn]
# sa2_linear = sa1_linear+70
# sa3_conv   = [x+70 for x in sa2_conv]
# sa3_bn     = [x+70 for x in sa2_bn]
# sa3_linear = sa2_linear+70
# fc1        = 210

# print(sa2_conv)
# print(sa3_conv)
# print(sa2_bn)
# print(sa3_bn)


# x = [1,2,3,4]
# print(x[2:3])


x = torch.tensor([[1,2,3],
			     [0,0,0],
			     [7,8,9]])


y = torch.tensor([[1,1],
			     [2,2],
			     [3,3]])

z = torch.matmul(input=x, other=y)

print(z)

# x = x.view(-1)


x = 1165255552
y = 63177368
print((x-y)/x)