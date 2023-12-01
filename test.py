import numpy as np


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
print(softmax([95, 96, 97, 98, 99]))


# 让我们来测试一下上面的代码：

# print(softmax([1, 2, 3]))
# result: array([0.09003057, 0.24472847, 0.66524096])

# 但是，当我们尝试输入一个比较大的数值向量时，就会出错：

# print(softmax([1000, 2000, 3000]))
# result: array([nan, nan, nan])


# 写成函数
def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


print(softmax([95, 96, 97, 98, 99]))