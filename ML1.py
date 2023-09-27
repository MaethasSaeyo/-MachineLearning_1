from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat")

mnist ={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
x = mnist["data"]
y = mnist["target"]

number = x[2000]
num_img = number.reshape(28,28)

print(int(y [2000]))
plt.imshow(
    num_img,
    cmap=plt.cm.binary,
    interpolation="nearest"
    )
plt.show()
