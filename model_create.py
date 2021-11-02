import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Model, load_checkpoint, load_param_into_net, Tensor, export
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import Callback
from mindspore import load_checkpoint, load_param_into_net


# 回调类
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model  # 计算图模型
        self.eval_dataset = eval_dataset  # 验证数据集
        self.steps_loss = steps_loss  # 收集step和loss值之间的关系，数据格式{"step": [], "loss_value": []}。
        self.steps_eval = steps_eval  # 收集step对应模型精度值accuracy的信息，数据格式为{"step": [], "acc": []}。

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch - 1) * 1875 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 125 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])


# 神经网络
class LeNet5(nn.Cell):
    # 定义神经网络各层参数
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')  # 一维卷积层输入通道数为num_channel，输出通道数为6，卷积窗口宽度为5，采用丢弃方式。
        # 将返回输出的可能最大宽度。多余的像素将被丢弃
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')  # 二维卷积层，输入通道为5，输出通道为16，卷积窗口宽度为5
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))  # 连接层，输入通道，输出通道，
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()  # 线性激活函数
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)  # 二维池化层
        self.flatten = nn.Flatten()  # 展平层

    # 构建神经网络
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 显示原始数据
def view(data_path):
    mnist_ds = ds.MnistDataset(data_path)
    print('The type of mnist_ds:', type(mnist_ds))
    print("Number of pictures contained in the mnist_ds：", mnist_ds.get_dataset_size())

    dic_ds = mnist_ds.create_dict_iterator()
    item = next(dic_ds)
    img = item["image"].asnumpy()
    label = item["label"].asnumpy()

    print("The item of mnist_ds:", item.keys())  # item为字典
    print("Tensor of image in item:", img.shape)  # 输出高度宽度
    print("The label of item:", label)  # 图片的数字

    plt.imshow(np.squeeze(img))
    plt.title("number:%s", item["label"].asnumpy())
    plt.show()  # 绘图


# 创建数据集
def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)

    # 数据增强使用的参数
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 根据参数生成数据增强方法
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # 像素缩放
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)  # 对数据归一标准化
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()  # 图片张量变换
    type_cast_op = C.TypeCast(mstype.int32)  # 类型转换

    # 调用数据增强方法
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 处理生成数据
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 随机将数据存放在可容纳10000张图片地址的内存中进行混洗
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)  # 从混洗的10000张图片地址中抽取32张图片组成一个batch
    mnist_ds = mnist_ds.repeat(repeat_size)  # 对batch数据进行增强

    return mnist_ds


def create_model():

    mnist_path = "./datasets/MNIST_Data"
    model_path = "./models/ckpt/mindspore_quick_start/"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    lr = 0.01
    epoch_size = 1
    momentum = 0.9

    # 创建神经网络
    network = LeNet5()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)  # 优化器，trainable_params返回所有可训练参数列表

    # 定义损失函数
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 创建数据集
    repeat_size = 1
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    eval_dataset = create_dataset(os.path.join(mnist_path, "test"), 32)

    # 定义模型
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # 保存数据的设置
    config_ck = CheckpointConfig(save_checkpoint_steps=375, keep_checkpoint_max=16)
    # 保存模型相关设置
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=model_path, config=config_ck)

    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    # collect the steps,loss and accuracy information
    step_loss_acc_info = StepLossAccInfo(model, eval_dataset, steps_loss, steps_eval)

    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125), step_loss_acc_info],
                dataset_sink_mode=False)
    input_data = np.ones([1, 1, 32, 32]).astype(np.float32)
    # 导出模型
    export(network, Tensor(input_data), file_name='./models/result/result', file_format='MINDIR')
    return model


# 显示损失值
def loss_show(steps_loss):
    steps = steps_loss["step"]
    loss_value = steps_loss["loss_value"]
    steps = list(map(int, steps))
    loss_value = list(map(float, loss_value))
    plt.plot(steps, loss_value, color="red")
    plt.xlabel("Steps")
    plt.ylabel("Loss_value")
    plt.title("Change chart of model loss value")
    plt.show()


# 显示精度值
def eval_show(steps_eval):
    plt.xlabel("step number")
    plt.ylabel("Model accuracy")
    plt.title("Model accuracy variation chart")
    plt.plot(steps_eval["step"], steps_eval["acc"], "red")
    plt.show()


# 验证模型
def test_net(network, mnist_path, ds_eval):
    print("============== Starting Testing ==============")
    # 载入保存模型
    param_dict = load_checkpoint("./models/ckpt/mindspore_quick_start/checkpoint_lenet-1_1875.ckpt")
    # 载入参数
    load_param_into_net(network, param_dict)
    # 载入数据集
    ds_eval = create_dataset(os.path.join(mnist_path, "test"))
    acc = eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


# 利用模型预测
def predict(model):

    ds_test = create_dataset("./datasets/MNIST_Data/train").create_dict_iterator()
    data = next(ds_test)
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)
    err_num = []
    index = 1
    for i in range(len(labels)):
        plt.subplot(4, 8, i + 1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title("pre:{}".format(pred[i]), color=color)
        plt.imshow(np.squeeze(images[i]))
        plt.axis("off")
        if color == 'red':
            index = 0
            print("Row {}, column {} is incorrectly identified as {}, the correct value should be {}".format(
                int(i / 8) + 1, i % 8 + 1, pred[i], labels[i]), '\n')
    if index:
        print("All the figures in this group are predicted correctly!")
    print(pred, "<--Predicted figures")
    print(labels, "<--The right number")
    plt.show()


if __name__ == '__main__':
    model = create_model()
    predict(model)





