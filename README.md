# reimplement-residual-attention-network

In recent years, nature of extracted attention has been studied in previous work and formulating attention drift is found to enhance image classification. Another advanced technology is the proposal of 'Deep Residual Network', which solved problems in deep neural networks. Based on these two techniques, the authors of this paper proposed a combination of attention module and residual network and this Attention Residual Network achieved great performance in image classification tasks.

In our implementation, based on the original architecture, we modified some units in it and generated our own attention residual network. We evaluated the performance of our design on both CIFAR-10 and CIFAR-100 dataset and obtained nearly the same results in the original paper. We also tested the performance of shortcut connection by replacing it with convolution layers. From the results, it truns out applying shortcut connection increases performance of deep network.

Original Project Github Page: https://github.com/ecbme4040/e4040-2021Fall-Project-REAL-xh2510-qt2139-kl3350

(This is a group work by Xin Huang(me), Qimeng Tao and Kangrui Li)
