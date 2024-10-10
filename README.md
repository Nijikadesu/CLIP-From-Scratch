## 从零实现 CLIP | CLIP from Scratch

在这个仓库中，我将实现一个简单的 CLIP 模型，其中图像编码器由一个以 ResNet 为主干的特征提取网络构成，文本编码器由一个 Transformer Decoder 构成。

在实现模型后，我会在 MNIST 手写数字数据集上训练一个简单的 CLIP 分类器。

关于 CLIP 模型，也可以了解我的[博客](https://nijikadesu.github.io/2024/10/06/dive-into-clip/)（更新中）

项目进度：
- [x] 模型架构部分代码
- [ ] 训练与推理
- [ ] 博客完善