# 深度学习教程目录

## 第一部分：基础与概念

1. 神经网络基础  
   - 人工神经元模型（感知机）  
   - 多层感知机（MLP）与前向传播  
   - 激活函数（Sigmoid, Tanh, ReLU, Leaky ReLU, GELU 等）  
   - 损失函数（MSE, 交叉熵, Huber 等）  

3. 训练深度网络  
   - 反向传播算法  
   - 梯度下降（SGD, Momentum, Nesterov）  
   - 学习率调度  
   - 过拟合与正则化（L1/L2, Dropout, 数据增强, BatchNorm）  

---

## 第二部分：核心模型与结构
4. 卷积神经网络 (CNN)  
   - 卷积、池化、padding、stride  
   - 经典网络：LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet  
   - 应用：图像分类、目标检测、语义分割  

5. 循环神经网络 (RNN) 与序列建模  
   - RNN原理与梯度消失/爆炸问题  
   - LSTM 与 GRU  
   - 应用：时间序列预测、文本生成、语音建模  

6. 注意力机制与Transformer  
   - 注意力机制的动机与公式  
   - Self-Attention & Multi-Head Attention  
   - Transformer架构  
   - BERT, GPT 系列  
   - 应用：NLP、跨模态学习  

---

## 第三部分：训练技巧与工程化
7. 优化与正则化技巧  
   - Adam, RMSprop 等自适应优化器  
   - Batch Normalization, Layer Normalization  
   - 学习率调度、warm-up  
   - 数据增强与迁移学习  

8. 模型训练与调参  
   - 超参数选择（batch size, learning rate, optimizer）  
   - Early stopping, checkpoint  
   - 大规模训练与分布式训练  

9. 模型压缩与部署  
   - 模型剪枝、蒸馏、量化  
   - 部署到移动端/边缘设备（TensorRT, ONNX, TFLite）  
   - 推理加速  

---

## 第四部分：进阶与前沿
10. 生成模型  
    - 自编码器 (AE, VAE)  
    - 生成对抗网络 (GAN, DCGAN, WGAN, StyleGAN)  
    - Diffusion Model 简介  

11. 图神经网络 (GNN)  
    - 图卷积 (GCN)  
    - 应用：社交网络、分子预测、推荐系统  

12. 多模态与大模型  
    - CLIP, Flamingo, GPT-4, LLaVA  
    - 大模型的训练与微调（LoRA, PEFT, RAG）  

13. 深度强化学习  
    - DQN, Policy Gradient, Actor-Critic  
    - AlphaGo 案例   

14. 未来发展方向  
    - 联邦学习  
    - 自监督学习  
    - 通用人工智能（AGI）  
