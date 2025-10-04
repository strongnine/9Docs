## LightStereo

LightStereo 的特点就是计算速度快，显存占用小，优化 3D 代价体的通道维度来提高性能，同时保证了精度和效率的平衡，证明了 2D 代价聚合体在立体匹配上的可用性。在论文中的计算量为 22 GFLOPs，推理时间为 17ms。

技术团队：中科慧拓、武汉大学、西安交通大学、中科院自动化所、加州大学伯克利分校、博洛尼亚大学、元橡科技。

用 MobileNetV2-100 网络作为特征提取的 Backbone 为例，以下的图把整个 LightStereo 的网络结构画出来。主要创新点：

- 多尺度卷积注意力模块（Multi-Scale Convolution Attention, MSCA）：捕获和融合多个尺度的特征。
- [倒残差块](https://strongnine.github.io/9Docs/dev/DL/CNN/#倒残差结构)：使用 MobileNetV2 中的倒残差块来聚合低分辨率代价体的误差，来降低计算复杂度。

![Conv2D](../assets/images/Stereo/LightStereo.png)