# MDPF-Net: Modality Decoupling and Polarized Fusion Multi-modal Detection Network

<div align="center">
  <p>
    <a href="https://github.com/zzyadad/MDPF-Net" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/zzyadad/MDPF-Net/master/ultralytics/over.jpg" alt="MDPF-Net Framework"></a>
  </p>
</div>

## 🎯 核心创新

- ✨ **Modality-Specific Decoupling Backbone (MSDB)**: 模态特异性解耦骨干网络，包含针对RGB的 **S³M** 模块和针对红外的 **SDSR** 模块。
- 🔥 **Polarized Linear Cross-Attention (PLCA)**: 极化线性交叉注意力机制，利用正负极化通路处理模态间的一致性与互补性。
- 🚀 **SOTA Performance**: 在 **M3FD (91.6% mAP)** 和 **LLVIP (96.3% mAP)** 数据集上均超越了现有的最先进方法。

## 📖 项目简介

**Multi-modal object detection aims to leverage the complementary features of RGB and Infrared (IR) images to enhance detection performance.**

However, existing feature extraction networks often overlook the inherent physical heterogeneity between modalities and struggle to effectively distinguish between feature commonalities and differences during the fusion phase. To address these issues, we propose a **Modality Decoupling and Polarized Fusion Multi-modal Detection Network (MDPF-Net)**.

Specifically:
1. We construct a **Modality-Specific Decoupling Backbone (MSDB)**.
2. For the RGB modality, we utilize a **Spatial-Spectral Synergistic Modulation Unit (S³M)**.
3. For the IR modality, we employ a specialized **Spectrum-Decoupled Structure Reshaping (SDSR)** Module.
4. We propose **Polarized Linear Cross-Attention (PLCA)** for the feature fusion stage.

## 📅 代码发布计划

> **重要说明**: 本项目的完整代码（基于 Ultralytics 框架）将在论文被期刊/会议正式录用后公开发布。

**预计发布内容**:
- [ ] 完整的 MDPF-Net 训练代码
- [ ] 模型推理与验证脚本
- [ ] M3FD 与 LLVIP 预训练模型权重
- [ ] 详细的配置文件 (yaml)
- [ ] 实验复现指南

## 📧 联系方式

如有学术交流或合作意向，请联系：
- 📧 Email: [10431240210@stu.qlu.edu.cn](mailto:10431240210@stu.qlu.edu.cn)

## 📄 许可证

本项目遵循 [MIT License](LICENSE) 或 [AGPL-3.0](LICENSE) (取决于 Ultralytics 协议)

---

⭐ **如果您对 MDPF-Net 感兴趣，请点击 Star 关注项目进展！**

**💡 提示**: 请持续关注本仓库 [zzyadad/MDPF-Net](https://github.com/zzyadad/MDPF-Net)，我们会在论文录用后第一时间发布完整代码。
