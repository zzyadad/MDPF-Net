<div align="center">
  <p>
    <a href="https://github.com/zzyadad/MDPF-Net" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/zzyadad/MDPF-Net/master/ultralytics/over.jpg" alt="MDPF-Net Framework Architecture"></a>
  </p>

  <h1>MDPF-Net: Modality Decoupling and Polarized Fusion Multi-modal Detection Network</h1>

<div>
    <a href="https://github.com/zzyadad/MDPF-Net/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="CI Status"></a>
    <a href="https://console.paperspace.com/github/zzyadad/MDPF-Net"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/zzyadad/MDPF-Net"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>

</div>

## Abstract

> **Multi-modal object detection aims to leverage the complementary features of RGB and Infrared (IR) images to enhance detection performance. However, existing feature extraction networks often overlook the inherent physical heterogeneity between modalities and struggle to effectively distinguish between feature commonalities and differences during the fusion phase.**
>
> **To address these issues, we propose a Modality Decoupling and Polarized Fusion Multi-modal Detection Network (MDPF-Net). Specifically, we construct a Modality-Specific Decoupling Backbone (MSDB). For the RGB modality, we utilize a Spatial-Spectral Synergistic Modulation Unit (S³M). For the IR modality, we employ a specialized Spectrum-Decoupled Structure Reshaping (SDSR) Module. Furthermore, we propose Polarized Linear Cross-Attention (PLCA). This mechanism utilizes positive and negative polarized pathways to process the consistency and complementarity between modalities during the feature fusion stage.**
>
> **Our method achieves 91.6% and 96.3% mAP on the M3FD and LLVIP datasets respectively, surpassing existing state-of-the-art (SOTA) methods.**

---

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) based implementation of **MDPF-Net**. This repository contains the source code for our proposed method. We hope that the resources here will help you reproduce our results.

<div align="center">
  </div>
