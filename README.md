# 🚀AdaQual-Diff: Diffusion-Based Image Restoration via Adaptive Quality Prompting
* [Xin Su](https://scholar.google.com/citations?user=0AvoocgAAAAJ&hl=zh-CN)<sup>1</sup>  
* Chen Wu<sup>2</sup>  
* Yu Zhang<sup>3</sup>  
* Chen Lyu<sup>4</sup>  
* [Zhuoran Zheng](https://scholar.google.com/citations?user=pXzPL-sAAAAJ&hl=zh-CN)<sup>5 ✉️</sup>  

<sup>1</sup> Fuzhou University  
<sup>2</sup> University of Science and Technology of China  
<sup>3</sup> University of the Chinese Academy of Sciences  
<sup>4</sup> Shandong Normal University  
<sup>5</sup> Sun Yat-sen University  

---

## 🌿 Introduction

[AdaQual-Diff] is a novel diffusion framework for image restoration that dynamically integrates perceptual quality assessment into the generative process. Leveraging **Adaptive Quality Prompting**, AdaQual-Diff allocates computational attention proportionally to local degradation severity, achieving state-of-the-art performance on both composite and adverse weather degradations.

Our framework establishes a direct mathematical relationship between spatial quality scores and guidance complexity, enabling fine-grained, region-specific restoration without increasing model parameters or inference steps.

AdaQual-Diff not only advances the theoretical understanding of quality-aware diffusion guidance but also demonstrates practical efficiency with only 2 sampling steps and competitive inference speed.

---
## 📑 Open-source Plan 
- ✅ **2025.3.31**: This repo is created.
- ✅ **2025.4.17**: Release our [manuscript](https://arxiv.org/abs/2504.12605).
- ⬜ Release our visual results.
- ⬜ Release our pretrained models.
---
## 🙏 Acknowledgement

We gratefully acknowledge inspiration and foundational work from prior image restoration and prompt-based diffusion methods, including:

- [DeQAScore](https://github.com/zhiyuanyou/DeQA-Score)
- [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion)

 
## 📄 Citation 
```
If our work assists your research, feel free to give us a star ⭐ or cite us using


```
