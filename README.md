[![arXiv](https://img.shields.io/badge/arXiv-2301.07969-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2301.07969)
![visitor badge](https://visitor-badge.glitch.me/badge?page_id=diegovalsesia/MMD-DDM)
# <p align='center'> MMD-DDM </p>
## <p align='center'>Fast Inference in Denoising Diffusion Models via MMD Finetuning</p>
</div>
<div align="center">
  <img src="figs/mmd-1.png"/>
  <img src="figs/MMD-DDM2.png"/>
</div><br/>

## Introduction 
Denoising Diffusion Models (DDMs) have become a popular tool for generating high-quality samples from complex data distributions. These models are able to capture sophisticated patterns and structures in the data, and can generate samples that are highly diverse and representative of the underlying distribution. However, one of the main limitations of diffusion models is the complexity of sample generation, since a large number of inference timesteps is required to faithfully capture the data distribution. In this paper, we present MMD-DDM, a novel method for fast sampling of diffusion models. Our approach is based on the idea of using the Maximum Mean Discrepancy (MMD) to finetune the learned distribution with a given budget of timesteps. This allows the finetuned model to significantly improve the speed-quality trade-off, by substantially increasing fidelity in inference regimes with few steps or, equivalently, by reducing the required number of steps to reach a target fidelity, thus paving the way for a more practical adoption of diffusion models in a wide range of applications.

## Finetuning a pretrained model
To finetune a pretrained diffusion modelusing our proposed MMD-DDM strategy, download the pretrained models and adjust the path in runners/diffusion.py, or use models present in /function/ckpt_util.py and run the command: 
```
python main.py --config {DATASET}.yml --timesteps {num_timesteps (e.g 5)} --exp {PROJECT_PATH} --train 

```

## Image Sampling for FID evaluation
To sample image generated from the finetuned model adjust the path in test_FID funciton in runners/diffusion.py with your newly trained model and run:
```
python main.py --config {DATASET}.yml --timesteps {num_timesteps (e.g 5)} --test_FID  

```


## Citation
If you find MMD-DDM helpful in your research, please consider citing: 
```bibtex
@article{aiello2023fast,
  title={Fast Inference in Denoising Diffusion Models via MMD Finetuning},
  author={Aiello, Emanuele and Valsesia, Diego and Magli, Enrico},
  journal={arXiv preprint arXiv:2301.07969},
  year={2023}
}
```

## Acknowledgements 
This repository is based on DDIM official implementation: https://github.com/ermongroup/ddim

## Contact 
If you have any questions, feel free to open an issue or contact us at emanuele.aiello@polito.it

<br><br>
<p align="center">:construction: :pick: :hammer_and_wrench: :construction_worker:</p>
<p align="center">pretrained models will be released soon!</p>
