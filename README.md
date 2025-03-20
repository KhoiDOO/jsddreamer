# Jsddreamer

##### Table of contents
1. [Getting Started](#Getting-Started)
2. [Experiments](#Experiments)
3. [Acknowledgments](#Acknowledgments)
4. [Contacts](#Contacts)

# Text-to-3D Generation using Jensen-Shannon Score Distillation

<!-- <a href="https://di-mi-ta.github.io/HyperInverter/"><img src="https://img.shields.io/badge/WEBSITE-Visit%20project%20page-blue?style=for-the-badge"></a> -->
<a href="https://arxiv.org/abs/2503.10660"><img src="https://img.shields.io/badge/arxiv-2112.00719-red?style=for-the-badge"></a>
<!-- <a href="http://colab.research.google.com/github/di-mi-ta/hyper-inverter-code/blob/main/notebooks/demo.ipynb"><img src="https://img.shields.io/badge/DEMO-open%20in%20colab-blue?style=for-the-badge"></a> -->

[Khoi Do](https://khoidoo.github.io/),
[Binh-Son Hua](https://sonhua.github.io/)<br>
Trinity College Dublin, Ireland

> **Abstract:** 
Score distillation sampling is an effective technique to generate 3D models from text prompts, utilizing pre-trained large-scale text-to-image diffusion models as guidance. However, the produced 3D assets tend to be over-saturating, over-smoothing, with limited diversity. These issues are results from a reverse Kullback-Leibler (KL) divergence objective, which makes the optimization unstable and results in mode-seeking behavior. In this paper, we derive a bounded score distillation objective based on Jensen-Shannon divergence (JSD), which stabilizes the optimization process and produces high-quality 3D generation. JSD can match well generated and target distribution, therefore mitigating mode seeking. We provide a practical implementation of JSD by utilizing the theory of generative adversarial networks to define an approximate objective function for the generator, assuming the discriminator is well trained. By assuming the discriminator following a log-odds classifier, we propose a minority sampling algorithm to estimate the gradients of our proposed objective, providing a practical implementation for JSD. We conduct both theoretical and empirical studies to validate our method. Experimental results on T3Bench demonstrate that our method can produce high-quality and diversified 3D assets.


Details of the model architecture and experimental results can be found in [our following paper](https://arxiv.org/abs/2503.10660).
```bibtex
@misc{do2025textto3dgenerationusingjensenshannon,
      title={Text-to-3D Generation using Jensen-Shannon Score Distillation}, 
      author={Khoi Do and Binh-Son Hua},
      year={2025},
      eprint={2503.10660},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.10660}, 
}
```
**Please CITE** our paper whenever our model implementation is used to help produce published results or incorporated into other software.


## Getting Started

The codebase is tested on
- Ubuntu
- CUDA 12.4, CuDNN 9.x 

### Installation

- Clone this repo:
```git 
git clone https://github.com/KhoiDOO/jsddreamer.git
cd jsddreamer
```

- Virtual Environment Creation
```bash
python3 -m venv .env
source .env/bin/activate
```
- Install dependencies:
```bash
pip install -U pip wheel
pip3 install torch torchvision torchaudio # isntall Pytorch
pip install -r requirements.txt
pip install accelerate mediapipe tomesd # install the support packages
```

## Experiments
- For generating 3D objects given a text prompt
```shell
# Short prompt
python launch.py --config configs/jsdlr.yaml --train --gpu 0 seed=0 \
      system.prompt_processor.prompt="A pair of polka-dotted sneakers"

# Long prompt
python launch.py --config configs/jsdlr.yaml --train --gpu 0 seed=0 \
      system.prompt_processor.prompt="A small, hollow, asymmetrical birdhouse, painted in cheerful colors, \
      with a round entrance and a tiny perch, swaying gently in a backyard apple tree"
```

- For generating 3D objects with complex surroundings
```shell
python launch.py --config configs/jsdlr-surr.yaml --train --gpu 0 seed=0 \
      system.prompt_processor.prompt="A medium-sized, layered, radially symmetrical conch shell, \
      with a rough texture on the outside, fading from pink to cream, sitting alone on a sandy beach"

python launch.py --config configs/jsdlr-surr.yaml --train --gpu 0 seed=0 \
      system.prompt_processor.prompt="A compact, cylindrical, vintage pepper mill, with a polished, \
      ornate brass body, slightly worn from use, placed beside a porcelain plate on a checkered tablecloth"
```

- For generative diverse 3D object

```shell
# Short prompt
PROMPT="A large multi-layered symmetrical wedding cake"

python launch.py --config configs/asd.yaml --train --gpu 0 seed=0 system.prompt_processor.prompt="$PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=1 system.prompt_processor.prompt="$PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=2 system.prompt_processor.prompt="$PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=3 system.prompt_processor.prompt="$PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=4 system.prompt_processor.prompt="$PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=5 system.prompt_processor.prompt="$PROMPT"

LONG_PROMPT="A large, multi-layered, symmetrical wedding cake, with smooth fondant, delicate piping, '
      and lifelike sugar flowers in full bloom, displayed on a silver stand"

# Long prompt
python launch.py --config configs/asd.yaml --train --gpu 0 seed=0 system.prompt_processor.prompt="$LONG_PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=1 system.prompt_processor.prompt="$LONG_PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=2 system.prompt_processor.prompt="$LONG_PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=3 system.prompt_processor.prompt="$LONG_PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=4 system.prompt_processor.prompt="$LONG_PROMPT"
python launch.py --config configs/asd.yaml --train --gpu 0 seed=5 system.prompt_processor.prompt="$LONG_PROMPT"
```


## Acknowledgments
This repository is built based on [ThreeStudio](https://github.com/threestudio-project/threestudio/tree/main) framework. Otherwise, we leverage [T3Bench](https://t3bench.com/) as the main benchmark for quality and alignment evaluation. The diversity evaluation is constructed based on [DiverseDream](https://github.com/VinAIResearch/DiverseDream).

This project is supported by Research Ireland under the Research Ireland Frontiers for the Future Programme - Project, award number 22/FFP-P/11522.

Overall, thank you so much to the authors for their great work and efforts to release source code and pre-trained weights.

## Contacts
If you have any questions, please drop an email to _khoido8899@gmail.com_ or open an issue in this repository.
