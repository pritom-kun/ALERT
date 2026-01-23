# ALERT: Efficient Extraction of ATT&CK Techniques from CTI Reports Using Active Learning

[![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-2.7.0-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pritom-kun/ALERT/blob/main/LICENSE)

Previous Version of the code can be found at [*ALERT*](https://github.com/space-urchin/ALERT) 

If the code or the paper has been useful in your research, please add a citation to our work:

```
@InProceedings{ALERT,
author="Rahman, Fariha Ishrat
and Halim, Sadaf Md
and Singhal, Anoop
and Khan, Latifur",
editor="Ferrara, Anna Lisa
and Krishnan, Ram",
title="ALERT: A Framework for Efficient Extraction of Attack Techniques from Cyber Threat Intelligence Reports Using Active Learning",
booktitle="Data and Applications Security and Privacy XXXVIII",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="203--220",
isbn="978-3-031-65172-4"
}
```

## Dependencies

The code is based on Python 3.14 and PyTorch 2.9.0 and requires a few further dependencies, listed in [requirements.txt](requirements.txt). It should work with newer versions as well.

This codebase is based on the [*Deep Deterministic Uncertainty (DDU)*](https://github.com/omegafragger/DDU) repository.

## Training

In order to train a model for the Active Learning task, use the [active_learning_alert.py](active_learning_alert.py) script. Following are the main parameters for training:

```
--seed: seed for initialization
--dataset: dataset used for training (tram/cti2mitre)
--model: model to train (scibert/roberta-base/modernbert)
--al-type: type of active learning acquisition model (entropy/energy/confidence/margin/gmm/dropout/coreset/ensemble/random)
--num-initial-samples: number of initial samples in the training set
--max-training-samples: maximum number of training samples
--acquisition-batch-size: batch size for each acquisition step
```

As an example, to run the active learning experiment on tram using the scibert model and entropy method, use:

```
python active_learning_script.py --seed 1 --model scibert --dataset tram --al-type entropy
```




