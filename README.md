# PAML

This repository is the official Pytorch implementation for paper **PMAL: Prototype-Aware Multimodal Alignment for Open-Vocabulary Visual Grounding**.

Visual grounding (VG) aims to utilize given natural language queries to locate specific target objects within images. While con-
temporary transformer-based approaches demonstrate strong localization performance in standard scene (i.e, scenarios without any
novel objects), they exhibit notable limitations in open-vocabulary scene (i.e, both familiar and novel object categories during test-
ing). These limitations primarily stem from three key factors: (1) imperfect alignment between visual and linguistic modalities, (2)
insufficient cross-modal feature integration, and (3) ineffective utilization of semantic prototype information. To overcome these
challenges, we present Prototype-Aware Multimodal Learning (PAML), an innovative framework that systematically addresses
these issues through several key components: First, we leverage ALBEF to establish robust cross-modal alignment during initial
feature encoding. Subsequently, our Visual Discriminative Feature Encoder selectively enhances target object representations while
suppressing irrelevant visual context. The framework then incorporates a novel prototype discovering and inheriting mechanism
that extracts and aggregates multi-neighbor semantic prototypes to facilitate open-vocabulary recognition. These enriched features
undergo comprehensive multimodal interaction and fusion through our Multi-stage Decoder before final bounding box regression.
Extensive experiments across five benchmark datasets validate our approach, showing competitive performance in standard scene
while achieving state-of-the-art results in open-vocabulary scene.



## Contents

1. [Usage](#usage)
2. [Results](#results)
3. [Contacts](#contacts)
4. [Acknowledgments](#acknowledgments)

## Usage

### Dependencies
- Python 3.11.0
- PyTorch 2.5.1 + cu121  

### Data Preparation

You can download the images follow [TransVG](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md) and place them in ./ln_data folder:

The training samples can be download from [data](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view). Finally, the `./data/` folder should have the following structure:

```
|-- data
      |-- flickr
      |-- gref
      |-- gref_umd
      |-- referit
      |-- unc
      |-- unc+
```

### Pretrained Checkpoints
1.You can download the DETR checkpoints from [detr_checkpoints](https://drive.google.com/drive/folders/1SOHPCCR6yElQmVp96LGJhfTP46RxVwzF). These checkpoints should be downloaded and move to the checkpoints directory.

```
mkdir pretrained_checkpoints
mv detr_checkpoints.tar.gz ./pretrained_checkpoints/
tar -zxvf detr_checkpoints.tar.gz
```

### Training and Evaluation

1.  Training on RefCOCO. 
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --config configs/PAML_R50_unc.py --test_split val
    ```

2.  Evaluation on RefCOCO.
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --use_env test.py --config configs/PAML_R50_unc.py --checkpoint PAML_R50_unc.pth --batch_size_test 24 --test_split testA;
    ```

## Results
**Comparison of open-vocabulary scene with the state-of-the-art methods by the models trained on RefCOCO dataset and test on ReferIt and Flickr30K Entities.**
| Models | ReferIt val | ReferIt test | Flickr30K val | Flickr30K test |
|--------|-------------|--------------|---------------|----------------|
| LBYL-Net | 21.76 | 22.93 | 22.97 | 21.62 |
| RefTR | 27.67 | 29.45 | 24.03 | 23.08 |
| VGTR | 23.11 | 23.46 | 9.57 | 9.17 |
| SeqTR | 28.78 | 29.53 | 15.45 | 13.97 |
| VLTVG | 38.76 | 40.54 | 25.28 | 24.31 |
| TransVG | 36.11 | 37.86 | 23.78 | 22.83 |
| TransCP | 38.35 | 40.62 | 29.01 | 27.71 |
| ResVG | 37.62 | 39.54 | 25.77 | 26.58 |
| PAML(Ours) | **41.70** | **42.78** | **29.07** | **27.75** |

