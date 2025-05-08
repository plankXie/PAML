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
3. [Acknowledgments](#acknowledgments)

## Usage

### Dependencies
- Python 3.11.0
- PyTorch 2.5.1 + cu121  
- Check [requirements.txt](requirements.txt) for other dependencies.
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

**Comparison of open-vocabulary scene with the state-of-the-art methods by the models trained on ReferIt and test on RefCOCO/+/g and Flickr30K Entities.**
| Models       | Flickr30k val | Flickr30k test | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val-g |
|--------------|---------------|-----------------|-------------|---------------|---------------|--------------|----------------|----------------|----------------|
| LBYL-Net     | 26.00         | 26.19           | 55.61       | 61.75         | 46.26         | 38.03        | 43.14          | 29.29          | 40.02          |
| RefTR        | 29.88         | 30.47           | 58.16       | 61.15         | 52.83         | 36.49        | 40.15          | 32.67          | 44.96          |
| VGTR         | 14.31         | 15.01           | 10.03       | 7.25          | 12.91         | 11.1         | 7.75           | 13.60          | 8.44           |
| SeqTR        | 17.64         | 18.38           | 9.96        | 9.61          | 10.13         | 11.05        | 9.78           | 11.82          | 5.76           |
| VLTVG        | 51.40         | 53.13           | 64.54       | 65.83         | **61.98**     | 41.04        | 43.70          | **38.33**      | 47.11          |
| TransVG      | 52.29         | 54.38           | 61.67       | 63.23         | 59.39         | 37.52        | 39.13          | 35.10          | 45.79          |
| TransCP      | 52.92         | 55.07           | 64.26       | 65.55         | 61.71         | 40.17        | 42.70          | 36.90          | 47.40          |
| ResVG        | 52.67         | 54.71           | 64.02       | 65.58         | 59.99         | 39.67        | 41.32          | 37.03          | 46.92          |
| PAML(Ours)   | **53.33**     | **55.63**       | **65.04**   | **67.16**     | 60.1          | **44.03**    | **47.43**      | 37.96          | **48.89**      |

**Comparison of open-vocabulary scene with the state-of-the-art methods by the models trained on Flickr30K dataset and test on RefCOCO/+/g and ReferIt.**
| Models       | ReferIt val | ReferIt test | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val-g |
|--------------|-------------|--------------|-------------|---------------|---------------|--------------|----------------|----------------|----------------|
| RefTR        | 36.61       | 35.70        | 29.72       | 36.16         | 23.21         | 27.67        | 31.85          | 22.21          | 35.13          |
| VGTR         | 8.59        | 8.49         | 18.33       | 17.35         | 11.85         | 9.09         | 5.78           | 11.95          | 13.05          |
| SeqTR        | 11.80       | 11.83        | 12.41       | 12.52         | 11.38         | 12.70        | 12.96          | 11.63          | 18.44          |
| VLTVG        | 44.13       | 42.25        | 39.67       | **46.33**     | **31.82**     | 36.28        | 41.83          | 31.34          | 38.63          |
| TransVG      | 43.07       | 41.08        | 36.82       | 44.67         | 30.70         | 35.17        | 41.97          | 31.01          | 44.85          |
| TransCP      | 45.36       | **43.99**    | **39.35**   | 46.26         | 31.76         | **37.80**    | **42.91**      | **32.95**      | **44.91**      |
| ResVG        | 44.96       | 42.58        | 37.01       | 45.86         | 30.32         | 36.71        | 41.83          | 30.91          | 44.62          |
| PAML(Ours)   | **45.64**   | 43.48        | 38.73       | 46.30         | 29.87         | 36.82        | 42.54          | 30.70          | 42.09          |

## Acknowledgment
The code is based on [ResVG](https://github.com/minghangz/ResVG), [TransVG](https://gthub.com/djiajunustc/TransVG), [TransCP](https://github.com/WayneTomas/TransCP), [ALBEF](https://github.com/salesforce/ALBEF). We thank the authors for their open-sourced code and encourage users to cite their works when applicable.  
