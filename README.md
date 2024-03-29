# Cut-FUNQUE
This repository contains the official implementation of the Cut-FUNQUE model proposed in the following paper.

1. A. K. Venkataramanan, C. Stejerean, I. Katsavounidis, H. Tmar and A. C. Bovik, "Cut-FUNQUE: An Objective Quality Model for Compressed Tone-Mapped High Dynamic Range Videos," arXiv preprint 2024.

Cut-FUNQUE achieves state-of-the-art quality prediction accuracy on tone-mapped and compressed high dynamic range videos, at a fraction of the computational complexity of SOTA models like MSML.

## Features of Cut-FUNQUE
1. A novel perceptually uniform encoding of color signals (__PUColor__) that we use to represent both HDR and SDR color stimuli in a common domain. In this manner, PUColor enables the meaningful comparison of stimuli across dynamic ranges, which is essential when comparing HDR and SDR videos.
2. A __binned-weighting__ approach to separately handle image regions having different visual characteristics such as brightness, contrast, and temporal complexity.
3. Novel __statistical similarity__ measures of visual quality to overcome the limitations of pixel-wise comparisons across dynamic ranges.

## Accuracy and Efficiency of Cut-FUNQUE
The Cut-FUNQUE model achieves SOTA accuracy at a fraction of the existing SOTA MSML!


| Model | Accuracy | GFLOPs/Frame
| ---------- | ------- | ---------
|FSITM | 0.4626 | 8.9487
|BRISQUE | 0.4833 | 0.2120
|TMQI | 0.4956 | 0.9061
|FFTMI | 0.5315 | 27.5161
|3C-FUNQUE+ | 0.5661 | 0.3667
|RcNet | 0.5824 | 134.5597
|HIGRADE | 0.6698 | 2.6533
|MSML | 0.7740 | 67.2578
|__Cut-FUNQUE__ | __0.7781__ | __2.9257__

## Usage
### Setting up the environment
Create and activate a virtual environment using
```
python3 -m virtualenv .venv
source .venv/bin/activate
```
Install all required dependencies
```
python3 -m pip install -r requirements.txt
```
### Extract features from one video pair
To compute Cut-FUNQUE features from one video pair, use the command

```
python3 extract_features.py --ref_video <path to reference video> --dis_video <path to distorted video>
```

For more options, run
```
python3 extract_features.py --help
```

### Extract features for all videos in a dataset
First, define a subjective dataset file using the same format as that in [datasets/](https://github.com/abhinaukumar/cut_funque/tree/main/datasets). The dataset file provided here is that of the [LIVE-TMHDR](https://live.ece.utexas.edu/research/LIVE_TMHDR/index.html) dataset, which was used to benchmark Cut-FUNQUE.


Then, run
```
python3 extract_features_from_dataset.py --dataset <path to dataset file> --processes <number of parallel processes to use>
```
*Note: This command computes features and saves the results to disk. It does __not__ print any features. Saved features may be used for downstream tasks - example below*

### Run cross-validation
To evaluate features using content-separated random cross-validation, run
```
python3 crossval_features_on_dataset.py --dataset <path to dataset file> --splits <number of random train-test splits> --processes <number of parallel processes to use>
```


*Note: This command may be run without running `extract_features_from_dataset.py` first. In that case, features will be extracted and saved first, before performing cross-validation*

This script is an example of down-stream tasks that can be performed easily after feature extraction.
