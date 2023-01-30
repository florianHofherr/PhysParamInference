<h2 align="center">Neural Implicit Representations for Physical Parameter Inference from a Single Video</h2>

<p align="center">
    <a href="https://vision.in.tum.de/members/hofherrf">Florian Hofherr</a><sup>1</sup> &emsp;
    <a href="https://lukaskoestler.com">Lukas Koestler</a><sup>1</sup> &emsp;
    <a href="http://florianbernard.net">Florian Bernard</a><sup>2</sup> &emsp;
    <a href="https://vision.in.tum.de/members/cremers">Daniel Cremers</a><sup>1</sup>
</p>

<p align="center">
    <sup>1</sup>Technical University of Munich&emsp;&emsp;&emsp;
    <sup>2</sup>University of Bonn<br>
</p>

<p align="center">
    IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023
</p>
<p align="center">
    <a href="http://arxiv.org/abs/2204.14030">arXiv</a> |
    <a href="https://florianhofherr.github.io/phys-param-inference/">Project Page</a>
</p>

## Getting Started
You can create an anaconda environment called `physParamInference` with all the required dependencies by using
```
conda env create -f environment.yml
conda activate physParamInference
```

You can download the data using
```
bash download_data.sh
```
The script downloads all data used in the paper and stores them into a `/data/` folder.

## Usage
### Training
The training for the different scenarios is run by `python training_***.py`. The parameters for each scenario are defined in the respective config file in the `/configs/` folder. 

The results, including checkpoints, as well as the logs are stored in a sub folder of the `/experiments/` folder. The path is defined in the config file. You can monitor the progress of the training using tensorboard by calling `tensorboard --logidr experiments/path/to/experiment`.

### Evaluation
For each of the scenarios there is an `evaluate_***.ipynb` notebook in the `/evaluations/` folder that can be used to load and analyze the trained models.
