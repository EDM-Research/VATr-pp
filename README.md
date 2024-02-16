# Handwritten Text Generation from Visual Archetypes ++

This repository includes the code for training the VATr++ Styled Handwritten Text Generation model.

## Installation

```bash
conda create --name vatr python=3.9
conda activate vatr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/aimagelab/VATr.git && cd VATr
pip install -r requirements.txt
```

From [this folder](https://drive.google.com/drive/folders/13rJhjl7VsyiXlPTBvnp1EKkKEhckLalr?usp=sharing) you have to download the files `IAM-32.pickle` and `resnet_18_pretrained.pth` and place them into the `files` folder.

## Training

```bash
python train.py
```

Useful arguments:
```bash
python train.py
        --feat_model_path PATH  # path to the pretrained resnet 18 checkpoint. By default this is the synthetically pretrained model
        --is_cycle              # use style cycle loss for training
        --dataset DATASET       # dataset to use. Default IAM
        --resume                # resume training from the last checkpoint with the same name
        --wandb                 # use wandb for logging
```

Use the following arguments to apply full VATr++ training
```bash
python train.py
        --d-crop-size 64 128          # Randomly crop input to discriminator to width 64 to 128
        --text-augment-strength 0.4   # Text augmentation for adding more rare characters
        --file-suffix pa              # Use the punctuation attached version of IAM
        --augment-ocr                 # Augment the real images used to train the OCR model
```

### Pretraining dataset
The model `resnet_18_pretrained.pth` was pretrained by using this dataset: [download link](https://drive.google.com/drive/folders/1Xs_rR0EWt09-K6vmlvAI8pwsrmHSknC8?usp=share_link)


## Generate Styled Handwtitten Text Images

We added some utility to generate handwritten text images using the trained model. These are used as follows:

```bash
python generate.py [ACTION] --checkpoint files/vatrpp.pth
```

The following actions are available with their respective arguments.

Generate the given text for a custom author. Style samples for the author are needed. These can be automatically generated from an image of a page using `create_style_sample.py`

```bash
text  --text STRING     # String to generate
      --text-path PATH  # Optional path to text file
      --output PATH     # Optional output location, default: files/output.png
      --style-folder    # Optional stye folder containing writer samples, default: 'files/style_samples/00'
```

Generate some text for all authors of IAM.

```bash
authors --test-set  # Generate authors of test set, otherwise training set is generated
```