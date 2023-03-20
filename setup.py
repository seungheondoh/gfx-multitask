from setuptools import setup

setup(
    name="gfx",
    packages=["gfx"],
    install_requires=[
        'librosa >= 0.8',
        'torchaudio_augmentations==0.2.1', # for augmentation
        'gradio==3.22.1',
        'numpy',
        'pandas',
        'einops',
        'sklearn',
        'wandb',
        'jupyter',
        'matplotlib',
        'omegaconf',
    ]
)