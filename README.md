# <img src="ICON_universal_model.png" width="35" height="35"> uniGradICON: A Foundation Model for Medical Image Registration

[<img src="https://github.com/uncbiag/unigradicon/actions/workflows/test_readme_works.yml/badge.svg">](https://github.com/uncbiag/unigradicon/actions) [![arXiv](https://img.shields.io/badge/arXiv-2403.05780-b31b1b.svg)](https://arxiv.org/abs/2403.05780)

This the official repository for `uniGradICON`: A Foundation Model for Medical Image Registration

`uniGradICON` is based on [GradICON](https://github.com/uncbiag/ICON) but trained on several different datasets (see details below). 
The result is a deep-learning-based registration model that works well across datasets. More results can be found [here](/demos/Examples.md).

![teaser](IntroFigure.jpg?raw=true)

Please (currently) cite as:
```
@misc{tian2024unigradicon,
      title={uniGradICON: A Foundation Model for Medical Image Registration}, 
      author={Lin Tian and Hastings Greer and Roland Kwitt and Francois-Xavier Vialard and Raul San Jose Estepar and Sylvain Bouix and Richard Rushmore and Marc Niethammer},
      year={2024},
      eprint={2403.05780},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Training and testing data

`uniGradICON` has currently been trained and tested on the following datasets.

**Training data:**
<table>
    <tr>
        <td> </td>
        <td>Dataset</td>
        <td>Anatomical region</td>
        <td># of patients</td>
        <td># per patient</td>
        <td># of pairs</td>
        <td>Type</td>
        <td>Modality</td>
    </tr>
    <tr>
        <td>1.</td> 
        <td>COPDGene</td>
        <td>Lung</td>
        <td>899</td>
        <td>2</td>
        <td>899</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>2.</td> 
        <td>OAI</td>
        <td>Knee</td>
        <td>2532</td>
        <td>1</td>
        <td>3,205,512</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>3.</td> 
        <td>HCP</td>
        <td>Brain</td>
        <td>1076</td>
        <td>1</td>
        <td>578,888</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>4.</td>
        <td>L2R-Abdomen</td>
        <td>Abdomen</td>
        <td>30</td>
        <td>1</td>
        <td>450</td>
        <td>Inter-pat.</td>
        <td>CT</td>
    </tr>
</table>

**Testing data:**
<table>
    <tr>
        <td> </td>
        <td>Dataset</td>
        <td>Anatomical region</td>
        <td># of patients</td>
        <td># per patient</td>
        <td># of pairs</td>
        <td>Type</td>
        <td>Modality</td>
    </tr>     
    <tr>
        <td>5.</td>
        <td>Dirlab-COPDGene</td>
        <td>Lung</td>
        <td>10</td>
        <td>2</td>
        <td>10</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>6.</td>
        <td>OAI-test</td>
        <td>Knee</td>
        <td>301</td>
        <td>1</td>
        <td>301</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>7.</td>
        <td>HCP-test</td>
        <td>Brain</td>
        <td>32</td>
        <td>1</td>
        <td>100</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>8.</td>
        <td>L2R-NLST-val</td>
        <td>Lung</td>
        <td>10</td>
        <td>2</td>
        <td>10</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>9.</td>
        <td>L2R-OASIS-val</td>
        <td>Brain</td>
        <td>20</td>
        <td>1</td>
        <td>19</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>10.</td>
        <td>IXI-test</td>
        <td>Brain</td>
        <td>115</td>
        <td>1</td>
        <td>115</td>
        <td>Atlas-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>11.</td>
        <td>L2R-CBCT-val</td>
        <td>Lung</td>
        <td>3</td>
        <td>3</td>
        <td>6</td>
        <td>Intra-pat.</td>
        <td>CT/CBCT</td>
    </tr>
    <tr>
        <td>12.</td>
        <td>L2R-CTMR-val</td>
        <td>Abdomen</td>
        <td>3</td>
        <td>2</td>
        <td>3</td>
        <td>Intra-pat.</td>
        <td>CT/MRI</td>
    </tr>
    <tr>
        <td>13.</td>
        <td>L2R-CBCT-train</td>
        <td>Lung</td>
        <td>3</td>
        <td>11</td>
        <td>22</td>
        <td>Intra-pat.</td>
        <td>CT/CBCT</td>
    </tr>
</table>

## Get involved

Our goal is to continuously improve the `uniGradICON` model, e.g., by training on more datasets with additional diversity. Feel free to point us to datasets that should be included or let us know if you want to help with future developments.

## Easy to use and install

To use:

```
python3 -m venv unigradicon_virtualenv
source unigradicon_virtualenv/bin/activate

pip install unigradicon

wget https://www.hgreer.com/assets/slicer_mirror/RegLib_C01_1.nrrd
wget https://www.hgreer.com/assets/slicer_mirror/RegLib_C01_2.nrrd

unigradicon-register --fixed=RegLib_C01_2.nrrd --moving=RegLib_C01_1.nrrd \
    --transform_out=trans.hdf5 --warped_moving_out=warped_C01_1.nrrd

```
We also provide a [colab](https://colab.research.google.com/drive/1JuFL113WN3FHCoXG-4fiBTWIyYpwGyGy?usp=sharing) demo.

## Plays well with others

`UniGradICON` is set up to work with [Itk](https://itk.org/) images and transforms. So you can easily read and write images and display resulting transformations for example in [3D Slicer](https://www.slicer.org/).

The result can be viewed in 3D Slicer:
![result](slicer_output.png?raw=true)

