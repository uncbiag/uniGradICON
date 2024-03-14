# uniGradICON

This the official repository for `uniGradICON`: A Foundation Model for Medical Image Registration

`UniGradICON` is based on [GradICON](https://github.com/uncbiag/ICON) but trained on several different datasets (see details below). 
The result is a deep-learning based registration model that works well across datasets. 

![teaser](IntroFigure.jpg?raw=true)

[arXiv](https://arxiv.org/abs/2403.05780)

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

`UniGradICON` has currently been trained and tested on the following datasets.

<table>
    <tr>
        <td>Dataset</td>
        <td>Anatomical</td>
        <td># of</td>
        <td># per</td>
        <td># of</td>
        <td>Type</td>
        <td>Modality</td>
    </tr>
    <tr>
        <td></td>
        <td>region</td>
        <td>patients</td>
        <td>patient</td>
        <td>pairs</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>1. COPDGene</td>
        <td>Lung</td>
        <td>899</td>
        <td>2</td>
        <td>899</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>2. OAI</td>
        <td>Knee</td>
        <td>2532</td>
        <td>1</td>
        <td>3,205,512</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>3. HCP</td>
        <td>Brain</td>
        <td>1076</td>
        <td>1</td>
        <td>578,888</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>4. L2R-Abdomen</td>
        <td>Abdomen</td>
        <td>30</td>
        <td>1</td>
        <td>450</td>
        <td>Inter-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>5. Dirlab-COPDGene</td>
        <td>Lung</td>
        <td>10</td>
        <td>2</td>
        <td>10</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>6. OAI-test</td>
        <td>Knee</td>
        <td>301</td>
        <td>1</td>
        <td>301</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>7. HCP-test</td>
        <td>Brain</td>
        <td>32</td>
        <td>1</td>
        <td>100</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>8. L2R-NLST-val</td>
        <td>Lung</td>
        <td>10</td>
        <td>2</td>
        <td>10</td>
        <td>Intra-pat.</td>
        <td>CT</td>
    </tr>
    <tr>
        <td>9. L2R-OASIS-val</td>
        <td>Brain</td>
        <td>20</td>
        <td>1</td>
        <td>19</td>
        <td>Inter-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>10. IXI-test</td>
        <td>Brain</td>
        <td>115</td>
        <td>1</td>
        <td>115</td>
        <td>Atlas-pat.</td>
        <td>MRI</td>
    </tr>
    <tr>
        <td>11. L2R-CBCT-val</td>
        <td>Lung</td>
        <td>3</td>
        <td>3</td>
        <td>6</td>
        <td>Intra-pat.</td>
        <td>CT/CBCT</td>
    </tr>
    <tr>
        <td>12. L2R-CTMR-val</td>
        <td>Abdomen</td>
        <td>3</td>
        <td>2</td>
        <td>3</td>
        <td>Intra-pat.</td>
        <td>CT/MRI</td>
    </tr>
    <tr>
        <td>13. L2R-CBCT-train</td>
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
pip install git+https://github.com/uncbiag/uniGradICON/

wget http://www.slicer.org/w/img_auth.php/5/59/RegLib_C01_1.nrrd
wget http://www.slicer.org/w/img_auth.php/e/e3/RegLib_C01_2.nrrd

unigradicon-register --fixed=RegLib_C01_2.nrrd --moving=RegLib_C01_1.nrrd \
    --transform_out=trans.hdf5 --warped_moving_out=warped_C01_1.nrrd

```

## Plays well with others

`uniGradICON` is set up to work with itk images and transforms. So you can easily read and write images and display resulting transformations for example in [3D Slicer](https://www.slicer.org/).

The result can be viewed in Slicer
![result](slicer_output.png?raw=true)

