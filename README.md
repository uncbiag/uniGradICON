# uniGradICON
The official website for uniGradICON: A Foundation Model for Medical Image Registration

[paper](https://arxiv.org/abs/2403.05780)

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


To use:

```
pip install unigradicon

wget http://www.slicer.org/w/img_auth.php/5/59/RegLib_C01_1.nrrd
wget http://www.slicer.org/w/img_auth.php/e/e3/RegLib_C01_2.nrrd

unigradicon-register --fixed=RegLib_C01_2.nrrd --moving=RegLib_C01_1.nrrd \
    --transform_out=trans.hdf5 --warped_moving_out=warped_C01_1.nrrd

```

The result can be viewed in Slicer
![result](slicer_output.png?raw=true)
