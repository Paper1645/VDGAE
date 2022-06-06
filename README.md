# VDGAE: Variational Directed Graph Auto-Encoders by Reconstructing Incidence Matrix

A Pytorch Implementation for "VDGAE: Variational Directed Graph Auto-Encoders by Reconstructing Incidence Matrix".

**This is the source code for this paper.**

## Requirements

Please install all the requirements in `requirements.txt`. Because some datasets are implemented in Tensorflow, in addition to installing Pytorch, you also need to install Tensorflow.

## Datasets

The datasets used in this paper could be downloaded from the following link.

**WebKB**

```angular2html
https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/
```

**Cora**

```angular2html
https://relational.fit.cvut.cz/dataset/CORA
```

**Citeseer**

```
https://linqs.soe.ucsc.edu/data
```

**WikiCS**

```angular2html
https://github.com/pmernyei/wiki-cs-dataset
```


## Training Method

**Graph Reconstruction**

```angular2html
python train_rec.py
```

**Directed Link Prediction**

```angular2html
python train_direct.py
```

**Node Classification**

```
python train_node.py
```

