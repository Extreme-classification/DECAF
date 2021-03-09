# DECAF
[DECAF: Deep Extreme Classification with Label Features](http://manikvarma.org/pubs/mittal21-main.pdf)
```bib
@InProceedings{Mittal21,
    author = "Mittal, A. and Dahiya, K. and Agrawal, S. and Saini, D. and Agarwal, S. and Kar, P. and Varma, M.",
    title = "DECAF: Deep Extreme Classification with Label Features",
    booktitle = "Proceedings of the ACM International Conference on Web Search and Data Mining",
    month = "March",
    year = "2021",
    }
```

#### SETUP WORKSPACE
```bash
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
```

#### SETUP DECAF
```bash
cd ${HOME}/scratch/XC/programs
git clone https://github.com/Extreme-classification/DECAF.git
conda create -f DECAF/decaf_env.yml
conda activate decaf
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install
cd ../DECAF
```

#### DOWNLOAD DATASET
```bash
cd ${HOME}/scratch/XC/data
gdown --id <dataset id>
unzip *.zip
```
| dataset                   | dataset id                        |
|---------------------------|-----------------------------------|
| LF-AmazonTitles-131K      | 1VlfcdJKJA99223fLEawRmrXhXpwjwJKn |
| LF-WikiSeeAlsoTitles-131K | 1edWtizAFBbUzxo9Z2wipGSEA9bfy5mdX |
| LF-AmazonTitles-1.3M      | 1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO |

#### RUNNING DECAF
```bash
cd ${HOME}/scratch/XC/programs/DECAF
chmod +x run_DECAF.sh
./run_DECAF.sh <gpu_id> <DECAF TYPE> <dataset> <folder name>
e.g.
./run_DECAF.sh 0 DECAF LF-AmazonTitles-131K DECAF_RUN
./run_DECAF.sh 0 DECAF-lite LF-AmazonTitles-131K DECAF_RUN

```
