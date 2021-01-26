# DECAF
DECAF: Deep Extreme Classification with Label Features

#### SETUP WORKSPACE
```
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
```

#### SETUP DECAF
```
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
```
cd ${HOME}/scratch/XC/data
gdown --id <dataset id>
unzip *.zip
```
| dataset                   | dataset id |
|---------------------------|------------|
| LF-AmazonTitles-131K      | <>         |
| LF-WikiSeeAlsoTitles-131K | <>         |
| LF-AmazonTitles-1.3M      | <>         |

#### RUNNING DECAF
```
cd ${HOME}/scratch/XC/programs/DECAF
chmod +x run_DECAF.sh
./run_DECAF.sh <gpu_id> <DECAF TYPE> <dataset> <folder name>
e.g.
./run_DECAF.sh 0 DECAF LF-AmazonTitles-131K DECAF_RUN
./run_DECAF.sh 0 DECAF-lite LF-AmazonTitles-131K DECAF_RUN

```