# Product Data Science - Data Analytics


## Preparando ambiente

* Clonar o repositório
```bash
git clone git@github.com:emdemor/case-neoway.git
```

* Configurar o ambiente conda
``` bash
cd case-neoway
conda env create -n case-neoway --file environment.yml
conda activate case-neoway
```

* Configurar as variáveis de ambiente
``` bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

* Rodar o makefile
``` bash
sudo chmod +x make.sh
./make.sh
```