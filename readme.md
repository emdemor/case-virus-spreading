# Product Data Science - Data Analytics


## Preparando ambiente

* Clonar o repositório
```bash
git clone git@github.com:emdemor/case-neoway.git
```

* Configurar o ambiente conda
``` bash
conda env create -n neoway --file environment.yml
conda activate neoway
cd case-neoway
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