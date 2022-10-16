# Product Data Science - Data Analytics


## Preparando ambiente

* Clonar o repositório
```bash
git clone git@github.com:emdemor/case-spreading.git
```

* Configurar o ambiente conda
``` bash
cd case-spreading
conda env create -n case-spreading --file environment.yml
conda activate case-spreading
```

* Rodar o makefile
``` bash
sudo chmod +x make.sh
source make.sh
```

## Aplicação

* Treinamento do modelo
``` bash
train
```

* Treinamento do modelo com otimização de hiperparâmetros
``` bash
train --optimize=True
```
* Otimização de hiperparâmetros
``` bash
optimize
```

* Predição dos dados faltantes
``` bash
predict --output_file=data/predicted/prob_predictions.csv
```
