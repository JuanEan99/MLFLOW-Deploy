# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py  
validate:
	python src/validate.py 