
# TCC - Fatoração de Matrizes Não Negativas (NMF) aplicada a Sistemas de Recomendação

autores: Leandro de Araújo Lima, Fernando Nascimento Oliveira, Lucas Lacerda Antunes

## Como executar o código

### Executando com Docker

1. Construa a imagem Docker:
	```zsh
	docker build -t nmf-recommender .
	```
2. Execute o container:
	```zsh
	docker run --rm nmf-recommender
	```

### Executando sem Docker

1. (Opcional, mas recomendado) Crie e ative um ambiente virtual:
	```zsh
	python3 -m venv .venv
	source .venv/bin/activate
	```
2. Instale as dependências:
	```zsh
	pip install -r requirements.txt
	```
3. Execute o script:
	```zsh
	python nmf_recommender.py
	```