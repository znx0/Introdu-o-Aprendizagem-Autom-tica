# Intro à Aprendizagem Automática
* Fica aqui o PDF também caso queiras, e se fizeres alterações é só dares commit a tudo que isto atualiza

* O `.py` que meti aqui, já tem as funções todas feitas, mas é melhor rever tudo para termos a certeza e começarmos a tratar do resto do trabalho

* Criei o ficheiro `requirements.txt` caso não tenhas algumas bibliotecas necessárias para o correr do programa, mas podem faltar algumas.

## Como começar (para quem ainda não tem o repo)

1. **Clonar o repositório:**
```bash
   git clone https://github.com/znx0/Introdu-o-Aprendizagem-Autom-tica.git
   cd Introdu-o-Aprendizagem-Autom-tica
```

2. **(Recomendado mas não obrigatório) Criar um ambiente virtual**, para não misturar bibliotecas com outros projetos:
```bash
   python3 -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
```

3. **Instalar as bibliotecas necessárias:**
```bash
   pip install -r requirements.txt
```

4. **Correr o código:**
```bash
   python main.py Xtrain.pkl
```
   ou abrir `main.ipynb` no Jupyter / Google Colab.

## Fluxo de trabalho em grupo (git)

Antes de começares a trabalhar:
```bash
git pull
```
para garantires que tens as últimas alterações do teu colega.

Depois de fazeres alterações:
```bash
git add .
git commit -m "descrição curta do que mudaste"
git push
```

Se aparecer conflito ao fazer `git push` (o GitHub recusa porque há alterações que não tens localmente), faz primeiro `git pull`, resolve os conflitos que aparecerem nos ficheiros, e só depois `git push` outra vez.
