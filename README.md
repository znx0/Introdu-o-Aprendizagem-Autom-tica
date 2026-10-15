# Intro à Aprendizagem Automática

- Fica aqui também o PDF do enunciado já meio preenchido. Se fizeres alterações ao código ou aos documentos, basta fazeres commit e push para atualizar o repositório.
- Os ficheiros `.py` contêm as implementações dos algoritmos. Convém rever o código e os resultados antes de avançarmos para as restantes questões.
- O `requirements.txt` contém as bibliotecas necessárias para executar o projeto.

## Como começar

1. **Clonar o repositório:**

   ```bash
   git clone https://github.com/znx0/Introdu-o-Aprendizagem-Autom-tica.git
   cd Introdu-o-Aprendizagem-Autom-tica
   ```

2. **Criar e ativar um ambiente virtual:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   No Windows, a ativação é feita com:

   ```powershell
   venv\Scripts\activate
   ```

3. **Instalar as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

## Executar o código

Executar os scripts a partir da raiz do projeto, onde se encontra o `Xtrain.pkl`:

```bash
python kmeans.py Xtrain.pkl
python pca.py | tee saida_7a.txt (por exemplo)
```

## Guardar os resultados

Os resultados devem ficar organizados automaticamente nas pastas correspondentes, sem ser necessário guardar cada ficheiro manualmente:

- `figures/` — gráficos gerados durante as experiências, identificados pela questão correspondente (por exemplo, `PartII5a.jpg`).
- `outputs/` — resultados e informação impressos pelo programa, guardados em ficheiros `.txt`.

**Importante:** os scripts devem guardar os ficheiros nos caminhos corretos. Consulta o código de cada script para confirmar os nomes dos ficheiros gerados.

- Os gráficos/Imagens são guardados com, por exemplo:
```bash
plt.savefig(figures/"PARTII5a.jpg", dpi=300)
```

## Fluxo de trabalho com Git

Antes de começares a trabalhar, atualiza a tua cópia local:

```bash
git pull
```

Depois de alterares o código ou os documentos:

```bash
git status
git add .
git commit -m "Descrição curta das alterações"
git push
```

Se o `git push` for recusado porque existem alterações no GitHub que ainda não tens localmente, faz primeiro `git pull`, resolve eventuais conflitos e volta a tentar.

Não é necessário enviar o ambiente virtual `venv/` nem os ficheiros temporários de Python (`__pycache__/`).
