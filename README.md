# DigIA: Pipeline de Visão Computacional e Testes de Generalização Extrema

O **DigIA** é um projeto de Ciência de Dados e Visão Computacional focado no desenvolvimento, otimização e avaliação de modelos de Aprendizado de Máquina para o reconhecimento de dígitos manuscritos. O projeto utiliza o dataset clássico **MNIST** como base de referência e estende a avaliação para cenários de estresse metodológico, como detecção de dados fora de distribuição (OOD) e inferência em produção com dígitos capturados manualmente por meio de um pipeline em **OpenCV**.

---

## Estrutura do Projeto

O projeto foi desenvolvido em um único ambiente contido, estruturado da seguinte forma:

*   **`main` / `develop` (Git Branches):** Fluxo de desenvolvimento baseado em boas práticas de versionamento.
*   **`Fase 1: EDA`:** Análise exploratória, dimensionalidade, tipos de dados e balanceamento das 70.000 amostras originais.
*   **`Fase 2: Pré-processamento`:** Divisão estratificada e modelagem de escalonamento.
*   **`Fase 3: Modelagem & Tuning`:** Otimização via Grid Search com validação cruzada para as famílias SVM, Random Forest e MLP.
*   **`Fase 4: Avaliação`:** Matrizes de confusão e tabelas comparativas de desempenho.
*   **`Fase 5: Generalização`:** Testes de Robustez com oclusão de classes (OOD) e testes com dados reais (Imagens Próprias).

---

## Tecnologias e Bibliotecas Utilizadas

*   **Linguagem:** Python 3.10+
*   **Processamento de Dados:** `NumPy`, `Pandas`
*   **Machine Learning (Scikit-Learn):**
    *   `SVC` (Support Vector Classifier)
    *   `RandomForestClassifier`
    *   `MLPClassifier` (Multi-Layer Perceptron)
    *   `GridSearchCV` / `train_test_split` / `MinMaxScaler`
*   **Visão Computacional:** `OpenCV (cv2)`, `SciPy (ndimage)`
*   **Visualização de Dados:** `Matplotlib`, `Seaborn`

---

## Metodologia e Resultados por Fase

### Fase 1 & 2: EDA e Isolamento de Dados (Anti-Data Leakage)
A análise exploratória confirmou o balanceamento uniforme do dataset (~10% de representatividade para cada dígito). Para evitar o **Data Leakage (Vazamento de Dados)**, a divisão estratificada (80% treino / 20% teste) foi realizada *antes* de qualquer transformação estatística. 

O `MinMaxScaler` foi ajustado exclusivamente no conjunto de treino (`fit_transform`) e replicado no teste (`transform`), preservando a esparsidade das bordas pretas (valor `0.0`) e a intuição física das imagens.

### Fase 3 & 4: Treinamento e Avaliação Comparativa
O ajuste de hiperparâmetros foi realizado em uma subamostra estratificada de 9.000 registros para garantir eficiência computacional. Os modelos finais foram treinados na base completa (56.000 imagens).

#### Tabela Consolidada de Desempenho (Teste)

| Modelo | Acurácia | Precisão (Macro) | Recall (Macro) | F1-Score (Macro) | Tempo de Treino |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (Kernel RBF, C=10)** | **98.32%** | **98.32%** | **98.32%** | **98.32%** | ~3.10 min |
| **Rede Neural (MLP - 100, 50)** | **96.88%** | **96.90%** | **96.88%** | **96.88%** | ~0.35 min |
| **Random Forest (150 árvores)** | **96.68%** | **96.68%** | **96.68%** | **96.68%** | ~0.23 min |

*   **Análise Técnica:** O **SVM** obteve a maior acurácia global, mas o **MLP** apresentou a melhor relação custo-benefício (eficiência/tempo), rodando quase 9 vezes mais rápido com perda mínima de acurácia.
*   **Padrão de Erro Compartilhado:** As matrizes de confusão revelaram que a maior fragilidade dos modelos reside na separação geométrica entre as classes **4 e 9** (com destaque para o MLP, que registrou 48 falsos positivos nessa interseção), devido à sobreposição de pixels no topo das curvas manuscritas.

---

## Testes de Generalização Extrema (Fase 5)

### Desafio A: Oclusão de Classes (Out-of-Distribution - OOD)
As classes **3 e 6** foram completamente removidas do conjunto de treinamento. O modelo foi forçado a classificar um conjunto composto *apenas* por esses dois dígitos inéditos. 

*   **Resultado:** O modelo operou sob o fenômeno da **falsa certeza** (Acurácia de 0%), distribuindo as imagens desconhecidas nas fronteiras mais próximas: o dígito **3** foi mapeado em massa como **8** e **5**, enquanto o dígito **6** migrou predominantemente para as classes **4** e **5**.

### Desafio C: Produção Real com Dígitos Próprios
Para validar o sistema em um cenário de produção real, dígitos foram escritos à mão em papel, digitalizados via scanner (Grayscale, 300 dpi) e submetidos a um pipeline customizado de Visão Computacional:

1.  **Threshold Binarizador Rígido:** Isolamento do traço e eliminação de sombras.
2.  **Inversão de Cores:** Ajuste do papel branco com traço escuro para fundo preto com traço brilhante (`cv2.THRESH_BINARY_INV`).
3.  **Dilatação Morfológica:** Engordamento do traço fino da caneta comum para simular o padrão espesso do MNIST.
4.  **Redimensionamento com Bounding Box:** Redução proporcional para uma caixa interna de 20x20.
5.  **Centralização por Centro de Massa (`scipy.ndimage.center_of_mass`):** Alinhamento perfeito dos pixels baseando-se no centro de gravidade da imagem, idêntico ao processo padrão do MNIST.

*   **Resultado:** Após a correção de clonagem de memória (`copy.deepcopy`) do modelo e o tratamento morfológico, as predições reais obtiveram **altíssima taxa de sucesso**, quebrando o viés de predição nula e validando o pipeline de ponta a ponta.

---

## Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com
   ```
2. Crie e ative um ambiente virtual (venv):
   ```bash
   python -m venv venv
   # No Windows:
   .\venv\Scripts\activate
   # No Mac/Linux:
   source venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute o Jupyter Notebook para visualizar os gráficos e treinamentos:
   ```bash
   jupyter notebook
   ```

---
*Projeto desenvolvido como critério de avaliação acadêmica para consolidação de conceitos de pipelines de Machine Learning e processamento digital de imagens.*

Link para video de apresentação: https://youtu.be/i11eZK2f-hY
