############################################################################################
# MODELOS DE MACHINE LEARNING: Introducao em PYTHON
#####################################################
# Aluno: Mario Diego Rocha Valente
# Habilidades: Estatistico, Especialista em Bioestatistica + Controle de Qualidade
# https://github.com/MarioDhiego
# UFPA: graduando em Sistema de Informacao
# Mat. 202211140042
############################################################################################


############################################################################################
# Passo 1: Instalacao de Pacotes no Python
# Via Terminal

(pip install + Nome do pacote)
..................
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
................
###########################################################################################


###########################################################################################
# Passo 2: Ativar os Pacotes

# Manipulacaoo de Dados
import pandas as pd
import numpy as np

# Transformar em Dummys
import category_encoders as ce

# Visualizacao Grafica
import seaborn as sns
import matplotlib.pyplot as plt 

# Graficos Dinamicos
import plotly.express as px

# Dashboard 
import sweetviz as sv


# Machine Learning/Treinamento
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Arvore de Decisao
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Floresta Aleatatoria
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn.preprocessing import LabelEncoder



###################################################################################


####################################################################################
# Ler a Base de dados Vinho 
# Fonte: Kagge
wine = pd.read_csv("C:/Users/mario Dhiego/Documents/Machine_Learning_Python/Machine_Learning_Python/wine_dataset.csv")
wine
#####################################################################################
# Estatistica Descritiva

# Tamanho
Wine.shape()

# minimo, maximo, q1, q3, media, mediana
wine1.describe()

# % de Missing
wine1.isna().mean()

# Numero de missing
wine1.isna().sum()

# Variaveis Categoricas
wine1.Housing.value_counts()

# Nome das colunas
wine1.columns

# 5 primeiras linhas
wine1.head()

# 5 ultimas linhas
wine1.tail()

# Agrupar por tipo de vinho com media
wine1.groupby("style")["alcohol", "residual_sugar"].mean()


# Categorizar Tipo de Vinho
wine['style'] = wine['style'].replace('red', 0)
wine['style'] = wine['style'].replace('white', 1)
wine

# Separar (Preditora e Variavel Alvo)
y = wine["style"]
x = wine.drop("style",axis=1, inplace=True)


######################################################################
# Visualização Grafica

fig1 = sns.relplot(x="alcohol", y="pH", hue="style", data=wine1)

# Dashboard
report = sv.analyze(wine1)
report.show_html()


#####################################################################


#####################################################################
# Modelagem de Machine Learning

# Ativar a funcao do Pacote Sikt-Learning
from sklearn.model_selection import train_test_split

# Dividir Base : Treino x Teste
x_treino,x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3) 

# Valores Divididos
x_treino  # x_treino = 4547)
x_teste   # x_teste = 1950
 
y_treino
y_teste


# Ativar a função da Arvore Decisão
from sklearn.ensemble import ExtraTreesClassifier

### Modelo1 via Arvore de Decisão
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino, y_treino)

# Resultado Modelo 1
resultado = modelo.score(x_teste, y_teste)
print("Acur?cia", resultado)


# Aplicar no Banco de Teste1

y_teste[500:505]
x_teste[500:505]

previsoes1= modelo.predict(x_teste[500:505])
previsoes



### Modelo2 via Floresta Aleatoria ########################################
# Ativar a funcao da Floresta Aleatoria
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint

# Rodar Modelo2
modelo2 = RandomForestClassifier()



# Treinar Modelo2
modelo2.fit(x_treino, y_treino)


# Aplicar no Banco de Teste2

y_teste[1500: 2500]
x_teste[1500: 2500]

y_pred2 = modelo2.predict(x_teste)
y_pred2


# Ativar a funcao de Metricas de avaliacao
from sklearn.metrics import accuracy_score

resultado2 = modelo2.score(x_teste, y_teste)
print("Acuracia", + str(accuracy_score(y_teste, y_pred2))


from sklearn.metrics import f1_score
# Quanto + próimo de 1 melhor
print("F1 Score : {}".format(f1_score(y_teste, y_pred2)))


from sklearn.metrics import confusion_matrix
print ("Matriz de Confusão : \n" + str(confusion_matrix(y_teste, y_pred2)))

tn, fp, fn, tp = confusion_matrix(y_teste, y_pred2).ravel()
print("True Positive :" +str(tp))
print("True Negative :" +str(tn))
print("False Positive :" +str(fp))
print("False Negative :" +str(fn))




# Curva Roc 
 from sklearn.metrics import roc_auc_score
 from sklearn.selction import cross_val_predict

# Probabilidades
y_scores = cross_val_predict(resultado2, x_teste, y_teste)

fpr, tpr, thressholds = roc_curve(y_teste, y_scores)


def plot_roc_curve(fpr, tpr, label=Nome):
  plt.plot(fpr,tpr, linewidth=2, label=label)
  plt.plot([0,1], [0,1], "k--")
  plt.axis([0,1, 0,1])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plot_roc_curve(fpr, tpr)

# Importancia da Variavel

feature_imp = pd.Series(resultado2.feature_importances_,index=x_teste.columns).sort_values(asceding=False)
feature_imp
