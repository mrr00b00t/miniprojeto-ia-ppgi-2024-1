from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pmlb import fetch_data
import numpy as np
from sklearn.tree import _tree

# TREINO DO MODELO / GERAÇÃO DA BASE DE CONHECIMENTO ATRAVÉS DE DADOS
# Returns a pandas DataFrame
# 1 = risco de crédito BOM
# 0 = risco de crédito RUIM
_X, _y = fetch_data(
    'credit_g', return_X_y=True, local_cache_dir='datasets'
)

# Separa o conjutno de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

# Define os parâmetros da grade
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'ccp_alpha': [0.0, 0.1, 0.01, 0.001, 0.0001]
}

# Inicializa a árvore de decisão
clf = DecisionTreeClassifier()

# Inicializa a busca em grade
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='accuracy')

# Ajusta modelo com a busca em grade
print('Procurando melhor modelo...')
grid_search.fit(X_train, y_train)
print('Melhor modelo encontrado!')

# Pega o melhor modelo e melhores parâmetros
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Faz predições no conjunto de teste usando o melhor modelo
y_pred = best_model.predict(X_test)

# Aavalia o modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(100*'=')
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print(100*'=')

# Pegando colunas para ajudar o usuário do sistema
columns = fetch_data('credit_g', local_cache_dir='datasets').columns.values
columns = np.delete(columns, np.where(columns == 'target'))

# Pegando novamente todos os dados para treinar um modelo sem split
# e selecionando uma amostra para guiar o usuário
X, y = fetch_data('credit_g', return_X_y=True, local_cache_dir='datasets')
random_index = np.random.randint(low=0, high=X.shape[0])
example = X[random_index, :]
example_target = y[random_index]
X = np.delete(X, random_index, axis=0)
y = np.delete(y, random_index, axis=0)

base_conhecimento = DecisionTreeClassifier(**best_params)
base_conhecimento.fit(X, y)


def explicabilidade(tree, feature_names, instance):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    path = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if instance[tree_.feature[node]] <= threshold:
                path.append(f"{name} <= {threshold:.2f}")
                recurse(tree_.children_left[node], depth + 1)
            else:
                path.append(f"{name} > {threshold:.2f}")
                recurse(tree_.children_right[node], depth + 1)
        else:
            path.append(f"Predict probability: {tree_.value[node]}")

    recurse(0, 1)

    return path


def interface_usuario(fields):

    user_info = []

    print("Digite as seguintes informações (como números float):")

    for field in fields:
        while True:
            replaced_field = field.replace('_', ' ').capitalize()
            try:
                value = float(input(f"{replaced_field}: "))
                user_info.append(value)
                break
            except ValueError:
                print(
                    f"Entrada inválida para {replaced_field}. \
                    Por favor digite um número float válido."
                )

    return user_info


def engenho_inferencia(clf, instance):
    return clf.predict(instance.reshape(1, -1))[0]


# Exemplo de uso
print(100*'=')
print('Required information:', columns)
print('Suggested information:')

_suggestion = {c: v for c, v in zip(columns, example)}
print(_suggestion)

user_information = interface_usuario(columns)
user_information = np.array(user_information)
print("Collected User Information:")
print(user_information)
print(100*'=')

print(100*'=')
res = engenho_inferencia(base_conhecimento, user_information)
print('Inferência para:', user_information)
print('Classe predita:', res)
print('Legenda: 1 para risco de crédito BOM e 0 para risco crédito RUIM')
print(100*'=')

print(100*'=')
res = explicabilidade(base_conhecimento, columns, user_information)
print('Explicabilidade da predição:')
print(res)
print(100*'=')
