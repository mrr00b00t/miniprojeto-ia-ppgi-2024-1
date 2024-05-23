# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from pmlb import fetch_data


# BASE DE CONHECIMENTO
# Carregar base de dados sobre câncer de mama
_X, _y = fetch_data(
    'breast_cancer', return_X_y=True, local_cache_dir='datasets'
)

# Separar o conjunto de dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

# Normalizar os atributos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ajustar os parâmetros para a busca em grade
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Inicializar o classificador KNN
knn = KNeighborsClassifier()

# Inicializar a busca em grade para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(
    estimator=knn, param_grid=param_grid,
    cv=5, scoring='accuracy', n_jobs=-1
)

# Executar a busca em grade
print('Procurando melhor modelo...')
grid_search.fit(X_train, y_train)
print('Melhor modelo encontrado!')

# Mostrar os melhores hiperparâmetros
print("Melhores hiperparâmetros encontrados: ", grid_search.best_params_)

# Pega o melhor modelo e melhores hiperparâmetros
best_params = grid_search.best_params_
best_knn = grid_search.best_estimator_

# Faz a predição nos dados de teste
y_pred = best_knn.predict(X_test)

# Avalia o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Mostra o relatório de classificação e melhores parâmetros
print('Melhores parâmetros:')
print(best_params)
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))

# Pegando colunas para ajudar o usuário do sistema
columns = fetch_data(
    'breast_cancer', local_cache_dir='datasets'
).columns.values
columns = np.delete(columns, np.where(columns == 'target'))

# Pegando novamente todos os dados para treinar um modelo sem split
# e selecionando uma amostra para guiar o usuário
__X, __y = fetch_data(
    'breast_cancer', return_X_y=True, local_cache_dir='datasets'
)
random_index = np.random.randint(low=0, high=__X.shape[0])
example = __X[random_index, :]
example_target = __y[random_index]
X_total = np.delete(__X, random_index, axis=0)
y_total = np.delete(__y, random_index, axis=0)

ajustador_entradas = StandardScaler()
X_ajustado = ajustador_entradas.fit_transform(X_total)

base_conhecimento = KNeighborsClassifier(**best_params)
base_conhecimento.fit(X_ajustado, y_total)


def explicabilidade(model, X_train, y_train, instance, k):
    """
    Explica a predição de um modelo KNN para uma instância de teste.

    Parâmetros:
    modelo (KNeighborsClassifier): Modelo KNN treinado.
    X_train (numpy.ndarray): Dados dos atributos de treino.
    y_train (numpy.ndarray): Dados dos rótulos de treino.
    instance (numpy.ndarray): Instância teste para ser explicada.
    k (int): Número de vizinhos para considerar.

    Retorna:
    dict: Explicação da predição.
    """
    # Encontra os k vizinhos mais próximos
    distances, indices = model.kneighbors(
        instance.reshape(1, -1), n_neighbors=k
    )

    # Pega os rótulos dos vizinhos mais próximos
    neighbor_labels = y_train[indices[0]]

    # Conta a ocorrência de cada classe nos vizinhos
    class_counts = np.bincount(neighbor_labels)

    # Pega a classe predita
    predicted_class = np.argmax(class_counts)

    explanation = dict()
    for i, abcd in enumerate(
        zip(indices[0], X_train[indices[0]], distances[0], neighbor_labels)
    ):
        idx, att, dst, lbl = abcd
        explanation.update({
            f'índices do vizinho {i}': idx,
            f'Atributos do vizinho {i}': att,
            f'Distância para o vizinho {i}': dst,
            f'Rótulo do vizinho {i}': lbl
        })

    explanation.update({
        "Instância de teste": instance,
        "Contagem das classes": class_counts,
        "Classe predita": predicted_class
    })

    return explanation


def interface_usuario(fields):

    user_info = []

    print("Por favor digite as informações a seguir (como número float):")

    for field in fields:
        while True:
            field_replaced = field.replace('_', ' ').capitalize()
            try:
                value = float(
                    input(
                        f"{field_replaced}: "
                    )
                )
                user_info.append(value)
                break
            except ValueError:
                print(
                    f"Entrada inválida para {field_replaced}. \
                    Por favor digite um número float válido."
                )

    return user_info


def engenho_inferencia(clf, scaler, instance):
    _instance = scaler.transform(instance.reshape(1, -1))
    return clf.predict(_instance)[0]


# Example usage
print(100*'=')
print('Informações necessárias:', columns)
print('Informação sugerida:')

_suggestion = {c: v for c, v in zip(columns, example)}
print(_suggestion)

user_information = interface_usuario(columns)
user_information = np.array(user_information)
print("Informações coletadas do usuário:")
print(user_information)
print(100*'=')

print(100*'=')
res = engenho_inferencia(
    base_conhecimento, ajustador_entradas, user_information
)
print('Inferência para:', user_information)
print(f'Inferência: {res}')
print('Legenda: 1 para risco de câncer de mama POSITIVO e 0 para NEGATIVO')
print(100*'=')

print(100*'=')
res = explicabilidade(
    base_conhecimento, X_ajustado, y_total,
    user_information, best_params['n_neighbors']
)
print('Explicabilidade da predição:')

for k, v in res.items():
    print(f'{k}: {v}')

print(100*'=')

recomendations = {
    1: 'Se você recebeu um diagnóstico positivo de câncer de mama, é crucial \
    seguir algumas recomendações importantes: agende consultas com um \
    oncologista ou mastologista para discutir seu diagnóstico e plano de \
    tratamento, que pode incluir cirurgia, quimioterapia, radioterapia e/ou \
    terapia hormonal. Realize todos os exames necessários, como mamografia, \
    ultrassonografia e ressonância magnética, para avaliar a extensão do \
    câncer. Informe-se sobre o seu tipo específico de câncer e suas opções \
    terapêuticas, buscando também uma segunda opinião médica se desejar. Além \
    disso, procure apoio emocional, seja através de grupos de apoio ou \
    aconselhamento psicológico, para lidar com o impacto emocional do \
    diagnóstico. Adote um estilo de vida saudável, mantendo uma alimentação \
    equilibrada e realizando atividades físicas regularmente, e evite álcool \
    e tabaco. Monitorize regularmente seu progresso com seu médico e siga \
    todas as orientações de tratamento e acompanhamento.',
    0: 'Para quem não tem suspeita de câncer de mama e deseja se prevenir, \
    é essencial adotar um estilo de vida saudável e realizar exames de \
    rastreamento regulares. Mantenha uma dieta equilibrada rica em frutas, \
    vegetais e grãos integrais, pratique atividade física regularmente, \
    evite o consumo excessivo de álcool e não fume. Realize mamografias \
    periódicas conforme a orientação médica, especialmente se você tiver \
    mais de 40 anos ou histórico familiar de câncer de mama. Aprenda a fazer \
    o autoexame das mamas mensalmente para conhecer o aspecto normal do seu \
    corpo e identificar qualquer alteração rapidamente. Considere também \
    discutir com seu médico a possibilidade de testes genéticos se houver \
    histórico familiar significativo. A detecção precoce e um estilo de vida \
    saudável são fundamentais para reduzir o risco de câncer de mama.'
}
print('Recomendações para a predição da sua entrada:')
print(recomendations[res['Classe predita']])
