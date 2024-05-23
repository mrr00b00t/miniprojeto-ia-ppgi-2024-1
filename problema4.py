# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from pmlb import fetch_data


# BASE DE CONHECIMENTO
# Carregar base de dados sobre câncer de mama
_X, _y = fetch_data(
    'dermatology', return_X_y=True, local_cache_dir='datasets'
)

classes_names = [
    "psoriasis", "seboreic dermatitis",
    "lichen planus", "pityriasis rosea",
    "cronic dermatitis", "pityriasis rubra pilaris"
]

# Separar o conjunto de dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

# Normalizar os atributos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar o classificador SVC Linear e treinar
svc = LinearSVC()
svc.fit(X_train, y_train)

# Faz a predição nos dados de teste
y_pred = svc.predict(X_test)

# Avalia o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Mostra o relatório de classificação
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))

# Pegando colunas para ajudar o usuário do sistema
columns = fetch_data(
    'dermatology', local_cache_dir='datasets'
).columns.values
columns = np.delete(columns, np.where(columns == 'target'))

# Pegando novamente todos os dados para treinar um modelo sem split
# e selecionando uma amostra para guiar o usuário
__X, __y = fetch_data(
    'dermatology', return_X_y=True, local_cache_dir='datasets'
)
random_index = np.random.randint(low=0, high=__X.shape[0])
example = __X[random_index, :]
example_target = __y[random_index]
X_total = np.delete(__X, random_index, axis=0)
y_total = np.delete(__y, random_index, axis=0)

ajustador_entradas = StandardScaler()
X_ajustado = ajustador_entradas.fit_transform(X_total)

base_conhecimento = LinearSVC()
base_conhecimento.fit(X_ajustado, y_total)


def explicabilidade(
    svm_model: LinearSVC, scaler, sample, feature_names, class_names
):
    """
    Explica a predição de um modelo LinearSVC para um dado exemplo.

    Parâmetros:
    svm_model (LinearSVC): O modelo LinearSVC treinado.
    sample (array): A instância que será explicada.
    feature_names (list): Lista dos atributos.
    class_names (list): Lista dos nomes das classes.
    """
    # Faz a predição da instância
    sample = scaler.transform(sample.reshape(1, -1))
    predicted_label = svm_model.predict(sample)[0]

    # Pega os coeficientes e o intercepto
    coefficients = svm_model.coef_
    intercept = svm_model.intercept_

    decision_scores = np.dot(coefficients, sample[0]) + intercept

    print("Explicando a predição para a instância dada:")
    print(f"Legenda da classe predita: {class_names[predicted_label-1]}")
    print(f"Classe predita: {predicted_label}")
    print(f"Pontuação das decisões: {decision_scores}")
    print(f"Classes do modelo: {svm_model.classes_}")
    print(f"Legenda das classes: {class_names}")

    # Detailed breakdown
    for i, class_label in enumerate(class_names):
        print(f"Pontuação da classe {class_label} passo a passo:")
        for j, feature in enumerate(feature_names):
            contribution = coefficients[i][j] * sample[0][j]
            print(f"Atributo: {feature}, \
            Coeficiente: {coefficients[i][j]:.4f}, \
            Valor: {sample[0][j]:.4f}, \
            Contribuição: {contribution:.4f}")
        print(f"Intercepto: {intercept[i]:.4f}")
        print(f"Pontuação da decisão: {decision_scores[i]:.4f}\n")


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

print('Legenda:')
for i, class_name in enumerate(classes_names):
    print(f'{i+1}: {class_name}')

print(100*'=')

print(100*'=')
print('Explicabilidade da predição:')
res = explicabilidade(
    base_conhecimento, ajustador_entradas, user_information,
    columns, classes_names
)
