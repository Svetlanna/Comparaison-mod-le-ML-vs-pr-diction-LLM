import os
import re

import pandas as pd
from groq import Groq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


csv_path = r"data\tp2-opendata-titanic.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["Age", "Embarked"])

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})
y = df["Survived"]

# Les mêmes 20 passagers serviront au test du ML et du LLM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20, random_state=42
)


# entraîner et évaluer le modèle ML
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred_ml = model.predict(X_test)
accuracy_ml = accuracy_score(y_test, pred_ml)

print(f"Score ML : {accuracy_ml:.2f}")



exemples = df.loc[X_train.index[:8], features + ["Survived"]]
exemples_txt = exemples.to_csv(index=False)

passagers_test = df.loc[X_test.index, features]
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "Clé Groq absente. Définis la variable d'environnement GROQ_API_KEY."
    )

client = Groq(api_key=api_key)


def ask_llm(row):
    prompt = f"""Tu dois prédire la colonne Survived. Base toi seulement sur les exemples, ne va pas chercher sur internet
Survived = 0 signifie décédé ; Survived = 1 signifie survivant.

Voici 8 exemples avec la bonne réponse :
{exemples_txt}

Voici le passager à prédire :
{row.to_dict()}

Réponds uniquement par 0 ou 1, sans explication."""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=2048,
    )

    choix = completion.choices[0]
    reponse = (choix.message.content or "").strip()

    if not re.fullmatch(r"[01]", reponse):
        raise ValueError(
            f"Réponse LLM inattendue : {reponse!r} ; "
            f"finish_reason={choix.finish_reason!r}"
        )

    return int(reponse)
def evaluer_llm():
    predictions = [ask_llm(row) for _, row in passagers_test.iterrows()]
    return accuracy_score(y_test, predictions)


# deux passages sans changer les données ni le prompt
accuracy_llm_1 = evaluer_llm()
accuracy_llm_2 = evaluer_llm()

scores = pd.DataFrame(
    {
        "Méthode": ["ML", "LLM passage 1", "LLM passage 2"],
        "Accuracy": [accuracy_ml, accuracy_llm_1, accuracy_llm_2],
    }
)

print("\nTableau des scores :")
print(scores.to_string(index=False))
