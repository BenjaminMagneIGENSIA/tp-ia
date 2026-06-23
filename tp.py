# Mission 0 — Générer le jeu de données sale :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
N = 200

prenoms = ["Lucas","Emma","  Hugo","NOÉMIE","chloé","Gabriel ","Léa","Nathan","jade","Louis",
           "Manon","Arthur","Camille","Raphaël","Sarah","Tom","inès","Paul","Lina","Adam"]
villes_brutes = ["Paris","paris","PARIS","Lyon","lyon","Marseille","marseille ","Toulouse",
                 "toulouse","Bordeaux","Lille","Nantes","nantes","Nice","Strasbourg"]
pays = ["France","france","FRANCE","Belgique","Suisse","France "]

rows = []
for i in range(N):
    nom = np.random.choice(prenoms)
    r = np.random.rand()
    if r < 0.08:
        age = np.nan
    elif r < 0.12:
        age = np.random.choice([-5, 0, 199, 999])   # âges aberrants
    else:
        age = max(int(np.random.normal(38, 12)), 16)
    sexe = np.random.choice(["M","F","Homme","Femme","male","female","H", np.nan],
                            p=[0.18,0.18,0.12,0.12,0.12,0.12,0.10,0.06])
    ville = np.random.choice(villes_brutes)
    p = np.random.choice(pays)
    jour, mois, annee = np.random.randint(1,28), np.random.randint(1,13), np.random.choice([2021,2022,2023,2024])
    fmt = np.random.choice([0,1,2])
    date = (f"{annee}-{mois:02d}-{jour:02d}" if fmt==0
            else f"{jour:02d}/{mois:02d}/{annee}" if fmt==1
            else f"{jour:02d}-{mois:02d}-{annee}")
    montant = "" if np.random.rand() < 0.07 else f"{round(np.random.uniform(9.9,450.0),2):.2f} €".replace(".",",")
    rows.append([1000+i, nom, age, sexe, ville, p, date, montant])

df = pd.DataFrame(rows, columns=["id_client","nom","age","sexe","ville","pays","date_inscription","montant_achat"])

# on injecte 15 doublons exacts puis on mélange
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.head(10)
## Mission 1 - Diagnostic :
print()
print("1. Affichez les dimensions, les types (df.info()) et un aperçu (df.head()).")
print(df.info())
print(df.head())
print()
print("2. Combien de valeurs manquantes par colonne ? (isna().sum())")
for columns in df.columns:
    print(f"Colonne {columns} : {df[columns].isna().sum()} valeur(s) manquante(s)")
print()
print("3. Combien de doublons exacts ? (duplicated().sum())")
print(f"{df.duplicated().sum()} doublon(s) exact(s).")
print()
print("4. Regardez df['sexe'].value_counts(dropna=False) et df['ville'].value_counts(). Que remarquez-vous ?")
print(f"4.1 : {df['sexe'].value_counts(dropna=False)}")
print(f"4.2 : {df['ville'].value_counts()}")
print("Les colonnes ne sont pas standardisées : il y a des doublons.")
print()
# Mission 2 - Valeurs manquantes
print()
print("1. Listez les colonnes concernées.")
colonnes_manquantes = df.columns[df.isna().any()].tolist()
colonnes_manquantes += [col for col in ["sexe", "montant_achat"] if col not in colonnes_manquantes]
print(colonnes_manquantes)
print()
print("2. Pour age : stratégie d'imputation.")
age_mediane = df.loc[df["age"].between(16, 100), "age"].median()
print(f"Médiane retenue après exclusion des valeurs aberrantes : {age_mediane}")
print()
print("3. Pour montant_achat : même réflexion.")
print("Traitement après conversion en float.")
print()
print('4. Pour sexe : création de la catégorie "Inconnu".')
print()

## Mission 3 - Doublons & cohérence des catégories :
print()
print("1. Supprimez les doublons exacts. Combien de lignes reste-t-il ?")
df = df.drop_duplicates().reset_index(drop=True)
print(f"{len(df)} ligne(s) restante(s).")
print()
print("2. Nettoyez la colonne nom.")
df["nom"] = df["nom"].str.strip().str.capitalize()
print(df["nom"].head())
print()
print("3. Harmonisez la colonne sexe.")
df["sexe"] = df["sexe"].replace("nan", np.nan)
mapping_sexe = {
    "M": "H",
    "H": "H",
    "Homme": "H",
    "male": "H",
    "F": "F",
    "Femme": "F",
    "female": "F",
}
df["sexe"] = df["sexe"].map(mapping_sexe).fillna("Inconnu")
print()
print("4. Vérifiez avec value_counts(dropna=False).")
print(df["sexe"].value_counts(dropna=False))
print()

## Mission 4 - Normalisation des types & formats :
print()
print("1. ville et pays : nettoyage et nombre de villes uniques.")
df["ville"] = df["ville"].str.strip().str.title()
df["pays"] = df["pays"].str.strip().str.title()
print(f"{df['ville'].nunique()} ville(s) unique(s).")
print()
print("2. montant_achat : conversion en float.")
df["montant_achat"] = (
    df["montant_achat"]
    .replace("", np.nan)
    .str.replace("€", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
    .astype(float)
)
montant_mediane = df["montant_achat"].median()
df["montant_achat"] = df["montant_achat"].fillna(montant_mediane)
print(df["montant_achat"].head())
print()
print("3. date_inscription : conversion en datetime.")
df["date_inscription"] = pd.to_datetime(
    df["date_inscription"],
    format="mixed",
    dayfirst=True,
    errors="coerce",
)
print(f"{df['date_inscription'].isna().sum()} date(s) non reconnue(s).")
print()