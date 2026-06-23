# TP complémentaire — « Sauvetage d'un jeu de données »

**Séquence 2 : Données & Outils — Nettoyage et préparation**
Travail individuel ou en binôme · Outils : NumPy, Pandas, Matplotlib/Seaborn (aucun ML)

---

## Contexte (scénario)

Un client vous transmet un export « brut » de sa base de commandes en ligne. Le fichier est **inexploitable en l'état** : doublons, valeurs manquantes, dates dans trois formats différents, montants stockés en texte avec des `€`, des âges impossibles, et des libellés incohérents (`M`, `Homme`, `male`…).

> **Votre mission :** livrer un DataFrame **propre, typé et exploitable**, prêt à être analysé. C'est exactement le travail qui occupe **70 à 80 %** du temps d'un data scientist en conditions réelles.

### Objectifs pédagogiques

- Diagnostiquer la qualité d'un jeu de données réel (et pas un dataset « jouet »).
- Distinguer les **vrais `NaN`** des fausses valeurs manquantes (`"nan"` en texte, chaîne vide…).
- Maîtriser : dédoublonnage, normalisation de chaînes, conversion de types, parsing de dates multi-formats.
- Détecter et traiter les **valeurs aberrantes** (outliers).
- Justifier chaque choix d'imputation.

---

## Mission 0 — Générer le jeu de données sale

Le dataset est généré directement dans le notebook (pas de fichier à télécharger). **Copiez-collez ce bloc tel quel** et exécutez-le.

```python
import numpy as np
import pandas as pd

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
```

Vous devez obtenir un DataFrame de **215 lignes × 8 colonnes**. Gardez précieusement ce `df` : c'est votre matière première.

---

## Mission 1 — Diagnostic

Avant de nettoyer, **on observe**. Répondez par du code :

1. Affichez les dimensions, les types (`df.info()`) et un aperçu (`df.head()`).
2. Combien de **valeurs manquantes** par colonne ? (`isna().sum()`)
3. Combien de **doublons exacts** ? (`duplicated().sum()`)
4. Regardez `df["sexe"].value_counts(dropna=False)` et `df["ville"].value_counts()`. **Que remarquez-vous ?**

> **Question piège — à débattre en groupe :** la colonne `sexe` affiche-t-elle vraiment toutes ses valeurs manquantes avec `isna()` ? Pourquoi ? (Indice : que vaut `np.random.choice` quand on mélange du texte et un `np.nan` ?)

---

## Mission 2 — Valeurs manquantes

1. Listez les colonnes concernées.
2. Pour `age` : choisissez une stratégie d'imputation (moyenne ? médiane ?) et **justifiez** votre choix. (Indice : regardez d'abord la Mission 5 — y a-t-il des valeurs extrêmes qui fausseraient la moyenne ?)
3. Pour `montant_achat` : même réflexion.
4. Pour `sexe` : faut-il imputer, ou créer une catégorie `"Inconnu"` ? Discutez le pour/contre.

> **Ne traitez pas encore `montant_achat` ici** s'il est toujours en texte — il faut d'abord le convertir (Mission 4). C'est volontaire : ça montre que **l'ordre des étapes compte**.

---

## Mission 3 — Doublons & cohérence des catégories

1. Supprimez les doublons exacts. Combien de lignes reste-t-il ? (attendu : 200)
2. Nettoyez la colonne `nom` : enlevez les espaces superflus et harmonisez la casse (`strip` + `capitalize`).
3. **Harmonisez la colonne `sexe`** vers seulement deux modalités `H` / `F`.
   - Pensez d'abord à convertir la fausse valeur `"nan"` (texte) en vrai `NaN`.
   - Construisez un dictionnaire de mapping et appliquez `.map()`.
4. Vérifiez avec `value_counts(dropna=False)`.

---

## Mission 4 — Normalisation des types & formats

C'est le cœur du TP.

1. **`ville` et `pays`** : supprimez les espaces et uniformisez la casse (`str.strip().str.title()`). Combien de villes uniques *réelles* obtenez-vous ?
2. **`montant_achat`** : transformez le texte (`"220,83 €"`) en **`float`**.
   - Retirez le `€`, remplacez la `,` par un `.`, gérez les chaînes vides → `NaN`, puis `astype(float)`.
3. **`date_inscription`** : les dates arrivent en **3 formats** (`2022-05-22`, `15/08/2022`, `11-02-2022`). Convertissez-les toutes en `datetime` avec `pd.to_datetime(..., format="mixed", dayfirst=True, errors="coerce")`.
   - Vérifiez qu'il n'y a **aucun `NaT`** (date non reconnue).

> **À retenir :** tant qu'une colonne numérique est en `object` (texte), **aucun calcul statistique n'est possible**. La conversion de type n'est pas un détail cosmétique.

---

## Mission 5 — Valeurs aberrantes (outliers)

La colonne `age` contient des valeurs impossibles (`-5`, `0`, `199`, `999`).

1. Affichez `df["age"].describe()` et un `boxplot` (`sns.boxplot`) **avant** traitement. Repérez les extrêmes.
2. Définissez une règle métier raisonnable (ex. : un client a entre **16 et 100 ans**). Remplacez tout ce qui sort de cette plage par `NaN`.
3. Imputez ensuite ces `NaN` (médiane recommandée — voir Mission 2).
4. Re-affichez `describe()` : la moyenne et l'écart-type ont-ils changé ? **Pourquoi est-ce important ?**

> **Discussion :** supprimer la ligne, ou corriger la valeur ? Quand choisit-on l'un ou l'autre ?

---

## Mission 6 — Synthèse & visualisation

Votre `df` doit maintenant être **propre**. Prouvez-le et explorez :

1. Vérification finale : `df.isna().sum()` (que des zéros) et `df.dtypes` (types corrects).
2. Encodez `sexe` en numérique (`H`→0, `F`→1, `Inconnu`→-1).
3. Produisez **2 visualisations** au choix :
   - répartition des clients par `ville` (`countplot`) ;
   - `montant_achat` moyen par `sexe` (`barplot`) ;
   - distribution de `age` (`histplot`) ;
   - heatmap de corrélation des colonnes numériques (`sns.heatmap(df.corr(numeric_only=True), annot=True)`).
4. **Livrable :** un notebook propre intitulé « Sauvetage Commandes » avec, en commentaire, la liste des problèmes trouvés et la stratégie retenue pour chacun.

---

## Extension

- Créez une colonne `anciennete_jours = (pd.Timestamp.today() - df["date_inscription"]).dt.days`.
- Détectez les outliers de `montant_achat` avec la **méthode de l'IQR** (Q1 − 1.5·IQR / Q3 + 1.5·IQR) au lieu d'une règle fixe.
- Écrivez une **fonction `nettoyer(df)`** qui encapsule tout le pipeline et renvoie le DataFrame propre. (Bonne transition vers la réutilisabilité du code.)