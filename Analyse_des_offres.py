import json
from datetime import datetime
import os
import requests


base_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(base_dir, "offres_analyste.json")

with open(json_file_path, "r", encoding="utf-8") as f:
    data= json.load(f)

#competence > libelle

offres = data if isinstance(data, list) else [data]

#competence_unique = set()
#for offre in offres:
#    for item in offre.get("competences", []):
#        lib = item.get("libelle")
#       if lib:
#            competence_unique.add(lib)

#print(competence_unique)
#print(len(competence_unique))

#updated_path = os.path.join(base_dir, "competence_un.json")
#with open(updated_path, "w", encoding="utf-8") as f:
 #   json.dump(list(competence_unique), f, indent=2, ensure_ascii=False)



import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from collections import Counter

# DICTIONNAIRES
OUTILS_TECH = {
    'Python': ['python', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'],
    'R': ['langage r', ' r ', 'rstudio', 'ggplot', 'dplyr', 'tidyverse'],
    'SQL': ['sql', 'mysql', 'postgresql', 'postgres', 'oracle', 't-sql', 'tsql', 'pl/sql', 'plsql', 'requête', 'requêtes', 'base de données', 'bases de données', 'sgbd'],
    'SAS': ['sas'],
    'VBA': ['vba', 'macro', 'macros excel'],
    'Power BI': ['power bi', 'powerbi', 'power-bi', 'dax'],
    'Tableau': ['tableau software', 'tableau desktop'],
    'Excel': ['excel', 'tableur', 'tableaux croisés', 'tcd', 'graphiques excel'],
    'Qlik': ['qlik', 'qlikview', 'qlik sense'],
    'Looker': ['looker', 'looker studio', 'data studio'],
    'AWS': ['aws', 'amazon web', 's3', 'redshift', 'athena'],
    'Azure': ['azure', 'synapse', 'azure data'],
    'GCP': ['google cloud', 'bigquery', 'gcp'],
    'Databricks': ['databricks'],
    'Snowflake': ['snowflake'],
    'Spark': ['spark', 'pyspark', 'apache spark'],
    'Talend': ['talend'],
    'Informatica': ['informatica'],
    'SSIS': ['ssis', 'integration services'],
    'Airflow': ['airflow'],
    'SPSS': ['spss'],
    'Stata': ['stata'],
    'MongoDB': ['mongodb', 'mongo'],
    'Elasticsearch': ['elasticsearch', 'elastic'],
    'Git': ['git', 'github', 'gitlab', 'versionning'],
    'SAP': ['sap'],
    'Salesforce': ['salesforce'],
    'Google Analytics': ['google analytics', 'ga4', 'analytics'],
}

CATEGORIES_THEMATIQUES = {
    'Analyse de données': ['analys', 'traiter', 'interpréter', 'exploiter', 'étudier', 'données', 'data', 'statistique', 'quantitatif'],
    'Bases de données': ['base', 'bdd', 'sgbd', 'requête', 'sql', 'mongodb', 'oracle'],
    'Visualisation': ['visualis', 'dashboard', 'tableau de bord', 'graphique', 'rapport', 'reporting', 'dataviz', 'bi', 'business intelligence'],
    'Modélisation statistique': ['modèle', 'modélis', 'régression', 'prévision', 'prédictif', 'algorithme', 'machine learning', 'statistique'],
    'Qualité & conformité': ['qualité', 'conformité', 'contrôle', 'audit', 'norme', 'iso', 'haccp', 'qse', 'procédure'],
    'Gestion de projet': ['projet', 'planif', 'coordin', 'pilotage', 'suivi', 'organisation', 'gestion de projet'],
    'Finance & comptabilité': ['financ', 'budget', 'compta', 'coût', 'prix', 'investissement', 'rentabilité', 'trésorerie'],
    'Communication & présentation': ['communic', 'présent', 'rédaction', 'rédiger', 'synthès', 'rapport', 'présentation', 'oral'],
    'Sécurité IT': ['sécurité', 'cyber', 'protection', 'sauvegarde', 'confidentialité'],
    'ETL & Data Engineering': ['etl', 'intégration', 'pipeline', 'extraction', 'transformation', 'chargement', 'flux de données'],
}

# FONCTIONS
def detecter_outils_ameliore(competence, outils_dict):
    comp_lower = competence.lower()
    outils_detectes = set()
    for outil, patterns in outils_dict.items():
        for pattern in patterns:
            if pattern.lower() in comp_lower:
                outils_detectes.add(outil)
                break
    return list(outils_detectes)

def categoriser_thematique(competence, categories):
    comp_lower = competence.lower()
    categories_trouvees = []
    for categorie, keywords in categories.items():
        for keyword in keywords:
            if keyword in comp_lower:
                categories_trouvees.append(categorie)
                break
    return categories_trouvees if categories_trouvees else ['Autre']



# CHARGEMENT
with open('offres_analyste.json', 'r', encoding='utf-8') as f:
    offres = json.load(f)

print(f"\nNombre d'offres d'emploi : {len(offres)}")

# Extraire compétences
competences_brutes = []
for offre in offres:
    if 'competences' in offre and offre['competences']:
        for comp in offre['competences']:
            if 'libelle' in comp:
                competences_brutes.append(comp['libelle'])

print(f"Total de compétences (avec répétitions) : {len(competences_brutes)}")

competences_count = Counter(competences_brutes)
competences_uniques = list(competences_count.keys())

print(f"Compétences uniques : {len(competences_uniques)}")


# PHASE 1 : OUTILS/LANGAGES


competences_analysees = {}
all_outils_avec_freq = []

for comp_unique in competences_uniques:
    freq = competences_count[comp_unique]
    outils = detecter_outils_ameliore(comp_unique, OUTILS_TECH)
    themes = categoriser_thematique(comp_unique, CATEGORIES_THEMATIQUES)
    
    competences_analysees[comp_unique] = {
        'frequence': freq,
        'outils': outils,
        'themes': themes
    }
    
    for outil in outils:
        all_outils_avec_freq.extend([outil] * freq)

outils_count = Counter(all_outils_avec_freq)

print(f"\nTotal mentions d'outils : {len(all_outils_avec_freq)}")

print("\n TOP 20 OUTILS/LANGAGES DATA ANALYST")
print("-" * 80)
print("(% = pourcentage d'offres demandant cet outil)\n")
for i, (outil, count) in enumerate(outils_count.most_common(20), 1):
    pct = (count / len(offres)) * 100
    bar = '█' * min(int(pct * 2), 50)
    print(f"{i:2d}. {outil:20s} │{bar} {count:4d} offres ({pct:5.1f}%)")


# PHASE 2 : THÉMATIQUES


print("PHASE 2 : RÉPARTITION PAR THÉMATIQUE")


all_themes_avec_freq = []
for comp_unique, data in competences_analysees.items():
    for theme in data['themes']:
        all_themes_avec_freq.extend([theme] * data['frequence'])

themes_count = Counter(all_themes_avec_freq)

print("\n RÉPARTITION DES COMPÉTENCES PAR THÉMATIQUE")
print("-" * 80)
for theme, count in themes_count.most_common():
    pct = (count / len(competences_brutes)) * 100
    print(f"{theme:35s} : {count:4d} mentions ({pct:5.1f}%)")

# PHASE 3 : CLUSTERING

print("PHASE 3 : CLUSTERING DES COMPÉTENCES")

comp_avec_outils = {k: v for k, v in competences_analysees.items() if v['outils']}
comp_sans_outils = {k: v for k, v in competences_analysees.items() if not v['outils']}

print(f"\nCompétences uniques avec outils : {len(comp_avec_outils)}")
print(f"Compétences uniques sans outils : {len(comp_sans_outils)}")

if len(comp_sans_outils) > 10:
    textes = list(comp_sans_outils.keys())
    
    vectorizer = TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(textes)
    
    n_clusters = min(15, len(comp_sans_outils) // 10)
    print(f"Nombre de clusters : {n_clusters}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf)
    
    cluster_names = {}
    for cluster_id in range(n_clusters):
        comps_cluster = [textes[i] for i in range(len(textes)) if labels[i] == cluster_id]
        
        mots_cluster = []
        for comp in comps_cluster:
            mots = re.findall(r'\b\w{5,}\b', comp.lower())
            mots_cluster.extend(mots)
        
        stop_words = {'dans', 'pour', 'avec', 'être', 'avoir', 'faire', 'mettre', 'donner'}
        mots_filtres = [m for m in mots_cluster if m not in stop_words]
        
        top_mots = Counter(mots_filtres).most_common(2)
        cluster_names[cluster_id] = ' / '.join([m.capitalize() for m, _ in top_mots])
    
    for i, comp in enumerate(textes):
        comp_sans_outils[comp]['cluster'] = cluster_names[labels[i]]
else:
    for comp in comp_sans_outils:
        comp_sans_outils[comp]['cluster'] = 'Général'

for comp, data in comp_avec_outils.items():
    data['cluster'] = ', '.join(data['outils'])


# STATISTIQUES FINALES

print("STATISTIQUES FINALES")

cluster_freq = Counter()
for comp, data in competences_analysees.items():
    cluster = data.get('cluster', 'Non classé')
    cluster_freq[cluster] += data['frequence']

print("\n TOP 20 CLUSTERS PAR FRÉQUENCE D'APPARITION")
print("-" * 80)
print("(= nombre total d'offres mentionnant ce cluster)\n")
for i, (cluster, count) in enumerate(cluster_freq.most_common(20), 1):
    pct = (count / len(offres)) * 100
    print(f"{i:2d}. {cluster:45s} : {count:4d} ({pct:5.1f}%)")

# RAPPORT DÉTAILLÉ


print("\nClassement par nombre d'offres :\n")
for i, (outil, count) in enumerate(outils_count.most_common(), 1):
    pct = (count / len(offres)) * 100
    print(f"{i:2d}. {outil:25s} : {count:4d} offres ({pct:5.1f}%)")

print("\n" + "="*80)
print("SECTION 2 : TOP 20 COMPÉTENCES LES PLUS DEMANDÉES")
print("="*80)
print()
for i, (comp, count) in enumerate(competences_count.most_common(20), 1):
    pct = (count / len(offres)) * 100
    print(f"{i:2d}. {comp:65s} : {count:3d} ({pct:4.1f}%)")

print("\n" + "="*80)
print("SECTION 3 : DÉTAIL DES CLUSTERS (TOP 5)")
print("="*80)

for i, (cluster, count) in enumerate(cluster_freq.most_common(5), 1):
    pct = (count / len(offres)) * 100
    print(f"\n{'='*80}")
    print(f"🔹 CLUSTER #{i}: {cluster}")
    print(f"   Fréquence : {count} mentions ({pct:.1f}% des offres)")
    print(f"{'='*80}")
    
    # Trouver les compétences de ce cluster
    comps_du_cluster = [
        (comp, data['frequence']) 
        for comp, data in competences_analysees.items() 
        if data.get('cluster') == cluster
    ]
    comps_du_cluster.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 compétences de ce cluster :")
    for j, (comp, freq) in enumerate(comps_du_cluster[:5], 1):
        pct_comp = (freq / len(offres)) * 100
        print(f"  {j:2d}. {comp:60s} : {freq:3d} ({pct_comp:4.1f}%)")
    
    if len(comps_du_cluster) > 5:
        print(f"  ... et {len(comps_du_cluster) - 5} autres compétences")

print("\n" + "="*80)
print("RÉSUMÉ FINAL")
print("="*80)
print(f"\n Statistiques globales :")
print(f"  • Offres analysées : {len(offres)}")
print(f"  • Compétences totales (avec répétitions) : {len(competences_brutes)}")
print(f"  • Compétences uniques : {len(competences_uniques)}")
print(f"  • Outils/langages détectés : {len(outils_count)}")
print(f"  • Thématiques identifiées : {len(themes_count)}")
print(f"  • Clusters créés : {len(cluster_freq)}")
print("\n" + "="*80)



#Plage des dates de création des offres
date_strings = [offre.get("dateCreation") for offre in offres if offre.get("dateCreation")]
if date_strings:
    dates = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in date_strings]
    print(min(dates), max(dates))












#appellationlibellé
#experiencelibelle
#niveaulibelle


#Count apparition de chaque compétence
#moyenne date creation
#min et max date creation

#count appellation libellé
