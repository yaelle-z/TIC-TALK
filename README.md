# Analyse multimodale du stand-up : Poses, Thèmes, Rires

Ce dépôt contient uniquement le code final du projet d’analyse multimodale du stand-up.
Les données volumineuses (vidéos, fichiers audio, `.srt` complets), les modèles entraînés et les résultats intermédiaires lourds sont hébergés séparément sur un drive.

Les notebooks sont organisés en fonction de l'étape de l'analyse
1. **La gestuelle (poses)** des humoristes sur scène.
2. **Les thématiques (topics)** abordées dans leurs spectacles.
3. **Les rires** déclenchés dans le public.
---

### 1.1 Analyse de la gestuelle (Poses)

#### `echantillonage_filemp4_toimages.ipynb`
- **Entrée** : vidéos brutes (`.mp4`) du corpus (non incluses).
- **Sortie** : série d’images extraites à intervalles réguliers (par exemple 1 image/sec).
- **Rôle** : créer un échantillon d’images représentatif pour chaque spectacle, servant de base à l’annotation et à l’analyse ultérieure.

#### `selection_image_aleatoirement_pour_annoter.ipynb`
- **Entrée** : dossier contenant les images extraites des vidéos.
- **Sortie** : sous-échantillon aléatoire d’images (CSV listant les fichiers choisis et copies dans un dossier dédié).
- **Rôle** : réduire la quantité d’images à annoter manuellement tout en conservant la diversité des poses et des plans.

#### `entrainement_modele_classification_de_plan.ipynb`
- **Entrée** : images annotées avec leur type de plan + CSV d’annotations.
- **Sortie** : modèle YOLO entraîné (poids `.pt`) pour classer automatiquement les images par type de plan.
- **Rôle** : automatiser la détection du type de plan (ex. plein-pied, plan taille) dans les images extraites.

#### `application_modele_classification_de_plans(2).ipynb`
- **Entrée** : poids du modèle YOLO entraîné + images extraites.
- **Sortie** : images triées dans des dossiers distincts selon le type de plan détecté.
- **Rôle** : filtrer automatiquement les images pour isoler celles utiles à l’analyse des poses (plein-pied).

#### `image_to_csv_keypoints.py`
- **Entrée** : images plein-pied issues de l’étape précédente.
- **Sortie** : un fichier CSV par image contenant :
  - Les coordonnées x, y de 17 points clés du corps (nez, yeux, épaules, coudes, poignets, hanches, genoux, chevilles).
  - Les coordonnées de la boîte englobante (bounding box).
- **Rôle** : convertir les images en données numériques structurées exploitables pour l’analyse statistique.

#### `capture_to_pose_valeurs_dans_csv.ipynb`
- **Entrée / Sortie** : identique au script précédent.
- **Rôle** : version notebook pour exécuter l’extraction des keypoints de manière interactive et vérifier visuellement les résultats.

#### `csv_poses_to_cluster.ipynb`
- **Entrée** : ensemble des CSV de keypoints.
- **Sortie** :
  - Groupes de poses similaires (clusters) calculés par clustering.
  - Moyenne des coordonnées pour chaque cluster.
  - Visualisations des poses moyennes.
- **Rôle** : identifier les postures récurrentes et caractériser la gestuelle typique.

#### `classifications_de_poses_images_trieesenfcduplan_to_modele_to_csv_dimages_trieesenfctdelapose.ipynb`
- **Entrée** : images triées par type de plan + modèle de classification de poses.
- **Sortie** : CSV listant chaque image et la classe de pose attribuée par le modèle.
- **Rôle** : affiner la typologie gestuelle en classifiant chaque posture détectée dans le corpus.

---

### 1.2 Analyse des thèmes

#### `nettoyage_fichiers_srt.ipynb`
- **Entrée** : fichiers `.srt` bruts des spectacles.
- **Sortie** : `.srt` nettoyés (texte uniquement, mise en forme standardisée, suppression des balises).
- **Rôle** : préparer les transcriptions textuelles pour l’analyse en traitement automatique du langage.

#### `notebook_topic_model_REUSSI.ipynb`
- **Entrée** : fichiers `.srt` nettoyés.
- **Sortie** :
  - Attribution d’un topic à chaque segment temporel.
  - Visualisations de la dynamique thématique.
  - Export des résultats au format CSV.
- **Rôle** : identifier les thèmes dominants dans chaque spectacle à l’aide de BERTopic et analyser leur évolution dans le temps.

---

### 1.3 Analyse des rires

#### `detection_de_rire_whisper_at.py`
- **Entrée** : fichiers audio ou vidéo (`.mp3`, `.wav`, `.mp4`, `.m4a`).
- **Sortie** : CSV listant pour chaque rire :
  - Horodatage de début et de fin.
  - Type de rire (selon la classification Whisper-AT).
  - Score de confiance.
- **Rôle** : détecter automatiquement les rires dans l’audio des spectacles avec Whisper-AT.

#### `calcul_statistique_rire.ipynb`
- **Entrée** : CSV de rires détectés.
- **Sortie** : tableau de métriques par spectacle (nombre total de rires, durée cumulée, densité).
- **Rôle** : quantifier et comparer la fréquence des rires entre spectacles.

#### `représentation_rires_dans_le_temps_dapres_srt_sourds.ipynb`
- **Entrée** : fichiers `.srt` pour sourds/malentendants contenant des mentions explicites de rires.
- **Sortie** : courbes représentant la distribution des rires dans le temps.
- **Rôle** : comparer la détection audio des rires aux mentions présentes dans les sous-titres.

#### `rires_poses_to_statistiques.ipynb`
- **Entrée** : données issues de la détection des rires + données de classification de poses.
- **Sortie** : graphiques et tableaux croisant la densité de rires et les types de poses.
- **Rôle** : analyser les corrélations entre la gestuelle et le déclenchement du rire.

---

### 1.4 Visualisation et croisement

#### `code_visualisation_clustering.ipynb`
- **Entrée** :
  - Rires : dossier de CSV par spectacle (`*_laughs.csv`).
  - Poses : `matrice_poses_par_spectacle_2.csv`.
  - Thèmes : `matrice_topics_par_spectacle.csv`.
- **Sortie** :
  - Graphiques de distribution pour chaque variable (barplots par spectacle).
- **Rôle** : visualiser séparément la répartition des rires, poses et thèmes sur l’ensemble du corpus.

#### `codecroismentfinalvaribales.ipynb`
- **Entrée** :
  - CSV de rires par spectacle.
  - Fichier CSV de classification de poses pour l’ensemble du corpus.
- **Sortie** :
  - Tableau temporel fusionnant, pour chaque spectacle :
    - Présence/absence de rires par seconde.
    - Geste dominant par seconde.
  - Graphiques combinant ces deux dimensions.
- **Rôle** : observer conjointement la dynamique des rires et des gestes dans le temps.

---

## 2. Données et modèles

Les données brutes, modèles et résultats lourds **ne sont pas inclus dans ce dépôt**.  
Ils sont disponibles sur kDriveà l’adresse suivante :  
 **[[Lien vers le drive](https://drive.google.com/drive/folders/1r7usWBJKX7F751YzaAdnSWIAkO6Xr0S-?usp=sharing)]**

Ces ressources incluent :.
- Fichiers `.srt` bruts et nettoyés.
- Modèles entraînés (YOLO, BERTopic…).
- Résultats intermédiaires (CSV, figures).
