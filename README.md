# Rapport d'Analyse : Conception et Modélisation du Simulateur de Trafic Routier

## Table des matières

1. [Introduction](https://github.com/Wartets/Autoroutes/tree/main#1-introduction)  
2. [Architecture du Simulateur](#2-architecture-du-simulateur)  
3. [Fondements Conceptuels de la Modélisation](#3-fondements-conceptuels-de-la-modélisation)  
   - [Représentation de l'Infrastructure Routière](#31-représentation-de-linfrastructure-routière)  
   - [Caractérisation des Véhicules](#32-caractérisation-des-véhicules)  
   - [Modélisation des Comportements de Conduite](#33-modélisation-des-comportements-de-conduite)  
   - [Prise de Décision et Manœuvres Latérales](#34-prise-de-décision-et-manœuvres-latérales)  
   - [Facteurs Environnementaux et Adaptation](#35-facteurs-environnementaux-et-adaptation)  
4. [Fonctionnalités Clés et Mécanismes de Simulation](#4-fonctionnalités-clés-et-mécanismes-de-simulation)  
   - [Gestion des Collisions et Accidents](#41-gestion-des-collisions-et-accidents)  
   - [Comportement d'Accélération/Décélération](#42-comportement-daccélérationdécélération)  
   - [Logique de Changement de Voie](#43-logique-de-changement-de-voie)  
   - [Adaptation des Comportements](#44-adaptation-des-comportements)  
   - [Effets Météorologiques](#45-effets-météorologiques)  
5. [Déroulement de la Simulation (`run_simulation`)](#5-déroulement-de-la-simulation-run_simulation)  
6. [Mesures et Analyse des Résultats](#6-mesures-et-analyse-des-résultats)  

## 1. Introduction

Ce rapport présente une analyse détaillée de la conception et des principes de modélisation sous-jacents au simulateur de trafic routier. L'objectif principal de ce script Python est de reproduire la dynamique du trafic sur une infrastructure routière multi-voies, en intégrant des comportements automobiles réalistes, des interactions complexes entre véhicules, et des facteurs environnementaux variés.  

L'analyse des données générées vise à éclairer l'impact de ces choix sur des métriques fondamentales du trafic telles que la vitesse, la densité, le débit et les schémas comportementaux des conducteurs.

---

## 2. Architecture du Simulateur

Le simulateur est organisé en modules logiques, chacun jouant un rôle spécifique dans la modélisation du système de trafic :

- **Constantes Globales** : Paramètres configurables définissant la route, les véhicules et les conditions initiales.
- **Classe Voiture** : Représente un agent autonome avec attributs et règles de comportement.
- **Fonctions Utilitaires** : Calculs géométriques, vérification des conditions de voie, etc.
- **Fonction Principale `run_simulation`** : Orchestre l’initialisation, les étapes temporelles, la mise à jour des véhicules et la collecte de données.

---

## 3. Fondements Conceptuels de la Modélisation

Les choix de conception reposent sur l’équilibre entre fidélité et complexité computationnelle.

### 3.1. Représentation de l’Infrastructure Routière

- Route modélisée comme une **ligne droite périodique** (route circulaire).
- **Avantages :**
  - Maintien de la densité et de la vitesse (pas d’entrées/sorties artificielles).
  - Simplification des calculs via arithmétique modulaire.

**Paramètres clés :**
- `ROAD_LENGTH` : Longueur de la route.
- `NUM_LANES` : Nombre de voies parallèles.

### 3.2. Caractérisation des Véhicules

Chaque **Voiture** est un agent avec propriétés physiques et préférences comportementales.

- `CAR_LENGTH` : Longueur standard.
- `SPEED_LIMIT_MIN_KMH` / `SPEED_LIMIT_MAX_KMH` : Plage de vitesses légales.
- **Profils de conducteurs (`DRIVER_PROFILES`)** :
  - *Prudents* : distances de sécurité longues, moins agressifs.
  - *Normaux* : comportement standard.
  - *Agressifs* : distances réduites, dépassements fréquents.  

Multiplicateurs appliqués à `reaction_time`, `acceleration_factor`, `following_distance_factor`, `lane_change_aggressiveness`, `return_right_propensity`.

### 3.3. Modélisation des Comportements de Conduite

- **Accélération/Décélération** : `ACCELERATION_RATE_MS2`, `DECELERATION_RATE_MS2`.
- **Distance de suivi** : `FOLLOWING_DISTANCE_IDEAL_S`, `MIN_SAFE_GAP`.
- **Tolérance à la vitesse** : `speed_limit_tolerance_ms`.

### 3.4. Prise de Décision et Manœuvres Latérales

- **Sécurité avant tout** : `is_lane_clear`, `check_overlap_robust`.  
- **Motivations** :
  - Dépassement (`OVERTAKE_SPEED_BOOST_KMH`).
  - Retour à droite (`return_right_propensity`).
  - Recherche de voie plus rapide.
  - Céder le passage.
- **Paramètres de manœuvre** : `BASE_MANEUVER_DISTANCE`, `SPEED_DEPENDENT_MANEUVER_FACTOR_S`.

### 3.5. Facteurs Environnementaux et Adaptation

- **Météo** (`CURRENT_WEATHER`, `WEATHER_EFFECTS`) : affecte réaction, distances, tolérance.
- **Accidents** (`ACCIDENT_LOCATIONS`) : introduisent des perturbations.
- **Adaptation comportementale** : impatience/frustration → augmentation agressivité, réduction distances de suivi.

---

## 4. Fonctionnalités Clés et Mécanismes de Simulation

### 4.1. Gestion des Collisions et Accidents
- Vérifications avec `get_segments`, `do_segments_overlap`, `check_overlap_robust`.
- Placement dynamique des accidents et gestion des conflits d’espace.
- Évitement par les conducteurs (freinage ou changement de voie).

### 4.2. Comportement d’Accélération/Décélération
- **Vitesse cible** : selon profil et limite.
- **Modèle de suivi** : ajustement en fonction de la distance.
- **Freinage d’urgence** : seuil critique (`MIN_FOLLOWING_DISTANCE_CRITICAL`).
- **Reprise progressive** : après arrêt ou blocage.

### 4.3. Logique de Changement de Voie
- Post-dépassement : retour à droite.
- Optimisation du flux : voie plus rapide.
- Évitement de blocage : déplacement à gauche.
- Céder le passage : dépend des probabilités (`YIELD_LIKELIHOOD_*`).
- Critères de sécurité stricts avant manœuvre.

### 4.4. Adaptation des Comportements
- **Déclenchement** : blocage prolongé.
- **Effet** : impatience → plus agressif, distances réduites.
- **Retour à la normale** : après fluidification.

### 4.5. Effets Météorologiques
- Multiplicateurs (`WEATHER_EFFECTS`) appliqués aux paramètres de conduite.
- Exemples : pluie, grêle → temps de réaction +, tolérance –, distances +.

---

## 5. Déroulement de la Simulation (`run_simulation`)

### Initialisation
- Création de véhicules (profils/vitesses aléatoires).
- Ajout d’accidents prévus.

### Boucle Temporelle
- **Phase de Décision** : intention de mouvement (vitesse, voie, statut).
- **Phase d’Application** : mouvements, changements de voie, calcul des métriques.
- **Introduction d’Accidents** : selon planification.
- **Collecte de Données** : vitesses, densités, événements.
- **Affichage Console (optionnel)** : résumé en temps réel.

### Analyse Post-Simulation
- Traitement des données pour générer graphiques (diagramme vitesse-densité, heatmaps, etc.).

---

## 6. Mesures et Analyse des Résultats

Le simulateur collecte de nombreuses données pour analyser le trafic :

- **Vitesse et Densité par voie** : historique, diagrammes fondamentaux.
- **Débit des véhicules** : comptage par voie et global.
- **Changements de voie** : cumulés et catégorisés.
- **Temps par statut** : rouler, bloqué, freinage d’urgence, dépassement.
- **Temps de parcours** : distribution des temps de trajet.
- **Délai moyen par profil** : comparaison entre prudents, normaux, agressifs.
- **Interactions critiques** : freinages d’urgence, quasi-collisions (`TTC_WARNING_THRESHOLD`).
- **Utilisation des voies** : répartition spatiale.
- **Heatmaps** : vitesses, densités, carburant, CO₂.
- **Consommation et émissions** : calculées par véhicule, agrégées par profil.

---
