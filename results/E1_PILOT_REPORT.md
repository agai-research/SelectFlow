# Pilote E1 — prédiction de la sélection (23 novembre 2026)

**Données réelles.** 2 000 sondes, Claude Sonnet 5.5 (`claude-sonnet-5-5`, sans réflexion avant l'appel,
appels parallèles désactivés), API Batch. Outils : 278 définitions réelles de 25 serveurs de MCP-Bench.
Requêtes : les 104 descriptions « floues » des tâches de MCP-Bench, qui ne nomment pas les outils.
Toolkits tirés au hasard (2 à 20 outils), avec les outils du serveur de la tâche dans 80 % des cas.
Découpage 70/15/15 par tâche : les tâches de test n'ont jamais été vues à l'estimation. Coût : 7,58 $.

| Modèle | Perte log. | Brier | Premier choix |
|---|---|---|---|
| Logit multinomial (SelectFlow) | 1,231 | 0,630 | 0,474 |
| Logit sans dilution (γ = 0) | 1,234 | 0,630 | 0,468 |
| Pertinence seule | 1,743 | 0,769 | 0,363 |

- **Critère d'arrêt ou de poursuite : rempli.** Écart de perte logarithmique avec la pertinence : 0,51,
  intervalle à 95 % [0,37 ; 0,65], entièrement au-dessus de zéro.
- **La dilution n'est pas confirmée** : écart avec γ = 0 de 0,002, intervalle [−0,008 ; 0,013].
  Avec des toolkits de 20 outils au plus (2 500 jetons en moyenne), la charge de contexte ne change pas
  la sélection de façon mesurable. À tester dans E1 complet avec des toolkits de 40 à 80 outils ;
  si l'effet reste nul, il faudra le dire dans l'article.
- **Ce que le pilote ne montre pas encore** : le modèle gagne en partie grâce à ses caractéristiques
  (outil du serveur de la tâche, nom de l'outil). Il reste à montrer que la *concurrence* entre outils compte :
  E1 complet doit ajouter des manipulations contrôlées (ajouter un outil voisin à un toolkit fixe et mesurer
  la part perdue par l'outil visé).
- Taux d'absence d'appel : 37 % (souvent, aucun outil du toolkit ne convient). Réponses tronquées : 1 %.

Fichiers : `results/e1_sonnet.json`, `results/e1_fit_sonnet.json`, `results/e1_observations_sonnet.json`,
`results/e1_sonnet_messages.jsonl` (réponses brutes). Reproduction : `scripts/e1_batch.py`.
