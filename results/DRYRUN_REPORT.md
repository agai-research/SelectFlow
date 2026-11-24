# Répétition générale (agent simulé)

**Ces chiffres ne sont pas des résultats.** L'agent simulé suit exactement le modèle que SelectFlow
suppose ; la répétition vérifie que le code tourne et que les mesures sont calculées correctement.
Les résultats de l'article viendront des sondes avec Claude.

Fichier complet : `results/dryrun.json` (durée totale : environ 15 secondes).

## Scénario grec

Reproduit l'issue de l'article : `v1` sur `hT`, `v2` et `v4` sur `hP`, `v3` laissée au demandeur
(aucun outil ne dépose la demande de certificat), outil voisin `notify.change.submit` exclu,
durée sur le chemin critique de 17 jours.

## Ce que la répétition a appris sur le protocole

1. **La dilution est faiblement identifiable.** Le terme γ et la constante se compensent quand la charge
   de contexte varie peu : il a fallu environ 12 000 observations avec des toolkits de 2 à 20 outils pour
   retrouver les paramètres. Les sondes réelles doivent donc faire varier fortement la charge, et E1 doit
   rapporter l'intervalle de confiance de γ.
2. **La comparaison avec la composition tâche par tâche n'apparaît que lorsque les transferts
   d'artefacts contraignent le choix** (densités de partage élevées) : il faut garder les trois densités.
3. **Une correction à l'Algorithme 1 de l'article** : la critique de fermeture doit retenir le *premier*
   producteur d'un artefact (comme le chaînage avant), sinon un workflow correct peut être déclaré
   superflu. Ligne à modifier : « sup[ϑ] ← v_i » devient « sup[ϑ] ← v_i si ϑ n'a pas encore de fournisseur ».
