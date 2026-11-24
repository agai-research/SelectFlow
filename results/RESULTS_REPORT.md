# SelectFlow — résultats expérimentaux obtenus à ce stade (23 novembre 2026)

Toutes les sondes ont été envoyées à Claude Sonnet 5.5 (`claude-sonnet-5-5`), sans réflexion avant l'appel,
une seule sélection par occasion, via l'API Batch. Coût total des appels : environ 60 $ (E1 : 24 $, E3 : 6,6 $, E6 : 15,6 $, E2 : environ 10 $, interprétation et formulations : environ 3 $).

## Données réelles construites

| Fédération | Procédures | Organismes | Outils | Variantes voisines | Requêtes avec workflow de référence |
|---|---|---|---|---|---|
| Grèce (Mitos) | 93 (81 fiches + 12 procédures émettrices référencées) | 29 | 252 | 22 | 67, toutes inter-organismes |
| Arabie saoudite | 24 (21 de pages officielles) | 7 | 56 | 6 | 15, dont 14 inter-organismes |

Les dépendances et les workflows de référence viennent du registre (liens « Relevant Procedure » de Mitos,
pièces requises des pages saoudiennes), par chaînage avant : aucun n'est produit par un LLM.

## E1 complet — prédiction de la sélection (1 866 sondes × 2, 120 capacités)

| Condition | Texte explicite : outil visé | Texte « citoyen » : outil visé | Texte « citoyen » : aucun appel |
|---|---|---|---|
| Toolkit de base (3 outils) | 98 % | 77 % | 23 % |
| + variante voisine (22 capacités) | 98 % (variante jamais choisie) | 70 % [−26 ; +14 pts] | 26 % (variante choisie 4,5 %) |
| + description assertive | 99 % | 83 % [−1 ; +11 pts] | 18 % |
| + 10 outils sans rapport | 99 % | 90 % [+6 ; +19 pts] | 10 % |
| + 30 outils | 100 % | 91 % [+7 ; +20 pts] | 8 % |
| + 60 outils (≈ 5 600 jetons) | 100 % | 90 % [+6 ; +19 pts] | 8 % |

Entre crochets : intervalle à 95 % de l'écart avec la base (bootstrap apparié sur les capacités).

Constats :
1. **La précision du texte de l'étape domine tout.** Avec l'intitulé officiel, le modèle choisit l'outil visé dans
   98 % des cas ; avec une formulation de citoyen, 77 %. Le passage d'une requête vague à des étapes précises, que fait
   la dérivation de SelectFlow, a donc un effet mesuré important.
2. **Les erreurs sont surtout des abstentions, pas des détournements.** La variante voisine n'est presque jamais choisie
   (4,5 % avec texte vague, 0 % avec texte explicite).
3. **La dilution n'est pas confirmée ; l'effet observé est inverse.** Ajouter 10 à 60 outils sans rapport *réduit* les
   abstentions de 13 à 15 points. Le terme γ ≥ 0 du modèle ne peut pas représenter cet effet (estimation à la borne 0).
4. **Prédiction.** Logit multinomial contre pertinence seule : écart de perte logarithmique 0,026 [0,000 ; 0,053] avec
   texte vague, 0,047 [0,025 ; 0,071] avec texte explicite. Le logit est bien calibré pour la variante voisine
   (prédit 0,56, observé 0,54 ; pertinence 0,67), mais sous-estime l'outil visé quand la charge augmente.

## E4 sur données réelles — flux refusés (sans appel au modèle)

Plans composés avec les étiquettes certifiées : 0 tâche sur 222 ouvre un flux refusé (comportements réels des outils).
Avec les annotations déclarées : 7 tâches sur 183 en Grèce (3,8 %), 0 sur 39 en Arabie saoudite. Cause : les
annotations déclarées ignorent les données que les outils renvoient.

### Mise à jour avec le modèle à effet de charge signé

Avec le modèle réestimé (γ < 0), le composeur ajoute davantage d'outils, et la différence se creuse : avec les
annotations déclarées, 47 tâches grecques sur 183 (26 %) et 13 tâches saoudiennes sur 39 (33 %) ouvrent un flux
refusé ; avec les étiquettes certifiées, toujours aucune. Après correction des catégories de données par le
vérificateur (voir plus bas) : 40 sur 183 et 13 sur 39, toujours 0 avec la certification.

## E5 sur données réelles — composition conjointe et réparation (sans appel au modèle)

| Densité des bases de partage | Tâches équipées | Laissées au demandeur |
|---|---|---|
| 1,0 (registre complet) | 100 % | 0 % |
| 0,75 | 97,5 % | 2,5 % |
| 0,5 | 88,2 % | 11,8 % |

Composition conjointe, tâche par tâche et sans réparation donnent les mêmes résultats sur ces registres : chaque
procédure a ses propres outils (les transferts d'artefacts ne contraignent pas le choix) et presque aucun document n'a
d'émetteur alternatif (Grèce : aucun ; Arabie saoudite : deux, la licence commerciale standard ou accélérée).
L'apport de la composition conjointe et de la réparation n'apparaît que sur les fédérations synthétiques.

## HYSET (décision : option 1, comparaison sur MCP-Bench uniquement)

Entraîné deux fois sur Colab avec l'encodeur du retriever de ToolLLM (26 623 puis 30 000 requêtes TOUCAN) :
rappel@5 en validation de 0,39 et 0,36, arrêt anticipé après 5 époques dans les deux cas. Première exécution :
0,09 sur 55 serveurs jamais vus. Sur les 104 tâches de MCP-Bench, classement limité à nos 278 outils : un outil du
bon serveur figure parmi ses 5 premiers dans 39 % des tâches, contre 38 % pour BM25.

HYSET n'est pas réentraîné sur les outils administratifs : aucun d'eux n'a d'historique d'appels dans TOUCAN, et
HYSET apprend ses représentations à partir de cet historique. Son rappel sur les serveurs inconnus indique qu'il
retomberait au niveau d'une recherche textuelle, déjà représentée dans E3 par BM25. La comparaison sur les données
administratives est reportée aux travaux futurs. Le texte à reprendre dans l'article est dans `paper_text_hyset.md`.

## Modèle de sélection avec effet de charge de signe libre

Réestimé sur les sondes d'E1 à texte « citoyen », γ = −0,09 par millier de jetons (plus d'outils, moins
d'abstentions). Gain de perte logarithmique sur γ ≥ 0 : 0,006 [−0,007 ; 0,019] ; sur la pertinence seule :
0,050 [0,006 ; 0,088]. Le taux d'abstention est mieux calibré (base : prédit 0,22, observé 0,23), mais la forme
linéaire ne capte pas la saturation au-delà de 10 outils. Le programme de composition reste exact avec γ < 0 (test).

## E3 sur données réelles — composition des toolkits (4 380 sondes, étapes formulées comme un citoyen)

| Méthode | Outil visé | Mauvais outil | Aucun appel |
|---|---|---|---|
| SelectFlow | 65,1 % | 23,3 % | 11,6 % |
| BM25 top-5 | 62,6 % | 27,7 % | 9,7 % |
| Classement par attraction | 62,6 % | 24,9 % | 12,6 % |
| Sélection par QoS | 61,6 % | 23,7 % | 14,6 % |
| Tool Search natif | 66,4 % | 30,6 % | 3,0 % |

SelectFlow contre BM25 : +2,5 points [−0,2 ; +5,4] ; contre QoS : +3,4 [+0,8 ; +6,1], p de Holm 0,05 ;
contre Tool Search : −1,4 [−4,0 ; +1,4]. HYSET n'est pas inclus dans E3 sur les données administratives (voir la section HYSET).

## E6 sur données réelles — de bout en bout (82 requêtes de citoyens, 3 systèmes)

Environnement simulé avec la sémantique des procédures : déposer la demande avec les pièces requises, payer s'il y a
des frais, puis obtenir le document. SelectFlow reçoit la confirmation des documents visés par le demandeur ; les
agents de référence reçoivent la même confirmation comme réponse du citoyen s'ils s'arrêtent sans le document.

| Système | Document obtenu | Grèce | Arabie saoudite | Appels d'outils | Ont eu besoin de la confirmation |
|---|---|---|---|---|---|
| SelectFlow | 100 % | 100 % | 100 % | 6,2 | — |
| Agent avec tous les outils | 92,7 % | 100 % | 60 % | 7,6 | 95 % |
| Agent avec Tool Search | 81,7 % | 86,6 % | 60 % | 7,5 | 94 % |

SelectFlow contre tous les outils : +7,3 points [+2,4 ; +13,4], McNemar p = 0,03 ; contre Tool Search : +18,3 points
[+11,0 ; +26,8], p < 0,001. Les 82 workflows dérivés coïncident avec les références du registre.

Interprétation des requêtes : 80 % des documents visés trouvés exactement en Arabie saoudite. En Grèce, les noms
internes des documents étaient opaques (« out_gr17 ») et l'interprétation a échoué ; avec des noms parlants, elle
atteint 86,6 % (mesure séparée sur les 67 requêtes). Cela ne change pas le résultat d'E6, où les documents visés
sont confirmés, mais les noms des artefacts grecs doivent être corrigés dans la version publiée des données.

Limites d'E6 : serveurs simulés ; descriptions d'outils dérivées des intitulés ; un seul modèle ; une seule exécution
par requête.

## Formulations des citoyens et vérification par un second modèle

Claude Sonnet 5.5 a rédigé, pour les 82 requêtes, une formulation vague en anglais, une en français et une en arabe ;
Claude Opus 5.5 a jugé les 246 formulations fidèles à l'objectif (100 %). Le même vérificateur a contrôlé
l'annotation déterministe des 117 procédures : finalité correcte dans 87 % des cas, catégories de données correctes
dans 73 % (erreurs surtout « santé » ajoutée à tort et « lieu » manquant). Les résultats d'E4 et d'E5 changent peu
avec les catégories corrigées.

Limite : le vérificateur est du même éditeur que le rédacteur ; le protocole prévoyait un modèle d'une autre famille
(Qwen), impossible à exécuter sans GPU ni accès à Hugging Face. Un taux de 100 % suggère un vérificateur indulgent.

## E2 sur données réelles — dérivation des workflows (82 requêtes, données version 2)

Requêtes de citoyens en anglais ; documents visés confirmés par le demandeur.

| Méthode | Workflow fermé | Égal à la référence | Temps |
|---|---|---|---|
| Génération contrainte, premier candidat (catalogue réduit aux 30 services les plus proches) | 18% | 18% | — |
| SelectFlow (critiques, révision, chaînage avant) | 100% | 100% | 16.5 s |
| Génération directe, catalogue complet dans le prompt | 100% | 100% | 2.1 s |

- Les critiques rejettent 82% des premiers candidats ; dans tous ces cas, la révision
  aboutit à un workflow accepté. Les rejets viennent de la réduction du catalogue : il manque au candidat un service
  émetteur d'une pièce, que la révision ajoute.
- **La génération directe réussit aussi dans 100 % des cas** quand le catalogue entier (93 ou 24 services) tient dans
  le prompt et que les documents visés sont confirmés. Sur ces registres, la vérification sert de garantie, pas de
  correction ; elle devient nécessaire dès qu'il faut réduire le catalogue, ce qui arrivera à l'échelle d'un registre
  national (Mitos compte des milliers de procédures). Elle coûte environ 8 fois plus de temps (16,5 s contre 2,1 s).
- **Le point faible est l'interprétation de la requête** : documents visés trouvés exactement dans 80%
  des cas (au moins un dans 85%). C'est ce qui justifie l'étape de confirmation par le demandeur.

## Coûts par méthode

| Expérience | Mesure | Valeurs |
|---|---|---|
| E3 | Jetons d'entrée par sonde ; coût pour 1 000 sondes (batch) | SelectFlow 767 jetons, 1.16 $ ; BM25 1091, 1.53 $ ; Tool Search 2856, 3.85 $ |
| E6 | Appels d'outils par requête | SelectFlow 6.2 ; tous les outils 7.6 ; Tool Search 7.5 |
| E6 | Temps de planification de SelectFlow | 17.0 s par requête |

SelectFlow atteint la précision du Tool Search natif en E3 pour environ 30 % de son coût par sonde.

## Données, version 2

Les documents grecs portent maintenant des noms parlants (`data/v2/`). La version 1, utilisée par E1, E3 et E6,
est reproduite à l'identique par le code (`mitos_descriptors(raw, meaningful_names=False)`), ce qui a été vérifié.
Les workflows de référence sont identiques dans les deux versions. E2 utilise la version 2.

## Figures et tableaux de la section 5

`results/figures/` : e1_manipulations, e3_toolkits, e4_flows, e6_end_to_end (PDF et PNG) ;
`results/tables/` : e2_derivation.tex, e3_toolkits.tex, e5_composition.tex. Ils sont régénérés par
`python scripts/make_figures.py`.

## Contrôle de la circularité (version 3 des données)

**Problème.** Dans les versions 1 et 2, les descriptions d'outils reprenaient les intitulés des procédures, comme
les étapes que SelectFlow transmet à l'agent. Une partie des résultats pouvait venir de cette correspondance de mots.

**Correction.** Les 308 descriptions ont été réécrites « comme le service informatique de l'organisme », sans les
intitulés : la part des mots des intitulés reprise tombe de 100 % à 19 %. Les capacités de chaque outil ont ensuite été
retrouvées à partir de ces seules descriptions (5 passes, Claude Haiku 5.5) : précision 0.31,
rappel 0.89. SelectFlow ne dispose alors d'aucune information que les agents de référence n'ont pas.

**E6 sur 45 requêtes** (les 15 saoudiennes, 30 grecques tirées au hasard) :

| Système | Descriptions tirées des intitulés | Descriptions indépendantes | Descriptions et capacités indépendantes |
|---|---|---|---|
| SelectFlow | 100 % | 100 % | **86.7%** |
| Agent avec tous les outils | 86.7% | 64.4% | (inchangé) |
| Agent avec Tool Search | 75.6% | 22.2% | (inchangé) |

Dans le cas le plus strict, SelectFlow devance l'agent avec tous les outils de 22 points
[7 ; 38], McNemar p = 0.021,
et l'agent avec Tool Search de 64 points
[47 ; 80], p < 0,001.
Les agents de référence dépendaient fortement de la correspondance de mots ; SelectFlow beaucoup moins. Ses échecs
restants viennent des erreurs de correspondance des capacités (précision 0,31).

**E1 avec descriptions indépendantes** (étapes précises, 120 capacités, 786 sondes) : outil visé dans
98.6% des cas (98,1 % avant) ; avec une variante voisine, 92.4%, variante choisie dans
6.1% des cas (0 % avant) ; avec 30 outils de plus, 98.9%.
L'effet de la formulation précise ne vient donc pas de la correspondance de mots. Avec des descriptions indépendantes,
les variantes voisines commencent à détourner des appels, ce qui redonne un rôle à la composition des toolkits.

Coût de ce contrôle : environ 21 $, dont environ 5 $ perdus sur une exécution invalide (correspondance vide, écartée).

## Conséquences pour l'article

- La thèse doit être recentrée : avec un modèle récent, le risque principal n'est pas le détournement vers un outil
  voisin mais l'abstention quand l'étape est mal formulée ; et la charge de contexte, jusqu'à 60 outils, ne nuit pas.
- Le modèle de sélection doit laisser le signe de l'effet de charge libre (γ réel au lieu de γ ≥ 0), ce qui reste
  compatible avec les propositions tant que le dénominateur reste positif.
- E3 et E6 situent l'apport : la composition des toolkits seule apporte peu (E3), mais la chaîne complète, avec la
  dérivation vérifiée et une tâche par épisode, obtient le document dans tous les cas, contre 93 % et 82 % pour les
  agents de référence (E6). L'apport principal est la dérivation vérifiée et le découpage en tâches.
