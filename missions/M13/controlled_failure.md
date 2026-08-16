# Controlled failure — incompatible scales

The notebook deliberately adds `interface_event_count`, a feature generated independently of the class label. Its values span thousands while `practice_hours` spans roughly ten and `assessment_score` spans tens.

The seeded root cause is **one incompatible high numeric scale allowing a weak feature to dominate distance**.

Do not begin by changing `k`, trying many models or deleting arbitrary rows.

Required diagnostic sequence:

1. record the expected behavior before adding the feature;
2. predict which coordinate will dominate raw Euclidean distance;
3. reproduce the metric and query-prediction change;
4. inspect the raw feature ranges;
5. retrieve the query's nearest neighbors before and after the feature is added;
6. decompose squared distance by feature for the closest polluted neighbor;
7. state one falsifiable root-cause hypothesis;
8. fit scaling on training rows only and apply the same transform through a pipeline;
9. rerun the original split, query and neighbor inspection;
10. compare the scaled polluted model with a scaled model that excludes the irrelevant feature;
11. explain whether scaling, feature removal, or both should be adopted;
12. state a prevention check for future distance-based models.

A repaired score alone is insufficient. Evidence must connect feature scale to distance contribution, neighbor membership, votes and the changed prediction.
