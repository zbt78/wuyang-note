Yahoo_explicit all data
```json
Best mse: 0.9561745524406433 [9]
{
    "eval_metric": "mse"
}
{
    "env_num": 5,
    "factor_num": 40,
    "reg_only_embed": true,
    "reg_env_embed": false
}
{
    "batch_size": 131072,
    "epochs": 1000,
    "cluster_interval": 20,
    "evaluate_interval": 10,
    "lr": 0.001,
    "invariant_coe": 0.007375309563638757,
    "env_aware_coe": 7.207790368836971,
    "env_coe": 7.30272189219841,
    "L2_coe": 5.105587170019545,
    "L1_coe": 0.004098813161410509,
    "alpha": null,
    "use_class_re_weight": false,
    "use_recommend_re_weight": false,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": null,
    "stop_cluster_epoch": null
}
{
    "mse": 0.9492667118708292,
    "rmse": 0.9742913246154785,
    "mae": 0.7647597789764404
}
Best perform mean: 0.9492667118708292
Random seed list: [17373331, 17373511, 17373423]
```
Yahoo Explicit uniform_data
```json
Best mse: 0.9537165760993958 [9]
{
    "eval_metric": "mse"
}
{
    "env_num": 5,
    "factor_num": 40,
    "reg_only_embed": true,
    "reg_env_embed": false
}
{
    "batch_size": 131072,
    "epochs": 1000,
    "cluster_interval": 20,
    "evaluate_interval": 10,
    "lr": 0.001,
    "invariant_coe": 0.007375309563638757,
    "env_aware_coe": 7.207790368836971,
    "env_coe": 7.30272189219841,
    "L2_coe": 5.105587170019545,
    "L1_coe": 0.004098813161410509,
    "alpha": null,
    "use_class_re_weight": false,
    "use_recommend_re_weight": false,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": null,
    "stop_cluster_epoch": null
}
{
    "mse": 0.9470450480779012,
    "rmse": 0.9731501340866089,
    "mae": 0.7638651927312216
}
Best perform mean: 0.9470450480779012
Random seed list: [17373331, 17373511, 17373423] 
```

Yahoo_implicit all_data
```json
Best ndcg@5: 0.6313382212660952 [1]
{
    "ndcg@3": 0.5572290075702447,
    "ndcg@5": 0.6287693886400786,
    "ndcg@7": 0.6759504947594545,
    "recall@3": 0.589378078817734,
    "recall@5": 0.7687014556780567,
    "recall@7": 0.8852369875153127,
    "precision@3": 0.34961685823754785,
    "precision@5": 0.28492063492063496,
    "precision@7": 0.2411447337555712
}
Best perform mean: 0.6287693886400786
Best perform var: 3.3521542516155046e-06
Best perform std: 0.0018308889238879307
Random seed list: [17373331, 17373511, 17373423]
```

Coat_InvPref_explicit all dataset:
```json
Best mse: 0.9985490441322327 [65]
{
    "eval_metric": "mse"
}
{
    "env_num": 4,
    "factor_num": 30,
    "reg_only_embed": true,
    "reg_env_embed": false
}
{
    "batch_size": 1024,
    "epochs": 1000,
    "cluster_interval": 30,
    "evaluate_interval": 10,
    "lr": 0.01,
    "invariant_coe": 2.050646960185343,
    "env_aware_coe": 8.632289952059462,
    "env_coe": 5.100067503854663,
    "L2_coe": 7.731619515414727,
    "L1_coe": 0.0015415961377493945,
    "alpha": 1.7379692382330174,
    "use_class_re_weight": true,
    "use_recommend_re_weight": true,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": null,
    "stop_cluster_epoch": null
}
{
    "mse": 1.000884433587392,
    "rmse": 1.0004412929217021,
    "mae": 0.7811758716901144
}
Best perform mean: 1.000884433587392
Random seed list: [17373331, 17373511, 17373423]
```