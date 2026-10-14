# Etapa 2 — validação externa geográfica e temporal

Decisão do autor: 06/10/2026, `PROMPT_STAGE2_NESTED_EXTERNAL.md`, continuando a etapa 1 `fb1d7f1`. Modelos finais, teste externo único e curva secundária concluídos. Nested CV completa não foi executada.

VIA: Fabricio Breve, 2020, Rio Claro/SP, 342 imagens. Extensão: Jefferson A. R. Passerini, coleta posterior, Fernandópolis e Jales/SP, cerca de 400 km de Rio Claro, 1.576 novas. Período exato/aparelho: TODO(autor). Definição de nonclear idêntica; auditoria de 100 rótulos aprovada. Hashes e revisão de cenas não identificaram imagens/cenas compartilhadas. O histórico público `4a5a1a5`, contendo os JPG brutos, foi preservado por decisão do autor.

**Aplicação da revisão**

| Verificação | Esperado | Obtido |
| --- | --- | --- |
| Exclusões por cena | 0 | 0 |
| Principal | 1576 | 1576 |
| clear | 844 | 844 |
| nonclear | 732 | 732 |
| Grupos | 1378 | 1378 |
| Auditoria concluída | True | True |
| Discordância | 0/100 | 0/100 |
| Aprovação | Jefferson Antonio Ribeiro Passerini (author) | Jefferson Antonio Ribeiro Passerini (author) |
| Data | 2026-10-06 | 2026-10-06 |

Wilson 95% da discordância: [0.00%; 3.70%]. CSV de nove rejeições e grafos automáticos preservados.

**Planejamento informado antes do treino**

| Etapa | Jobs | Tempo (h) | Disco (GiB) |
| --- | --- | --- | --- |
| final_models | 19 | 0.84–2.32 | 3.00–12.00 |
| external | 19 | 0.18–0.49 | 0.02–0.15 |
| learning_curve | 57 | 5.89–16.31 | 8.00–35.00 |

Estimativas derivadas dos timings existentes na RTX 4070 Ti; a curva mantém cinco CNNs, pois FT fica abaixo de 48 h. Nested principal foi dispensada devido às 900–2.702 h previstas por versão; análise principal usa externo geográfico único. O protocolo/configuração com o desvio são commitados antes de qualquer classificação externa.

Nested suplementar sem Relief-F: 8528 configurações; 32.70–98.09 h e 10.23–17.78 GiB. Somente dry-run, sem execução; não valida a seleção na grade integral do artigo.
**Correção de viés sem treinamento**

| approach | accuracy_original | accuracy_bias | accuracy_corrected | obstructed_recall_original | obstructed_recall_bias | obstructed_recall_corrected |
| --- | --- | --- | --- | --- | --- | --- |
| A | 94.16% | 4.09% | 90.07% | 91.58% | 7.83% | 83.75% |
| B | 93.87% | 5.25% | 88.62% | 92.83% | 7.17% | 85.66% |
| C | 93.87% | 4.39% | 89.48% | 90.48% | 9.52% | 80.96% |
| D | 93.87% | 4.69% | 89.18% | 90.48% | 9.52% | 80.96% |
| winner | 93.87% | 5.25% | 88.62% | 92.83% | 7.17% | 85.66% |

Fórmula solicitada: média dos dez folds de máximo da abordagem menos finalista; corrigida = original − viés. Vencedor usa toda a grade. Versão conservadora de Tibshirani & Tibshirani (2009), recall como extensão secundária; não é uma validação independente nem garantia de estimativa sem viés. [Referência](https://doi.org/10.1214/08-AOAS224).

| approach | pca_finalist | best_relieff | pca_accuracy_mean | relieff_accuracy_mean | pca_minus_relieff | definition |
| --- | --- | --- | --- | --- | --- | --- |
| B | b_efficientnetb3_pca_300_linear_svm | b_densenet201_relieff_150_random_forest | 0.9387394957983192 | 0.9299159663865544 | 0.0088235294117648 | Highest mean-fold accuracy among all Relief-F configurations; name breaks ties |
| C | c_mobilenet_resnet50_pca_300_linear_svm | c_mobilenet_resnet50_relieff_200_mlp | 0.938655462184874 | 0.915378151260504 | 0.0232773109243697 | Highest mean-fold accuracy among all Relief-F configurations; name breaks ties |
| D | d_mobilenet_resnet50_pca_300_linear_svm | d_efficientnetb0_mobilenet_relieff_300_logistic | 0.938655462184874 | 0.9094117647058824 | 0.0292436974789915 | Highest mean-fold accuracy among all Relief-F configurations; name breaks ties |

Diferença é PCA − melhor Relief-F por média da accuracy nos folds. Diagnósticos completos: `tibshirani_fold_diagnostics.csv`; saídas também em LaTeX.

Commit do protocolo/desvio: `cb7c510e2f649ead6284bb6b033fe8356f3f22b1`. Config SHA-256: `8b64190aef090eca20ae16139bbc6493e51bc3c486e6eaf6f2e9694a6257ca4b`.

**Modelos e limiares congelados**

| artifact | sha256 |
| --- | --- |
| latency.csv | a515e884d35ff88344d4cac812236ca55a89a34c3bc2cc8ab364826f8bb968fc |
| models/ft_EfficientNetB4__seed_1982/model.keras | cf71aa6e904d2586006a795020bf4cb57d91a8adcaaf7a47bcf3c7e3d1b6b7b5 |
| models/ft_EfficientNetB4__seed_1981/model.keras | 657dba2fe8a06b07942e648661068d5c5eb518b3f892d1911a5758012b67cd10 |
| models/ft_EfficientNetB4__seed_1980/model.keras | 28ac2524ad8edd8e8010ada30b9b517ae29f4a6f9b0316a9ff751a7d0fdf5fc9 |
| models/ft_ResNet50__seed_1982/model.keras | 9ec47a71077ee424fbfb8c4b41571033d437eee867cd933be59d8ffcaacab0e4 |
| models/ft_ResNet50__seed_1981/model.keras | cddf27a1b613aa8e31c4a04925a863e4ebee53a1ac79a134a8cb1c495433b28f |
| models/ft_ResNet50__seed_1980/model.keras | 6b88721bd63812a36ba01c4f2f2e75c47c35796122d033c2ba09e8f82ed6ccc9 |
| models/ft_MobileNet__seed_1982/model.keras | 652fb8de96955179929ba010923aec1ed50438a6d06a3774ca7c4bc4be781448 |
| models/ft_MobileNet__seed_1981/model.keras | 85d3d434179b1ed371e2189e85c333a0428dd44df920a6dc984154716a42efe8 |
| models/ft_MobileNet__seed_1980/model.keras | 55bcf03fbb7014a5c4b74f0b4049261a256d5d7c9f7f8017c5f97e06cfd2d763 |
| models/ft_EfficientNetB3__seed_1982/model.keras | b1b887d0ab70387f38e4d640e4e63aa183e12f6e3169e8b4bfcb0704b07b04d1 |
| models/ft_EfficientNetB3__seed_1981/model.keras | fdb92bca07e5ab96bf6a2d3cc8f8a8c6cde4b0807737a047549edf1cf74aff79 |
| models/ft_EfficientNetB3__seed_1980/model.keras | 8fbe80c53a302dbb69d57842e26c6ebcc0b6ecc3f32242c4b12d636c4575c847 |
| models/ft_ResNet101V2__seed_1982/model.keras | 6dee4aafb1ce700691cf626896919f14c136b9b8eeb08fd45ad3cb7f3f13e1bc |
| models/ft_ResNet101V2__seed_1981/model.keras | 3d1ea4ef6c90b782191fddbd6900c661861fa44bc8a9b867f39aeafee79b5999 |
| models/ft_ResNet101V2__seed_1980/model.keras | 5ec940878b1218c32c4b0299c0d681464f8fd1316d697fc3e5da206116445f82 |
| models/b_efficientnetb3_pca_300_linear_svm/pipeline.joblib | 2aa793d50a2461805621a99d2248125649362783419bc9b12bea697f6163798f |
| models/d_mobilenet_resnet50_pca_300_linear_svm/pipeline.joblib | 8c1b8ea0dad457d7d5c0a7039efd3dba35c88ad3e4101c7cc88033373b5118d8 |
| models/c_mobilenet_resnet50_pca_300_linear_svm/pipeline.joblib | 55d2346543f92cfd21f28cf72d5c9118353ea7e9f537f83804289af23f551358 |
| models/a_resnet101v2_logistic/pipeline.joblib | edbc1a5035cc7f3fba8a09b143a1bf183709a59c0ae94f303a364d1549694534 |
| encoders/EfficientNetB3.keras | a95e591c5c360ca8bf45bcfe4c8e9d63f87c932adcf354636f06ad45939e303e |
| encoders/MobileNet.keras | 9949827c04583416302dc0a6970e0e35c587e1f00ade0d2e877547ea73280fcc |
| encoders/ResNet101V2.keras | b3998260ba7af56eaf8f5f6f9e2867204caef14c282d68a1bf8f60c27863bf07 |
| encoders/ResNet50.keras | eeab2df82fb50bdfa48ed203cf8cf2c2d9c72c33a9c87cb4fe8313128556266a |
| thresholds.json | 4b77c86ca8eae15e03fc3332f2d8210c47471638b35170b5eb9fa06354b1e002 |
| calibrated_via_oof.csv | f39eeb9349a76dc493dc58afdbb6d536fa64dc94c09ad08dbe76926d5b8a09f7 |
| fixed_oof_predictions.csv | 4229697b57f5831ed95c5a091fed6fabb92808675a7156eb0b91aceb773d7597 |
| fixed_oof_provenance.json | ee565e3068f249dfbdbd72dddf1d2428b03848490840c2e735c76352321647e0 |
| adapter_D.joblib | 4a7b9a2e3a414553ba948051ce0c040c35478dd53c203b4e5fa010a3ddff11b3 |
| adapter_B.joblib | 340db8b14516bc249cbb42780bd0b19ac68b6266c7ead238675997dd04cb3136 |
| adapter_A.joblib | 19b792ecf93ce6583a42a6c18e164f7599449e3e1729eb708f89b860247d9e31 |
| adapter_winner.joblib | 340db8b14516bc249cbb42780bd0b19ac68b6266c7ead238675997dd04cb3136 |
| adapter_C.joblib | 786e2dfd9617b5ac58dc0c7e3749ec3debfe9eeeb962a0171dd0fe87f78374c5 |

Freeze SHA-256: `d83b57f6b3b853653fe27e8acf88df1e2e725b0542f237ed4d4696d1cdd42f0e`. Modelos/checkpoints binários são locais e ignorados; manifests, custos e hashes são versionados. Treino final só VIA; FT-D usa validação de aproximadamente 20% por grupos, idêntica entre seeds. Os 40 refits originais fornecem OOF canônico verificado sem novos ajustes. Adaptadores/limiares dos fixos herdam o otimismo da seleção; o teste externo verifica a manutenção do alvo de recall 0,95.

**Resultados externos principais, IC 95% por cenas**

| Modelo | Limiar | Accuracy | Balanced accuracy | MCC | AUC | Recall nonclear | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | default | 78.93% [76.95%; 80.98%] | 80.01% [78.06%; 81.96%] | 0.6203 [0.5852; 0.6556] | 0.9029 [0.8850; 0.9197] | 95.22% [93.66%; 96.71%] | 4.78% [3.29%; 6.34%] |
| A | safety | 72.46% [70.41%; 74.55%] | 74.14% [72.15%; 76.18%] | 0.5352 [0.5014; 0.5689] | 0.9029 [0.8850; 0.9197] | 97.68% [96.53%; 98.68%] | 2.32% [1.32%; 3.47%] |
| B | default | 80.90% [78.80%; 82.98%] | 81.63% [79.58%; 83.66%] | 0.6394 [0.6000; 0.6783] | 0.9132 [0.8980; 0.9281] | 91.94% [89.73%; 94.04%] | 8.06% [5.96%; 10.27%] |
| B | safety | 78.36% [76.25%; 80.43%] | 79.56% [77.54%; 81.55%] | 0.6174 [0.5824; 0.6524] | 0.9132 [0.8980; 0.9281] | 96.45% [95.09%; 97.72%] | 3.55% [2.28%; 4.91%] |
| C | default | 79.63% [77.60%; 81.65%] | 80.63% [78.66%; 82.56%] | 0.6290 [0.5931; 0.6647] | 0.9101 [0.8942; 0.9256] | 94.67% [93.00%; 96.24%] | 5.33% [3.76%; 7.00%] |
| C | safety | 76.33% [74.27%; 78.38%] | 77.69% [75.70%; 79.62%] | 0.5876 [0.5527; 0.6211] | 0.9101 [0.8942; 0.9256] | 96.72% [95.39%; 97.94%] | 3.28% [2.06%; 4.61%] |
| D | default | 80.08% [78.10%; 82.03%] | 81.05% [79.13%; 82.93%] | 0.6367 [0.6012; 0.6715] | 0.9085 [0.8926; 0.9242] | 94.81% [93.10%; 96.39%] | 5.19% [3.61%; 6.90%] |
| D | safety | 76.40% [74.28%; 78.43%] | 77.73% [75.70%; 79.70%] | 0.5871 [0.5516; 0.6212] | 0.9085 [0.8926; 0.9242] | 96.45% [95.07%; 97.72%] | 3.55% [2.28%; 4.93%] |
| EfficientNetB3 | default | 77.79% [75.88%; 79.70%] | 78.27% [76.40%; 80.15%] | 0.5669 [0.5306; 0.6036] | 0.8557 [0.8379; 0.8733] | 85.02% [82.83%; 87.19%] | 14.98% [12.81%; 17.17%] |
| EfficientNetB3 | safety | 77.83% [75.93%; 79.74%] | 78.31% [76.45%; 80.18%] | 0.5677 [0.5316; 0.6043] | 0.8557 [0.8379; 0.8733] | 85.02% [82.83%; 87.19%] | 14.98% [12.81%; 17.17%] |
| EfficientNetB4 | default | 71.19% [69.25%; 73.15%] | 72.48% [70.61%; 74.37%] | 0.4752 [0.4390; 0.5113] | 0.7620 [0.7383; 0.7849] | 90.62% [88.81%; 92.36%] | 9.38% [7.64%; 11.19%] |
| EfficientNetB4 | safety | 70.26% [68.33%; 72.17%] | 71.70% [69.84%; 73.53%] | 0.4664 [0.4308; 0.5010] | 0.7620 [0.7383; 0.7849] | 91.89% [90.24%; 93.47%] | 8.11% [6.53%; 9.76%] |
| MobileNet | default | 75.72% [73.73%; 77.69%] | 76.22% [74.25%; 78.16%] | 0.5317 [0.4930; 0.5695] | 0.8480 [0.8294; 0.8665] | 83.20% [80.89%; 85.50%] | 16.80% [14.50%; 19.11%] |
| MobileNet | safety | 74.96% [72.99%; 76.87%] | 75.86% [73.94%; 77.71%] | 0.5328 [0.4960; 0.5679] | 0.8480 [0.8294; 0.8665] | 88.62% [86.74%; 90.42%] | 11.38% [9.58%; 13.26%] |
| ResNet101V2 | default | 65.80% [63.63%; 67.92%] | 65.72% [63.58%; 67.81%] | 0.3141 [0.2714; 0.3561] | 0.6996 [0.6747; 0.7246] | 64.57% [61.63%; 67.40%] | 35.43% [32.60%; 38.37%] |
| ResNet101V2 | safety | 63.01% [61.08%; 64.97%] | 64.05% [62.09%; 66.01%] | 0.2914 [0.2517; 0.3313] | 0.6996 [0.6747; 0.7246] | 78.64% [76.18%; 81.04%] | 21.36% [18.96%; 23.82%] |
| ResNet50 | default | 66.05% [63.92%; 68.20%] | 66.78% [64.65%; 68.92%] | 0.3408 [0.2984; 0.3833] | 0.7267 [0.7013; 0.7518] | 76.96% [74.32%; 79.54%] | 23.04% [20.46%; 25.68%] |
| ResNet50 | safety | 64.55% [62.45%; 66.62%] | 65.90% [63.85%; 67.98%] | 0.3404 [0.2991; 0.3816] | 0.7267 [0.7013; 0.7518] | 84.88% [82.66%; 87.01%] | 15.12% [12.99%; 17.34%] |

FT principal é a média das métricas das três seeds individuais, não a métrica das decisões médias. Bootstrap: 10.000 réplicas, seed 1980, cenas inteiras em estratos nonclear/clear/mistos. Wilson por seed para recall/FNR está no CSV de ICs.

**Ensemble de três seeds, sensibilidade**

| Modelo | Limiar | Accuracy | Balanced accuracy | MCC | AUC | Recall nonclear | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNetB3 | default | 79.25% [77.05%; 81.41%] | 79.84% [77.69%; 81.94%] | 0.5998 [0.5582; 0.6411] | 0.8696 [0.8507; 0.8881] | 88.11% [85.60%; 90.57%] | 11.89% [9.43%; 14.40%] |
| EfficientNetB3 | safety | 79.25% [77.10%; 81.39%] | 79.83% [77.71%; 81.91%] | 0.5995 [0.5582; 0.6404] | 0.8696 [0.8507; 0.8881] | 87.98% [85.46%; 90.40%] | 12.02% [9.60%; 14.54%] |
| EfficientNetB4 | default | 73.22% [70.99%; 75.38%] | 74.51% [72.38%; 76.59%] | 0.5175 [0.4773; 0.5564] | 0.7904 [0.7651; 0.8154] | 92.62% [90.64%; 94.47%] | 7.38% [5.53%; 9.36%] |
| EfficientNetB4 | safety | 71.64% [69.46%; 73.79%] | 73.17% [71.08%; 75.24%] | 0.5036 [0.4650; 0.5416] | 0.7904 [0.7651; 0.8154] | 94.67% [93.02%; 96.28%] | 5.33% [3.72%; 6.98%] |
| MobileNet | default | 77.98% [75.70%; 80.22%] | 78.46% [76.20%; 80.67%] | 0.5704 [0.5259; 0.6139] | 0.8639 [0.8449; 0.8825] | 85.25% [82.45%; 87.96%] | 14.75% [12.04%; 17.55%] |
| MobileNet | safety | 74.11% [71.92%; 76.29%] | 75.54% [73.42%; 77.64%] | 0.5474 [0.5089; 0.5844] | 0.8639 [0.8449; 0.8825] | 95.63% [94.04%; 97.12%] | 4.37% [2.88%; 5.96%] |
| ResNet101V2 | default | 68.65% [66.09%; 71.23%] | 68.78% [66.23%; 71.34%] | 0.3746 [0.3240; 0.4260] | 0.7284 [0.7002; 0.7560] | 70.49% [66.80%; 74.06%] | 29.51% [25.94%; 33.20%] |
| ResNet101V2 | safety | 66.56% [64.24%; 68.87%] | 67.93% [65.66%; 70.19%] | 0.3832 [0.3374; 0.4283] | 0.7284 [0.7002; 0.7560] | 87.16% [84.63%; 89.55%] | 12.84% [10.45%; 15.37%] |
| ResNet50 | default | 67.32% [64.77%; 69.87%] | 68.16% [65.61%; 70.69%] | 0.3705 [0.3200; 0.4208] | 0.7456 [0.7177; 0.7727] | 79.92% [76.73%; 83.06%] | 20.08% [16.94%; 23.27%] |
| ResNet50 | safety | 63.83% [61.61%; 66.09%] | 65.72% [63.54%; 67.98%] | 0.3647 [0.3198; 0.4091] | 0.7456 [0.7177; 0.7727] | 92.35% [90.24%; 94.33%] | 7.65% [5.67%; 9.76%] |

**Substituição e McNemar**

| fixed | fine_tuning | operating_point | endpoint | difference | ci_low | ci_high | substitution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | EfficientNetB3 | default | accuracy | 1.14% | -0.57% | 2.85% | supported |
| A | EfficientNetB3 | default | obstructed_recall | 10.20% | 8.07% | 12.44% | supported |
| A | EfficientNetB3 | default | obstructed_false_negative_rate | -10.20% | -12.44% | -8.07% | supported |
| A | EfficientNetB3 | safety | accuracy | -5.37% | -7.30% | -3.51% | not_supported |
| A | EfficientNetB3 | safety | obstructed_recall | 12.66% | 10.61% | 14.78% | supported |
| A | EfficientNetB3 | safety | obstructed_false_negative_rate | -12.66% | -14.78% | -10.61% | supported |
| A | EfficientNetB4 | default | accuracy | 7.74% | 5.98% | 9.57% | supported |
| A | EfficientNetB4 | default | obstructed_recall | 4.60% | 2.63% | 6.59% | supported |
| A | EfficientNetB4 | default | obstructed_false_negative_rate | -4.60% | -6.59% | -2.63% | supported |
| A | EfficientNetB4 | safety | accuracy | 2.20% | 0.42% | 4.00% | supported |
| A | EfficientNetB4 | safety | obstructed_recall | 5.78% | 4.17% | 7.43% | supported |
| A | EfficientNetB4 | safety | obstructed_false_negative_rate | -5.78% | -7.43% | -4.17% | supported |
| A | MobileNet | default | accuracy | 3.21% | 1.43% | 5.05% | supported |
| A | MobileNet | default | obstructed_recall | 12.02% | 9.73% | 14.34% | supported |
| A | MobileNet | default | obstructed_false_negative_rate | -12.02% | -14.34% | -9.73% | supported |
| A | MobileNet | safety | accuracy | -2.50% | -4.21% | -0.79% | inconclusive |
| A | MobileNet | safety | obstructed_recall | 9.06% | 7.38% | 10.77% | supported |
| A | MobileNet | safety | obstructed_false_negative_rate | -9.06% | -10.77% | -7.38% | supported |
| A | ResNet101V2 | default | accuracy | 13.13% | 10.86% | 15.50% | supported |
| A | ResNet101V2 | default | obstructed_recall | 30.65% | 27.52% | 33.78% | supported |
| A | ResNet101V2 | default | obstructed_false_negative_rate | -30.65% | -33.78% | -27.52% | supported |
| A | ResNet101V2 | safety | accuracy | 9.45% | 7.23% | 11.65% | supported |
| A | ResNet101V2 | safety | obstructed_recall | 19.03% | 16.47% | 21.61% | supported |
| A | ResNet101V2 | safety | obstructed_false_negative_rate | -19.03% | -21.61% | -16.47% | supported |
| A | ResNet50 | default | accuracy | 12.88% | 10.58% | 15.21% | supported |
| A | ResNet50 | default | obstructed_recall | 18.26% | 15.47% | 21.04% | supported |
| A | ResNet50 | default | obstructed_false_negative_rate | -18.26% | -21.04% | -15.47% | supported |
| A | ResNet50 | safety | accuracy | 7.91% | 5.76% | 10.08% | supported |
| A | ResNet50 | safety | obstructed_recall | 12.80% | 10.61% | 15.05% | supported |
| A | ResNet50 | safety | obstructed_false_negative_rate | -12.80% | -15.05% | -10.61% | supported |
| B | EfficientNetB3 | default | accuracy | 3.11% | 1.16% | 5.01% | supported |
| B | EfficientNetB3 | default | obstructed_recall | 6.92% | 4.56% | 9.28% | supported |
| B | EfficientNetB3 | default | obstructed_false_negative_rate | -6.92% | -9.28% | -4.56% | supported |
| B | EfficientNetB3 | safety | accuracy | 0.53% | -1.43% | 2.44% | supported |
| B | EfficientNetB3 | safety | obstructed_recall | 11.43% | 9.28% | 13.60% | supported |
| B | EfficientNetB3 | safety | obstructed_false_negative_rate | -11.43% | -13.60% | -9.28% | supported |
| B | EfficientNetB4 | default | accuracy | 9.71% | 7.81% | 11.70% | supported |
| B | EfficientNetB4 | default | obstructed_recall | 1.32% | -0.91% | 3.54% | supported |
| B | EfficientNetB4 | default | obstructed_false_negative_rate | -1.32% | -3.54% | 0.91% | supported |
| B | EfficientNetB4 | safety | accuracy | 8.10% | 6.24% | 9.99% | supported |
| B | EfficientNetB4 | safety | obstructed_recall | 4.55% | 2.79% | 6.30% | supported |
| B | EfficientNetB4 | safety | obstructed_false_negative_rate | -4.55% | -6.30% | -2.79% | supported |
| B | MobileNet | default | accuracy | 5.18% | 3.20% | 7.16% | supported |
| B | MobileNet | default | obstructed_recall | 8.74% | 6.04% | 11.44% | supported |
| B | MobileNet | default | obstructed_false_negative_rate | -8.74% | -11.44% | -6.04% | supported |
| B | MobileNet | safety | accuracy | 3.41% | 1.53% | 5.26% | supported |
| B | MobileNet | safety | obstructed_recall | 7.83% | 5.96% | 9.81% | supported |
| B | MobileNet | safety | obstructed_false_negative_rate | -7.83% | -9.81% | -5.96% | supported |
| B | ResNet101V2 | default | accuracy | 15.10% | 12.60% | 17.60% | supported |
| B | ResNet101V2 | default | obstructed_recall | 27.37% | 23.91% | 30.84% | supported |
| B | ResNet101V2 | default | obstructed_false_negative_rate | -27.37% | -30.84% | -23.91% | supported |
| B | ResNet101V2 | safety | accuracy | 15.36% | 13.11% | 17.63% | supported |
| B | ResNet101V2 | safety | obstructed_recall | 17.81% | 15.25% | 20.41% | supported |
| B | ResNet101V2 | safety | obstructed_false_negative_rate | -17.81% | -20.41% | -15.25% | supported |
| B | ResNet50 | default | accuracy | 14.85% | 12.45% | 17.25% | supported |
| B | ResNet50 | default | obstructed_recall | 14.98% | 11.92% | 18.10% | supported |
| B | ResNet50 | default | obstructed_false_negative_rate | -14.98% | -18.10% | -11.92% | supported |
| B | ResNet50 | safety | accuracy | 13.81% | 11.64% | 16.05% | supported |
| B | ResNet50 | safety | obstructed_recall | 11.57% | 9.28% | 13.89% | supported |
| B | ResNet50 | safety | obstructed_false_negative_rate | -11.57% | -13.89% | -9.28% | supported |
| C | EfficientNetB3 | default | accuracy | 1.84% | 0.06% | 3.59% | supported |
| C | EfficientNetB3 | default | obstructed_recall | 9.65% | 7.52% | 11.79% | supported |
| C | EfficientNetB3 | default | obstructed_false_negative_rate | -9.65% | -11.79% | -7.52% | supported |
| C | EfficientNetB3 | safety | accuracy | -1.50% | -3.30% | 0.34% | inconclusive |
| C | EfficientNetB3 | safety | obstructed_recall | 11.70% | 9.58% | 13.84% | supported |
| C | EfficientNetB3 | safety | obstructed_false_negative_rate | -11.70% | -13.84% | -9.58% | supported |
| C | EfficientNetB4 | default | accuracy | 8.44% | 6.56% | 10.34% | supported |
| C | EfficientNetB4 | default | obstructed_recall | 4.05% | 2.15% | 5.94% | supported |
| C | EfficientNetB4 | default | obstructed_false_negative_rate | -4.05% | -5.94% | -2.15% | supported |
| C | EfficientNetB4 | safety | accuracy | 6.07% | 4.16% | 7.98% | supported |
| C | EfficientNetB4 | safety | obstructed_recall | 4.83% | 3.09% | 6.55% | supported |
| C | EfficientNetB4 | safety | obstructed_false_negative_rate | -4.83% | -6.55% | -3.09% | supported |
| C | MobileNet | default | accuracy | 3.91% | 2.07% | 5.74% | supported |
| C | MobileNet | default | obstructed_recall | 11.48% | 9.27% | 13.67% | supported |
| C | MobileNet | default | obstructed_false_negative_rate | -11.48% | -13.67% | -9.27% | supported |
| C | MobileNet | safety | accuracy | 1.37% | -0.40% | 3.21% | supported |
| C | MobileNet | safety | obstructed_recall | 8.11% | 6.42% | 9.84% | supported |
| C | MobileNet | safety | obstructed_false_negative_rate | -8.11% | -9.84% | -6.42% | supported |
| C | ResNet101V2 | default | accuracy | 13.83% | 11.46% | 16.23% | supported |
| C | ResNet101V2 | default | obstructed_recall | 30.10% | 26.91% | 33.25% | supported |
| C | ResNet101V2 | default | obstructed_false_negative_rate | -30.10% | -33.25% | -26.91% | supported |
| C | ResNet101V2 | safety | accuracy | 13.32% | 11.10% | 15.59% | supported |
| C | ResNet101V2 | safety | obstructed_recall | 18.08% | 15.40% | 20.76% | supported |
| C | ResNet101V2 | safety | obstructed_false_negative_rate | -18.08% | -20.76% | -15.40% | supported |
| C | ResNet50 | default | accuracy | 13.58% | 11.24% | 15.89% | supported |
| C | ResNet50 | default | obstructed_recall | 17.71% | 14.77% | 20.66% | supported |
| C | ResNet50 | default | obstructed_false_negative_rate | -17.71% | -20.66% | -14.77% | supported |
| C | ResNet50 | safety | accuracy | 11.78% | 9.62% | 13.99% | supported |
| C | ResNet50 | safety | obstructed_recall | 11.84% | 9.55% | 14.21% | supported |
| C | ResNet50 | safety | obstructed_false_negative_rate | -11.84% | -14.21% | -9.55% | supported |
| D | EfficientNetB3 | default | accuracy | 2.28% | 0.60% | 3.95% | supported |
| D | EfficientNetB3 | default | obstructed_recall | 9.79% | 7.75% | 11.88% | supported |
| D | EfficientNetB3 | default | obstructed_false_negative_rate | -9.79% | -11.88% | -7.75% | supported |
| D | EfficientNetB3 | safety | accuracy | -1.44% | -3.25% | 0.38% | inconclusive |
| D | EfficientNetB3 | safety | obstructed_recall | 11.43% | 9.33% | 13.57% | supported |
| D | EfficientNetB3 | safety | obstructed_false_negative_rate | -11.43% | -13.57% | -9.33% | supported |
| D | EfficientNetB4 | default | accuracy | 8.88% | 7.07% | 10.74% | supported |
| D | EfficientNetB4 | default | obstructed_recall | 4.19% | 2.34% | 6.04% | supported |
| D | EfficientNetB4 | default | obstructed_false_negative_rate | -4.19% | -6.04% | -2.34% | supported |
| D | EfficientNetB4 | safety | accuracy | 6.13% | 4.22% | 8.03% | supported |
| D | EfficientNetB4 | safety | obstructed_recall | 4.55% | 2.85% | 6.27% | supported |
| D | EfficientNetB4 | safety | obstructed_false_negative_rate | -4.55% | -6.27% | -2.85% | supported |
| D | MobileNet | default | accuracy | 4.36% | 2.57% | 6.13% | supported |
| D | MobileNet | default | obstructed_recall | 11.61% | 9.43% | 13.80% | supported |
| D | MobileNet | default | obstructed_false_negative_rate | -11.61% | -13.80% | -9.43% | supported |
| D | MobileNet | safety | accuracy | 1.44% | -0.33% | 3.25% | supported |
| D | MobileNet | safety | obstructed_recall | 7.83% | 6.08% | 9.62% | supported |
| D | MobileNet | safety | obstructed_false_negative_rate | -7.83% | -9.62% | -6.08% | supported |
| D | ResNet101V2 | default | accuracy | 14.28% | 11.94% | 16.62% | supported |
| D | ResNet101V2 | default | obstructed_recall | 30.24% | 27.03% | 33.38% | supported |
| D | ResNet101V2 | default | obstructed_false_negative_rate | -30.24% | -33.38% | -27.03% | supported |
| D | ResNet101V2 | safety | accuracy | 13.39% | 11.13% | 15.64% | supported |
| D | ResNet101V2 | safety | obstructed_recall | 17.81% | 15.07% | 20.51% | supported |
| D | ResNet101V2 | safety | obstructed_false_negative_rate | -17.81% | -20.51% | -15.07% | supported |
| D | ResNet50 | default | accuracy | 14.02% | 11.73% | 16.34% | supported |
| D | ResNet50 | default | obstructed_recall | 17.85% | 14.97% | 20.71% | supported |
| D | ResNet50 | default | obstructed_false_negative_rate | -17.85% | -20.71% | -14.97% | supported |
| D | ResNet50 | safety | accuracy | 11.84% | 9.69% | 14.05% | supported |
| D | ResNet50 | safety | obstructed_recall | 11.57% | 9.26% | 13.95% | supported |
| D | ResNet50 | safety | obstructed_false_negative_rate | -11.57% | -13.95% | -9.26% | supported |

Diferença fixo − FT; margem 3 p.p. Supported: limite inferior ≥ −0,03; not_supported: limite superior < −0,03; demais inconclusive. FNR tem orientação invertida. Bootstrap pareado usa os mesmos sorteios de cenas e média das métricas por seed.

| fixed | fine_tuning | seed | operating_point | fixed_only_correct | ft_only_correct | p_value | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1980 | default | 116 | 114 | 0.9474462046254554 | 1.0 |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1981 | default | 161 | 119 | 0.0141318653900061 | 0.1695823846800742 |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1982 | default | 133 | 123 | 0.573861376594554 | 1.0 |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1980 | safety | 100 | 201 | 5.9735507080042464e-09 | 1.9712717336414019e-07 |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1981 | safety | 145 | 206 | 0.0013288485987766 | 0.0292346691730855 |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 1982 | safety | 116 | 208 | 3.6102576435621657e-07 | 1.155282445939893e-05 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1980 | default | 179 | 78 | 2.6316863171382567e-10 | 6.579215792845642e-09 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1981 | default | 235 | 76 | 4.724770247330406e-20 | 1.7954126939855544e-18 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1982 | default | 178 | 72 | 1.5528465028452382e-11 | 4.192685557682143e-10 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1980 | safety | 140 | 124 | 0.3559334048627964 | 1.0 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1981 | safety | 177 | 99 | 3.0958294491821253e-06 | 9.287488347546377e-05 |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 1982 | safety | 122 | 112 | 0.5563912799559659 | 1.0 |
| a_resnet101v2_logistic | ft_MobileNet | 1980 | default | 148 | 135 | 0.4757085730971819 | 1.0 |
| a_resnet101v2_logistic | ft_MobileNet | 1981 | default | 144 | 91 | 0.0006617102716424 | 0.0099256540746365 |
| a_resnet101v2_logistic | ft_MobileNet | 1982 | default | 211 | 125 | 3.1425820255270634e-06 | 6.285164051054128e-05 |
| a_resnet101v2_logistic | ft_MobileNet | 1980 | safety | 101 | 175 | 9.915424082890184e-06 | 0.000267716450238 |
| a_resnet101v2_logistic | ft_MobileNet | 1981 | safety | 100 | 118 | 0.2495157452864138 | 1.0 |
| a_resnet101v2_logistic | ft_MobileNet | 1982 | safety | 147 | 173 | 0.1621464685622107 | 1.0 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1980 | default | 368 | 139 | 6.445647739928668e-25 | 2.964997960367188e-23 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1981 | default | 286 | 123 | 4.668717675109327e-16 | 1.5406768327860783e-14 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1982 | default | 363 | 134 | 2.0104428434931856e-25 | 9.449081364417973e-24 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1980 | safety | 325 | 145 | 6.242919474718282e-17 | 2.8093137636232275e-15 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1981 | safety | 228 | 168 | 0.0029803181171123 | 0.0625866804593583 |
| a_resnet101v2_logistic | ft_ResNet101V2 | 1982 | safety | 325 | 118 | 1.918235203414825e-23 | 9.207528976391158e-22 |
| a_resnet101v2_logistic | ft_ResNet50 | 1980 | default | 322 | 89 | 4.563219329526677e-32 | 2.6010350178302055e-30 |
| a_resnet101v2_logistic | ft_ResNet50 | 1981 | default | 282 | 112 | 5.048614399079018e-18 | 1.7670150396776559e-16 |
| a_resnet101v2_logistic | ft_ResNet50 | 1982 | default | 322 | 116 | 1.7205272994438893e-23 | 7.570320117553113e-22 |
| a_resnet101v2_logistic | ft_ResNet50 | 1980 | safety | 247 | 98 | 5.626535161406935e-16 | 2.4756754710190515e-14 |
| a_resnet101v2_logistic | ft_ResNet50 | 1981 | safety | 211 | 133 | 3.07024005263367e-05 | 0.000736857612632 |
| a_resnet101v2_logistic | ft_ResNet50 | 1982 | safety | 271 | 124 | 1.0657128624841445e-13 | 4.156280163688164e-12 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | default | 135 | 102 | 0.0374280854514103 | 0.3742808545141036 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | default | 196 | 123 | 5.174778288931667e-05 | 0.0008797123091183 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | default | 160 | 119 | 0.0164782347758384 | 0.1812605825342231 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | safety | 128 | 136 | 0.6666801784700446 | 1.0 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | safety | 183 | 151 | 0.0896890776959817 | 1.0 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | safety | 147 | 146 | 1.0 | 1.0 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | default | 217 | 85 | 1.8156492571956943e-14 | 5.628512697306652e-13 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | default | 276 | 86 | 2.3721944070257954e-24 | 1.0674874831616078e-22 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | default | 220 | 83 | 1.8281191159633064e-15 | 5.84998117108258e-14 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | safety | 197 | 88 | 9.404048173516176e-11 | 3.479497824200985e-09 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | safety | 248 | 77 | 4.165643639348884e-22 | 1.9578525104939756e-20 |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | safety | 188 | 85 | 4.182814496053827e-10 | 1.4639850736188397e-08 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1980 | default | 168 | 124 | 0.0117274029867616 | 0.1524562388279014 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1981 | default | 176 | 92 | 3.2147856218405837e-07 | 6.751049805865224e-06 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1982 | default | 235 | 118 | 4.652719163183096e-10 | 1.1166525991639432e-08 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1980 | safety | 142 | 123 | 0.268810030283091 | 1.0 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1981 | safety | 161 | 86 | 2.102848231881478e-06 | 6.518829518832582e-05 |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 1982 | safety | 199 | 132 | 0.0002737453537896 | 0.0062961431371609 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1980 | default | 397 | 137 | 2.4334695536864008e-30 | 1.3384082545275203e-28 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1981 | default | 308 | 114 | 1.0620286391142109e-21 | 4.354317420368265e-20 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1982 | default | 389 | 129 | 2.5983298482397473e-31 | 1.4550647150142586e-29 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1980 | safety | 390 | 117 | 2.5231111089581784e-35 | 1.41294222101658e-33 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1981 | safety | 269 | 116 | 4.106665326229012e-15 | 1.7247994370161852e-13 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 1982 | safety | 394 | 94 | 1.0607721054041792e-44 | 6.364632632425076e-43 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1980 | default | 357 | 93 | 1.6149236123233584e-37 | 9.689541673940149e-36 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1981 | default | 313 | 112 | 4.3970063891623314e-23 | 1.846742683448179e-21 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1982 | default | 344 | 107 | 4.516570106766408e-30 | 2.43894785765386e-28 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1980 | safety | 315 | 73 | 6.384573304280943e-37 | 3.7668982495257567e-35 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1981 | safety | 269 | 98 | 1.5074280162785797e-19 | 6.934168874881467e-18 |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 1982 | safety | 333 | 93 | 9.088536565190329e-33 | 4.9986951108546796e-31 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | default | 123 | 110 | 0.4318458053881849 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | default | 165 | 112 | 0.0017319583224545 | 0.0242474165143637 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | default | 141 | 120 | 0.2156500696366559 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | safety | 108 | 148 | 0.0146283488975205 | 0.2633102801553693 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | safety | 155 | 155 | 1.0 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | safety | 133 | 164 | 0.0815486750371454 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | default | 199 | 87 | 2.955159700243308e-11 | 7.683415220632598e-10 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | default | 251 | 81 | 2.214653952033561e-21 | 8.637150412930887e-20 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | default | 198 | 81 | 1.7967018909926085e-12 | 5.030765294779303e-11 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | safety | 182 | 105 | 6.43084716906581e-06 | 0.0001800637207338 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | safety | 232 | 93 | 7.484259011155358e-15 | 3.068546194573696e-13 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | safety | 173 | 102 | 2.203626545961361e-05 | 0.0005509066364903 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1980 | default | 146 | 122 | 0.1599103154943453 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1981 | default | 151 | 87 | 4.013754744072749e-05 | 0.000722475853933 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1982 | default | 212 | 115 | 8.9529377132288e-08 | 1.969646296910336e-06 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1980 | safety | 119 | 132 | 0.4488553936096015 | 1.0 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1981 | safety | 145 | 102 | 0.0074065503659359 | 0.1407244569527835 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1982 | safety | 171 | 136 | 0.0521438872483488 | 0.7821583087252332 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1980 | default | 378 | 138 | 7.854057410974681e-27 | 3.8484881313775938e-25 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1981 | default | 295 | 121 | 7.254272675631641e-18 | 2.4664527097147573e-16 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1982 | default | 377 | 137 | 6.15080742554529e-27 | 3.075403712772645e-25 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1980 | safety | 359 | 118 | 2.5101837042113322e-29 | 1.330397363232006e-27 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1981 | safety | 249 | 128 | 4.5732626671455637e-10 | 1.554909306829492e-08 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1982 | safety | 378 | 110 | 1.9299574581325204e-35 | 1.1000757511355369e-33 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1980 | default | 331 | 87 | 1.359926705117444e-34 | 7.887574889681174e-33 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1981 | default | 292 | 111 | 7.143913233510155e-20 | 2.6432478963987575e-18 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1982 | default | 322 | 105 | 1.047592164944568e-26 | 5.028442391733926e-25 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1980 | safety | 294 | 84 | 2.029610601511064e-28 | 1.0351014067706426e-26 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1981 | safety | 249 | 110 | 1.625888382840897e-13 | 6.1783758547954086e-12 |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1982 | safety | 308 | 100 | 9.696642729720019e-26 | 4.7513549375628085e-24 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | default | 121 | 101 | 0.2021343206599584 | 1.0 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | default | 160 | 100 | 0.0002389419030681 | 0.0038230704490902 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | default | 140 | 112 | 0.0887728478579271 | 0.7101827828634169 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1980 | safety | 105 | 144 | 0.0158630908540703 | 0.2696725445191966 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1981 | safety | 150 | 149 | 1.0 | 1.0 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 1982 | safety | 131 | 161 | 0.0895061211871608 | 1.0 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | default | 203 | 84 | 1.5736401542716617e-12 | 4.563556447387819e-11 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | default | 253 | 76 | 2.282534176977213e-23 | 9.814896961002019e-22 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | default | 197 | 73 | 2.4716540517427527e-14 | 7.41496215522826e-13 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1980 | safety | 181 | 103 | 4.296603121448619e-06 | 0.000124601490522 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1981 | safety | 231 | 91 | 3.4555391245729165e-15 | 1.4858818235663545e-13 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 1982 | safety | 173 | 101 | 1.6139512859660104e-05 | 0.0004196273343511 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1980 | default | 146 | 115 | 0.0631108724023878 | 0.5679978516214909 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1981 | default | 158 | 87 | 6.73401288864515e-06 | 0.0001279462448842 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1982 | default | 208 | 104 | 3.98604334137928e-09 | 9.167899685172345e-08 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1980 | safety | 117 | 129 | 0.4831705602293273 | 1.0 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1981 | safety | 144 | 100 | 0.0057970522645218 | 0.1159410452904367 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 1982 | safety | 167 | 131 | 0.0424311755040528 | 0.6788988080648463 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1980 | default | 382 | 135 | 2.446362982869071e-28 | 1.2476451212632261e-26 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1981 | default | 298 | 117 | 2.6380037241224985e-19 | 9.496813406840994e-18 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1982 | default | 378 | 131 | 8.535534112834869e-29 | 4.5238330798024805e-27 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1980 | safety | 361 | 119 | 2.232942699045351e-29 | 1.2057890574844896e-27 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1981 | safety | 250 | 128 | 3.44369796774917e-10 | 1.2397312683897012e-08 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 1982 | safety | 378 | 109 | 8.66870810197175e-36 | 5.0278506991436145e-34 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1980 | default | 335 | 84 | 1.441462931913126e-36 | 8.504631298287443e-35 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1981 | default | 293 | 105 | 1.1860855511223696e-21 | 4.744342204489479e-20 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1982 | default | 326 | 102 | 2.231287425370767e-28 | 1.1602694611927988e-26 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1980 | safety | 295 | 84 | 1.302046498289515e-28 | 6.77064179110548e-27 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1981 | safety | 249 | 109 | 9.894909411722215e-14 | 3.957963764688886e-12 |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 1982 | safety | 305 | 96 | 1.8641811110382215e-26 | 9.320905555191107e-25 |

McNemar exato por seed; Holm aplicado à família de comparações de cada painel/ponto. Ensemble é sensibilidade e não substitui a análise principal.

**VIA × domínio externo e calibração**

| model | via_original_accuracy | via_original_obstructed_recall | via_tibshirani_accuracy | via_tibshirani_obstructed_recall | external_accuracy | external_balanced_accuracy | external_mcc | external_roc_auc | external_obstructed_recall | external_obstructed_false_negative_rate | external_minus_original_accuracy | external_minus_tibshirani_accuracy | notice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | 0.941596638655462 | 0.9158088235294116 | 0.900672268907563 | 0.8374999999999999 | 0.7893401015228426 | 0.8001450288762852 | 0.6203449194691154 | 0.9029358635692643 | 0.9521857923497268 | 0.0478142076502732 | -0.1522565371326195 | -0.1113321673847204 | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| b_efficientnetb3_pca_300_linear_svm | 0.9387394957983192 | 0.9283088235294118 | 0.886218487394958 | 0.8566176470588233 | 0.8090101522842639 | 0.8163345246419599 | 0.6394286275861601 | 0.9132384171134073 | 0.9193989071038252 | 0.0806010928961748 | -0.1297293435140554 | -0.077208335110694 | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| c_mobilenet_resnet50_pca_300_linear_svm | 0.938655462184874 | 0.9047794117647056 | 0.8947899159663864 | 0.8095588235294116 | 0.7963197969543147 | 0.8062990443632972 | 0.6290176872685465 | 0.910057817315412 | 0.9467213114754098 | 0.0532786885245901 | -0.1423356652305591 | -0.0984701190120715 | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| d_mobilenet_resnet50_pca_300_linear_svm | 0.938655462184874 | 0.9047794117647056 | 0.8917647058823529 | 0.8095588235294116 | 0.800761421319797 | 0.8105366068422551 | 0.6367183753564216 | 0.9085444021443556 | 0.9480874316939892 | 0.0519125683060109 | -0.1378940408650769 | -0.0910032845625559 | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| ft_EfficientNetB3 | 0.9278431372549018 | 0.9181372549019609 | — | — | 0.7779187817258882 | 0.7827135075406382 | 0.566933549234372 | 0.8557105659579243 | 0.8501821493624772 | 0.1498178506375227 | -0.1499243555290136 | — | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| ft_EfficientNetB4 | 0.9249299719887952 | 0.9104166666666667 | — | — | 0.7119289340101522 | 0.7248184980878632 | 0.475157106859416 | 0.7619762126744879 | 0.906193078324226 | 0.0938069216757741 | -0.2130010379786431 | — | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| ft_MobileNet | 0.9055462184873948 | 0.888235294117647 | — | — | 0.7571912013536379 | 0.76215264289229 | 0.5317228456909588 | 0.8480431352998558 | 0.831967213114754 | 0.1680327868852458 | -0.1483550171337569 | — | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| ft_ResNet101V2 | 0.7174509803921569 | 0.6291666666666667 | — | — | 0.657994923857868 | 0.6571804400935781 | 0.3140730431788595 | 0.6995741395384973 | 0.645719489981785 | 0.3542805100182149 | -0.0594560565342888 | — | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |
| ft_ResNet50 | 0.8364425770308122 | 0.8096813725490198 | — | — | 0.6605329949238578 | 0.6677684113295177 | 0.3407605073884595 | 0.7266529407194467 | 0.7695810564663023 | 0.2304189435336976 | -0.1759095821069542 | — | Original VIA native-default mean fold metrics; external default uses frozen probabilities. Corrected recall is a conservative secondary adaptation. |

Mudança de domínio com ICs independentes entre VIA e externo: `domain_shift.csv`. Brier/confiabilidade por domínio: `domain_calibration.csv`; confiabilidade externa: `calibration_reliability.csv`, `reliability.png/.pdf`. O 10-fold VIA conserva decisões nativas; o padrão externo usa probabilidades congeladas. Calibração VIA dos fixos é otimista porque o adaptador é ajustado no mesmo OOF selecionado. Tibshirani é complemento descritivo, sem IC independente.

**Curva de aprendizado e custo de atualização**

| pool_fraction | model | training_images | operating_point | endpoint | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | a_resnet101v2_logistic | 342 | default | accuracy | 0.7842639593908629 | 0.7537106292150954 | 0.8128205128205128 |
| 0.0 | a_resnet101v2_logistic | 342 | default | obstructed_recall | 0.9453551912568307 | 0.9201101928374656 | 0.967479674796748 |
| 0.0 | b_efficientnetb3_pca_300_linear_svm | 342 | default | accuracy | 0.8210659898477157 | 0.7916666666666666 | 0.8487928843710291 |
| 0.0 | b_efficientnetb3_pca_300_linear_svm | 342 | default | obstructed_recall | 0.9344262295081968 | 0.9071618037135278 | 0.9590163934426228 |
| 0.0 | c_mobilenet_resnet50_pca_300_linear_svm | 342 | default | accuracy | 0.8083756345177665 | 0.7795969773299747 | 0.8358778625954199 |
| 0.0 | c_mobilenet_resnet50_pca_300_linear_svm | 342 | default | obstructed_recall | 0.9480874316939892 | 0.9228571428571428 | 0.9694444444444444 |
| 0.0 | d_mobilenet_resnet50_pca_300_linear_svm | 342 | default | accuracy | 0.8045685279187818 | 0.7751626206037971 | 0.8325183374083129 |
| 0.0 | d_mobilenet_resnet50_pca_300_linear_svm | 342 | default | obstructed_recall | 0.9480874316939892 | 0.9222222222222222 | 0.970108695652174 |
| 0.0 | ft_EfficientNetB3 | 342 | default | accuracy | 0.7766497461928932 | 0.7495748299319728 | 0.8024059289116263 |
| 0.0 | ft_EfficientNetB3 | 342 | default | obstructed_recall | 0.8506375227686703 | 0.8194060042062572 | 0.8790254148067042 |
| 0.0 | ft_EfficientNetB4 | 342 | default | accuracy | 0.7072758037225042 | 0.6792366963335753 | 0.7337692810457516 |
| 0.0 | ft_EfficientNetB4 | 342 | default | obstructed_recall | 0.9007285974499089 | 0.8740740740740741 | 0.9252271809624752 |
| 0.0 | ft_MobileNet | 342 | default | accuracy | 0.7580372250423012 | 0.7301371739222047 | 0.7852242955763109 |
| 0.0 | ft_MobileNet | 342 | default | obstructed_recall | 0.8369763205828779 | 0.8055555555555554 | 0.866485074663307 |
| 0.0 | ft_ResNet101V2 | 342 | default | accuracy | 0.6357868020304567 | 0.6033143038505313 | 0.6686946770558118 |
| 0.0 | ft_ResNet101V2 | 342 | default | obstructed_recall | 0.6174863387978142 | 0.5727427902511657 | 0.6622466727162734 |
| 0.0 | ft_ResNet50 | 342 | default | accuracy | 0.6497461928934009 | 0.6173312273312273 | 0.681946486435346 |
| 0.0 | ft_ResNet50 | 342 | default | obstructed_recall | 0.750455373406193 | 0.710502283105023 | 0.7884632616487454 |
| 0.25 | a_resnet101v2_logistic | 540 | default | accuracy | 0.9086294416243654 | 0.8861573463297171 | 0.9295774647887324 |
| 0.25 | a_resnet101v2_logistic | 540 | default | obstructed_recall | 0.9262295081967212 | 0.8965493751761721 | 0.9518413597733713 |
| 0.25 | b_efficientnetb3_pca_300_linear_svm | 540 | default | accuracy | 0.8934010152284264 | 0.8690629011553272 | 0.916252246732026 |
| 0.25 | b_efficientnetb3_pca_300_linear_svm | 540 | default | obstructed_recall | 0.9316939890710382 | 0.9030470914127424 | 0.9565217391304348 |
| 0.25 | c_mobilenet_resnet50_pca_300_linear_svm | 540 | default | accuracy | 0.8959390862944162 | 0.8720481937195443 | 0.9178926141352062 |
| 0.25 | c_mobilenet_resnet50_pca_300_linear_svm | 540 | default | obstructed_recall | 0.9398907103825136 | 0.9142070729950316 | 0.9622641509433962 |
| 0.25 | d_mobilenet_resnet50_pca_300_linear_svm | 540 | default | accuracy | 0.901015228426396 | 0.8778047935716264 | 0.9225806451612903 |
| 0.25 | d_mobilenet_resnet50_pca_300_linear_svm | 540 | default | obstructed_recall | 0.953551912568306 | 0.9289772727272728 | 0.9748603351955308 |
| 0.25 | ft_EfficientNetB3 | 540 | default | accuracy | 0.920896785109983 | 0.9027593776552854 | 0.9381313131313131 |
| 0.25 | ft_EfficientNetB3 | 540 | default | obstructed_recall | 0.918943533697632 | 0.8935574229691877 | 0.9418282548476452 |
| 0.25 | ft_EfficientNetB4 | 540 | default | accuracy | 0.91243654822335 | 0.8946678936255675 | 0.9290240811153359 |
| 0.25 | ft_EfficientNetB4 | 540 | default | obstructed_recall | 0.9335154826958106 | 0.9115456238361264 | 0.9534662867996202 |
| 0.25 | ft_MobileNet | 540 | default | accuracy | 0.9090524534686972 | 0.890527146962449 | 0.9260694794290512 |
| 0.25 | ft_MobileNet | 540 | default | obstructed_recall | 0.9225865209471767 | 0.8983516483516483 | 0.943674976915974 |
| 0.25 | ft_ResNet101V2 | 540 | default | accuracy | 0.7461928934010151 | 0.723062952592769 | 0.7693590244410587 |
| 0.25 | ft_ResNet101V2 | 540 | default | obstructed_recall | 0.587431693989071 | 0.5519414456439646 | 0.6221616712079927 |
| 0.25 | ft_ResNet50 | 540 | default | accuracy | 0.808798646362098 | 0.7839582086029819 | 0.8329103482619032 |
| 0.25 | ft_ResNet50 | 540 | default | obstructed_recall | 0.8442622950819672 | 0.8078931789857157 | 0.8769953051643192 |
| 0.5 | a_resnet101v2_logistic | 740 | default | accuracy | 0.9251269035532996 | 0.9048208242687936 | 0.9444444444444444 |
| 0.5 | a_resnet101v2_logistic | 740 | default | obstructed_recall | 0.9453551912568307 | 0.920545610211706 | 0.966759002770083 |
| 0.5 | b_efficientnetb3_pca_300_linear_svm | 740 | default | accuracy | 0.8946700507614213 | 0.8713375796178344 | 0.916669254658385 |
| 0.5 | b_efficientnetb3_pca_300_linear_svm | 740 | default | obstructed_recall | 0.918032786885246 | 0.8866257976928269 | 0.9460227272727272 |
| 0.5 | c_mobilenet_resnet50_pca_300_linear_svm | 740 | default | accuracy | 0.9124365482233504 | 0.8901220259995095 | 0.9325710079826568 |
| 0.5 | c_mobilenet_resnet50_pca_300_linear_svm | 740 | default | obstructed_recall | 0.953551912568306 | 0.9301675977653632 | 0.9731903485254692 |
| 0.5 | d_mobilenet_resnet50_pca_300_linear_svm | 740 | default | accuracy | 0.9263959390862944 | 0.9057784011681445 | 0.9450833896777102 |
| 0.5 | d_mobilenet_resnet50_pca_300_linear_svm | 740 | default | obstructed_recall | 0.9562841530054644 | 0.933518005540166 | 0.9756756756756756 |
| 0.5 | ft_EfficientNetB3 | 740 | default | accuracy | 0.9357021996615904 | 0.9185282387133727 | 0.951040456810445 |
| 0.5 | ft_EfficientNetB3 | 740 | default | obstructed_recall | 0.9280510018214936 | 0.9025500910746812 | 0.9508196721311476 |
| 0.5 | ft_EfficientNetB4 | 740 | default | accuracy | 0.9297800338409477 | 0.912546613707507 | 0.9455529935629936 |
| 0.5 | ft_EfficientNetB4 | 740 | default | obstructed_recall | 0.929872495446266 | 0.9046320895145828 | 0.9517048751793704 |
| 0.5 | ft_MobileNet | 740 | default | accuracy | 0.9297800338409477 | 0.9120785317968416 | 0.946168502819465 |
| 0.5 | ft_MobileNet | 740 | default | obstructed_recall | 0.918032786885246 | 0.8909211356622395 | 0.94229112833764 |
| 0.5 | ft_ResNet101V2 | 740 | default | accuracy | 0.7567681895093062 | 0.7346234491315137 | 0.7786198523698524 |
| 0.5 | ft_ResNet101V2 | 740 | default | obstructed_recall | 0.657559198542805 | 0.6217948717948718 | 0.6923799816345269 |
| 0.5 | ft_ResNet50 | 740 | default | accuracy | 0.8502538071065989 | 0.8251980303142423 | 0.8738515799424118 |
| 0.5 | ft_ResNet50 | 740 | default | obstructed_recall | 0.8469945355191256 | 0.8101604278074866 | 0.8801420539702732 |
| 1.0 | a_resnet101v2_logistic | 1130 | default | accuracy | 0.9365482233502536 | 0.9172054861116832 | 0.9544271657436708 |
| 1.0 | a_resnet101v2_logistic | 1130 | default | obstructed_recall | 0.942622950819672 | 0.91644908616188 | 0.9657894736842104 |
| 1.0 | b_efficientnetb3_pca_300_linear_svm | 1130 | default | accuracy | 0.9238578680203045 | 0.9035981651968092 | 0.9426023133612418 |
| 1.0 | b_efficientnetb3_pca_300_linear_svm | 1130 | default | obstructed_recall | 0.9289617486338798 | 0.8983946200857966 | 0.9564032697547684 |
| 1.0 | c_mobilenet_resnet50_pca_300_linear_svm | 1130 | default | accuracy | 0.932741116751269 | 0.9123020065493564 | 0.9514447937477964 |
| 1.0 | c_mobilenet_resnet50_pca_300_linear_svm | 1130 | default | obstructed_recall | 0.9617486338797812 | 0.940340909090909 | 0.980227381236572 |
| 1.0 | d_mobilenet_resnet50_pca_300_linear_svm | 1130 | default | accuracy | 0.9390862944162436 | 0.9197452229299364 | 0.9563603810985356 |
| 1.0 | d_mobilenet_resnet50_pca_300_linear_svm | 1130 | default | obstructed_recall | 0.9590163934426228 | 0.9366381260866458 | 0.9781420765027322 |
| 1.0 | ft_EfficientNetB3 | 1130 | default | accuracy | 0.9513536379018612 | 0.9380339195979898 | 0.9630542388935076 |
| 1.0 | ft_EfficientNetB3 | 1130 | default | obstructed_recall | 0.942622950819672 | 0.9197352288289355 | 0.9623507805325988 |
| 1.0 | ft_EfficientNetB4 | 1130 | default | accuracy | 0.9483925549915398 | 0.9352319612358474 | 0.9604593561831724 |
| 1.0 | ft_EfficientNetB4 | 1130 | default | obstructed_recall | 0.9681238615664844 | 0.9514652014652012 | 0.9824074074074076 |
| 1.0 | ft_MobileNet | 1130 | default | accuracy | 0.9500846023688662 | 0.936026936026936 | 0.9625324234954012 |
| 1.0 | ft_MobileNet | 1130 | default | obstructed_recall | 0.9508196721311476 | 0.9287584896902176 | 0.969378827646544 |
| 1.0 | ft_ResNet101V2 | 1130 | default | accuracy | 0.8257191201353637 | 0.800771208226221 | 0.8490966032680777 |
| 1.0 | ft_ResNet101V2 | 1130 | default | obstructed_recall | 0.8105646630236794 | 0.7738086993800289 | 0.8453716896948763 |
| 1.0 | ft_ResNet50 | 1130 | default | accuracy | 0.8904399323181048 | 0.8703454632510033 | 0.9090175933945008 |
| 1.0 | ft_ResNet50 | 1130 | default | obstructed_recall | 0.8852459016393442 | 0.8542208333333332 | 0.91324200913242 |

| model | seed | pool_fraction | training_images | elapsed_seconds | training_seconds | effective_epochs | best_epoch | batch_size | peak_gpu_memory_bytes | peak_gpu_memory_status | peak_host_memory_bytes | total_parameters | trainable_parameters | model_size_mb | reload_check | update_reduction_seconds | update_classifier_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft_EfficientNetB4 | 1982 | 0.25 | 540 | 438.8878504899913 | 389.9311738140241 | 16.0 | 6.0 | 8.0 | 5850774784.0 | measured | 7481610240.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1981 | 0.25 | 540 | 485.9687650710112 | 430.3640834510442 | 18.0 | 8.0 | 8.0 | 6225653248.0 | measured | 7578058752.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1980 | 0.25 | 540 | 535.3230961800145 | 475.99396813902416 | 20.0 | 10.0 | 8.0 | 6630668032.0 | measured | 7858950144.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1982 | 0.25 | 540 | 247.545499550004 | 201.13569701998492 | 22.0 | 12.0 | 16.0 | 2792039168.0 | measured | 8097124352.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1981 | 0.25 | 540 | 235.79931915798807 | 190.66117331496207 | 21.0 | 11.0 | 16.0 | 3118473216.0 | measured | 8235298816.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1980 | 0.25 | 540 | 304.3080345610215 | 250.51806447899435 | 28.0 | 18.0 | 16.0 | 3378483456.0 | measured | 8397303808.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1982 | 0.25 | 540 | 129.84703730701585 | 102.72045257099673 | 21.0 | 11.0 | 16.0 | 2749414144.0 | measured | 8397303808.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1981 | 0.25 | 540 | 174.85765257599996 | 144.3926853009325 | 30.0 | 20.0 | 16.0 | 2796436480.0 | measured | 8397303808.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1980 | 0.25 | 540 | 140.48953372100368 | 111.74326631909936 | 23.0 | 13.0 | 16.0 | 2881990656.0 | measured | 8397303808.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1982 | 0.25 | 540 | 758.8626234859984 | 672.8321478070575 | 45.0 | 35.0 | 16.0 | 7819701248.0 | measured | 9095155712.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1981 | 0.25 | 540 | 423.7585774000036 | 367.78968133401935 | 24.0 | 14.0 | 16.0 | 8067092224.0 | measured | 9495891968.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1980 | 0.25 | 540 | 457.54875890599214 | 396.97659789194586 | 26.0 | 16.0 | 16.0 | 8321941248.0 | measured | 9898106880.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1982 | 0.25 | 540 | 498.6646984899999 | 409.5150544839853 | 31.0 | 21.0 | 16.0 | 5070167552.0 | measured | 10846822400.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1981 | 0.25 | 540 | 575.3338557870011 | 470.7803636630706 | 36.0 | 26.0 | 16.0 | 5645287680.0 | measured | 11208364032.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1980 | 0.25 | 540 | 542.9109391059901 | 449.0151713890664 | 34.0 | 24.0 | 16.0 | 6195678720.0 | measured | 11559317504.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.003.jpg', 'clear.005.jpg', 'clear.013.jpg', 'clear.035.jpg', 'clear.038.jpg', 'clear.040.jpg', 'clear.041.jpg', 'clear.044.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| b_efficientnetb3_pca_300_linear_svm | -1 | 0.25 | 540 | 7.170347437000601 | 6.556093099003192 | — | — | — | — | — | — | — | — | — | — | 6.548208235995844 | 0.007884863007348 |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | 0.25 | 540 | 6.666402145987377 | 6.077944097021828 | — | — | — | — | — | — | — | — | — | — | 6.066415708017304 | 0.0115283890045247 |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | 0.25 | 540 | 7.016111229022499 | 6.396750652027549 | — | — | — | — | — | — | — | — | — | — | 6.389118832012173 | 0.0076318200153764 |
| a_resnet101v2_logistic | -1 | 0.25 | 540 | 0.6108271669945678 | 0.458871837996412 | — | — | — | — | — | — | — | — | — | — | 0.023342202999629 | 0.4355296349967829 |
| ft_EfficientNetB4 | 1982 | 0.5 | 740 | 30.203398960991763 | 889.2902369299553 | 26.0 | 16.0 | 4.0 | 10058529536.0 | measured | 13754609664.0 | 17903585.0 | 17778378.0 | 73.258183 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1981 | 0.5 | 740 | 750.100619705976 | 687.4800704490625 | 24.0 | 14.0 | 8.0 | 6039207168.0 | measured | 6882516992.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1980 | 0.5 | 740 | 493.6826432469534 | 443.53824907494715 | 15.0 | 5.0 | 8.0 | 6564316160.0 | measured | 7375798272.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1982 | 0.5 | 740 | 312.2606719179894 | 266.70925961516326 | 25.0 | 15.0 | 16.0 | 2584346368.0 | measured | 7687802880.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1981 | 0.5 | 740 | 334.3848026019987 | 284.464478537906 | 27.0 | 17.0 | 16.0 | 2830985984.0 | measured | 7789883392.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1980 | 0.5 | 740 | 260.96911844296847 | 217.22496378619687 | 20.0 | 10.0 | 16.0 | 3208721408.0 | measured | 7975698432.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1982 | 0.5 | 740 | 291.8532833570498 | 256.999591953354 | 45.0 | 35.0 | 16.0 | 2588637952.0 | measured | 7975698432.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1981 | 0.5 | 740 | 194.24722958303755 | 165.75437693111598 | 29.0 | 19.0 | 16.0 | 2639516928.0 | measured | 7975698432.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1980 | 0.5 | 740 | 154.07853923604125 | 126.693202932016 | 22.0 | 12.0 | 16.0 | 2690065152.0 | measured | 7975698432.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1982 | 0.5 | 740 | 571.1543821759988 | 507.8145079209353 | 28.0 | 18.0 | 16.0 | 7595998464.0 | measured | 8659591168.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1981 | 0.5 | 740 | 884.0312138599693 | 802.4052963600261 | 45.0 | 35.0 | 16.0 | 7952679936.0 | measured | 9065906176.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1980 | 0.5 | 740 | 441.0996384539758 | 386.531122923945 | 21.0 | 11.0 | 16.0 | 8243168512.0 | measured | 9466683392.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1982 | 0.5 | 740 | 439.0829623320024 | 363.6054393430241 | 24.0 | 14.0 | 16.0 | 4795223296.0 | measured | 10406998016.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1981 | 0.5 | 740 | 539.0968786189915 | 452.7139446781948 | 30.0 | 20.0 | 16.0 | 5408691456.0 | measured | 10775846912.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1980 | 0.5 | 740 | 342.49872309400234 | 277.7393392000231 | 18.0 | 8.0 | 16.0 | 6001772800.0 | measured | 11126009856.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.000.jpg', 'clear.006.jpg', 'clear.009.jpg', 'clear.020.jpg', 'clear.023.jpg', 'clear.032.jpg', 'clear.033.jpg', 'clear.037.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| b_efficientnetb3_pca_300_linear_svm | -1 | 0.5 | 740 | 8.10032588202739 | 7.55896961397957 | — | — | — | — | — | — | — | — | — | — | 7.5471777800121345 | 0.0117918339674361 |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | 0.5 | 740 | 7.4356373909977265 | 6.868837829970289 | — | — | — | — | — | — | — | — | — | — | 6.853769377979916 | 0.0150684519903734 |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | 0.5 | 740 | 7.859705650014803 | 7.270984696981031 | — | — | — | — | — | — | — | — | — | — | 7.259931746986695 | 0.0110529499943368 |
| a_resnet101v2_logistic | -1 | 0.5 | 740 | 0.7251657380256801 | 0.5720945840002968 | — | — | — | — | — | — | — | — | — | — | 0.0312289129942655 | 0.5408656710060313 |
| ft_EfficientNetB4 | 1982 | 1.0 | 1130 | 30.246938374999445 | 1037.2779930629183 | 23.0 | 13.0 | 4.0 | 9948101120.0 | measured | 13324795904.0 | 17903585.0 | 17778378.0 | 73.258183 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1981 | 1.0 | 1130 | 914.785848828964 | 856.2642768899677 | 22.0 | 12.0 | 8.0 | 6029389824.0 | measured | 6907019264.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB4 | 1980 | 1.0 | 1130 | 953.7405308060115 | 893.112557744782 | 23.0 | 13.0 | 8.0 | 6326080512.0 | measured | 7405654016.0 | 17903585.0 | 17778378.0 | 73.275795 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1982 | 1.0 | 1130 | 336.4916518959799 | 293.1565690120333 | 21.0 | 11.0 | 16.0 | 2508204032.0 | measured | 7685599232.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1981 | 1.0 | 1130 | 306.35239029000513 | 266.63254705793224 | 19.0 | 9.0 | 16.0 | 2815553024.0 | measured | 7811719168.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet50 | 1980 | 1.0 | 1130 | 401.82118388602976 | 352.7850144089316 | 26.0 | 16.0 | 16.0 | 3185221120.0 | measured | 8017907712.0 | 23850242.0 | 23797122.0 | 96.057143 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1982 | 1.0 | 1130 | 205.98374310298823 | 179.4726841719239 | 23.0 | 13.0 | 16.0 | 2603201792.0 | measured | 8017907712.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1981 | 1.0 | 1130 | 211.9478648029617 | 184.98874826775864 | 24.0 | 14.0 | 16.0 | 2644469248.0 | measured | 8017907712.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_MobileNet | 1980 | 1.0 | 1130 | 174.437516201986 | 149.18635259394068 | 19.0 | 9.0 | 16.0 | 2726151168.0 | measured | 8017907712.0 | 3360322.0 | 3338434.0 | 13.763727 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1982 | 1.0 | 1130 | 485.2336771489936 | 436.6344109188649 | 18.0 | 8.0 | 16.0 | 7613849088.0 | measured | 8685096960.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1981 | 1.0 | 1130 | 411.7151858219877 | 367.3991778919008 | 15.0 | 5.0 | 16.0 | 7963477504.0 | measured | 9087406080.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_EfficientNetB3 | 1980 | 1.0 | 1130 | 652.1543478120002 | 595.4322473341017 | 25.0 | 15.0 | 16.0 | 8086582272.0 | measured | 9492799488.0 | 10980529.0 | 10893226.0 | 45.272971 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1982 | 1.0 | 1130 | 577.7905757440021 | 499.43724197905976 | 27.0 | 17.0 | 16.0 | 4781187072.0 | measured | 10481192960.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1981 | 1.0 | 1130 | 481.8625523539959 | 410.9876051080646 | 22.0 | 12.0 | 16.0 | 5355275008.0 | measured | 10847875072.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| ft_ResNet101V2 | 1980 | 1.0 | 1130 | 580.8176459140377 | 502.1596298168879 | 27.0 | 17.0 | 16.0 | 6005851904.0 | measured | 11205656576.0 | 42889090.0 | 42791426.0 | 172.894126 | {'status': 'passed', 'image_count': 8, 'filenames': ['clear.009.jpg', 'clear.011.jpg', 'clear.012.jpg', 'clear.019.jpg', 'clear.021.jpg', 'clear.033.jpg', 'clear.036.jpg', 'clear.038.jpg'], 'max_abs_difference': 0.0, 'atol': 1e-05, 'kind': 'probabilities', 'decisions_identical': True} | — | — |
| b_efficientnetb3_pca_300_linear_svm | -1 | 1.0 | 1130 | 10.079174605023582 | 9.435986687021796 | — | — | — | — | — | — | — | — | — | — | 9.410952287027612 | 0.0250343999941833 |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | 1.0 | 1130 | 9.086729112023022 | 8.45508519199211 | — | — | — | — | — | — | — | — | — | — | 8.419550349004567 | 0.0355348429875448 |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | 1.0 | 1130 | 9.695546970993746 | 9.02801967196865 | — | — | — | — | — | — | — | — | — | — | 9.008446168969384 | 0.0195735029992647 |
| a_resnet101v2_logistic | -1 | 1.0 | 1130 | 1.060384874988813 | 0.8823970970115624 | — | — | — | — | — | — | — | — | — | — | 0.0589696540264412 | 0.823427442985121 |

Teste B e pool são disjuntos por cena. Frações 0/25/50/100% são subconjuntos aninhados de cenas inteiras; valores reais estão em `subsets.json` e `split.csv`. 0% reutiliza modelos finais; demais tamanhos reajustam sem busca nova. Custo de atualização é retreino fresco, não warm start. B nunca entra em seleção, early stopping ou limiares. Figura `learning_curve.png/.pdf`.

**Verificações finais da curva**

| Verificação | Esperado | Obtido |
| --- | --- | --- |
| Ajustes novos | 57 | 57 |
| Imagens no teste B | 788 | 788 |
| Imagens no pool | 788 | 788 |
| Predições nos quatro pontos | 59888 | 59888 |
| Predições duplicadas | 0 | 0 |
| Cenas entre teste B e pool | 0 | 0 |

**Tempo e disco medidos**

| Etapa | Estado | Início UTC | Fim UTC | Horas ativas |
| --- | --- | --- | --- | --- |
| final_models | failed | 2026-10-07T00:55:24.446722+00:00 | 2026-10-07T02:07:48.346606+00:00 | 1.2066 |
| final_models | completed | 2026-10-07T10:24:41.396712+00:00 | 2026-10-07T10:46:38.972996+00:00 | 0.366 |
| external_evaluation | completed | 2026-10-07T10:46:38.977300+00:00 | 2026-10-07T10:57:02.360121+00:00 | 0.1732 |
| external_analysis | completed | 2026-10-07T10:57:02.360830+00:00 | 2026-10-07T11:04:27.642839+00:00 | 0.1237 |
| learning_curve | failed | 2026-10-07T11:04:27.643400+00:00 | 2026-10-07T13:03:07.553644+00:00 | 1.9778 |
| final_models | completed | 2026-10-07T14:49:36.600166+00:00 | 2026-10-07T14:51:05.661967+00:00 | 0.0247 |
| external_evaluation | completed | 2026-10-07T14:51:05.674328+00:00 | 2026-10-07T14:51:11.880947+00:00 | 0.0017 |
| learning_curve | failed | 2026-10-07T14:51:11.882116+00:00 | 2026-10-07T16:51:27.263683+00:00 | 2.0043 |
| learning_curve | completed | 2026-10-07T16:51:27.278043+00:00 | 2026-10-07T18:49:28.618926+00:00 | 1.967 |
| stage2_report | completed | 2026-10-07T18:49:28.631617+00:00 | 2026-10-07T18:49:30.128480+00:00 | 0.0004 |

| Diretório | Arquivos locais | GiB lógicos | GiB alocados |
| --- | --- | --- | --- |
| results/external_test | 261 | 8.5968 | 8.5975 |
| results/learning_curve | 540 | 27.2578 | 27.2591 |

Horas ativas são duração de cada processo, incluindo recarga, I/O, inferência e verificações; intervalos entre retomadas não contam como computação. Retomadas verificam o freeze e reutilizam predições principais idênticas, sem nova inferência principal. Históricos completos de tentativas: `execution_history.json`; checkpoints e custos por job ficam nos respectivos manifests. Quando reconstruído a partir do registro de recuperação, o fim da tentativa é aproximado pelo evento imediatamente seguinte. Disco inclui modelos/checkpoints locais e saídas das etapas; exclui imagens e cache canônico já existentes. Interrupções e recuperação de memória estão documentadas em PROVENANCE.md.

**Testes e evidência de execução**

168 passed, 1 skipped, 46 warnings, 7 subtests passed in 161.38s (0:02:41)

Log completo: `results/stage2_verification/pytest.log`. Progresso: `progress.json`; logs por etapa em `logs/stage2/`. Resultados científicos publicados no commit `c8ddc77df5a7fae3356a5ec689be85fea4c623a9`; auditorias posteriores acrescentam verificações sem alterar as predições congeladas.
