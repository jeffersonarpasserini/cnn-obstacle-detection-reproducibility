# Etapa 3 — exploratory/post hoc

Análises exploratórias e posteriores ao teste externo, continuando `b64221f`. Não substituem resultados principais. Ordem executada: 0 → 4 → 3 → 2 → 1 → 5. Nenhum modelo principal, limiar, predição ou rótulo protegido foi alterado. Nenhum FT agrupado executado. Revisão do autor concluída em 2026-10-13: 12 clusters nomeados e 148 erros categorizados; rotulagem independente às cegas analisada.

**0. Registro e hashes protegidos — exploratory/post hoc**

| Verificação | Esperado | Obtido |
| --- | --- | --- |
| Arquivos protegidos | todos intactos | 5364 |
| Conteúdo protegido (GiB) | hash antes = depois | 123.274 |
| Arquivos alterados | 0 | 0 |
| Pacotes de treino | inalterados | True |
| Nomes de clusters preenchidos | 12 | 12 |
| Categorias preenchidas | 148 | 148 |

Inventários SHA-256 completos: `results/stage3/protected_before.json`, `protected_after.json` e `protected_hash_comparison.csv`. Incluem binários, imagens, resultados e configs; ambiente auxiliar OpenCV reutilizado, sem instalação no ambiente de treino. Registro: `2026-10-08T17:52:30.196963+00:00`. Baixa luz pré-fixada em Y médio < **93.746906**, percentil 10 das 342 imagens VIA, sem consultar erros/rótulos. Protocolo e PROVENANCE receberam o adendo pós-teste antes da execução.

| Verificação | Esperado | Obtido | analysis_scope |
| --- | --- | --- | --- |
| Grouped refits | 40 | 40 | exploratory/post hoc |
| Cluster mosaics | 12 | 12 | exploratory/post hoc |
| Paired AUC contrasts | 20 | 20 | exploratory/post hoc |
| Paired specificity contrasts | 60 | 60 | exploratory/post hoc |
| Image attribute rows | 1918 | 1918 | exploratory/post hoc |
| Logistic models/seeds | 19 | 19 | exploratory/post hoc |
| Blinded images | 100 | 100 | exploratory/post hoc |

Dry-run: [0.35, 2.5] horas e [6.0, 9.0] GiB adicionais. O espaço inclui os 40 pipelines completos de refits com PCA. Estimativas amplas para CPU, features em cache e hashing dos binários.

**4. Decisão nativa × adaptador — exploratory/post hoc**

| model | native_accuracy | adapter_accuracy | disagreements | native_only_correct | adapter_only_correct | mcnemar_p | expected_native_accuracy | expected_disagreements | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | 0.809010152284264 | 0.7893401015228426 | 61 | 46 | 15 | 8.837556327558756e-05 | 0.80901 | 61 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 0.8128172588832487 | 0.809010152284264 | 12 | 9 | 3 | 0.14599609375 | 0.8128173 | 12 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 0.807741116751269 | 0.7963197969543148 | 34 | 26 | 8 | 0.00293505564332 | 0.8077411 | 34 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 0.8153553299492385 | 0.800761421319797 | 39 | 31 | 8 | 0.0002940768754342 | 0.8153553 | 39 | exploratory/post hoc |

Todas as métricas/ICs: `native_adapter_metrics.csv/.tex`; proporções também com Wilson. Decisão nativa: margem > 0. Principal original: score_clear ≥ 0,5. AUC da sensibilidade nativa usa native_score; AUC do adaptador usa score_clear. Resultados de substituição nativa (Δ=3 p.p.), como sensibilidade:

| fixed | fine_tuning | endpoint | difference | ci_low | ci_high | substitution | decision | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | ft_EfficientNetB3 | accuracy | 0.0310913705583757 | 0.0144369636651328 | 0.0478616593430657 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB3 | obstructed_recall | 0.081511839708561 | 0.0599988814317673 | 0.1029550574562611 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB3 | obstructed_false_negative_rate | -0.081511839708561 | -0.1029550574562612 | -0.0599988814317674 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | accuracy | 0.0970812182741117 | 0.0793145956212656 | 0.1157412467902005 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | obstructed_recall | 0.0255009107468123 | 0.0051563345234231 | 0.0459467225846534 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | obstructed_false_negative_rate | -0.0255009107468124 | -0.0459467225846535 | -0.0051563345234231 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | accuracy | 0.051818950930626 | 0.0336618918013302 | 0.070122330388541 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | obstructed_recall | 0.0997267759562842 | 0.0759258011380651 | 0.123641304347826 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | obstructed_false_negative_rate | -0.0997267759562841 | -0.123641304347826 | -0.0759258011380652 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | accuracy | 0.1510152284263959 | 0.1278833321113152 | 0.1744286915090319 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | obstructed_recall | 0.2859744990892532 | 0.2537033929903056 | 0.3182653555209915 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | obstructed_false_negative_rate | -0.2859744990892532 | -0.3182653555209915 | -0.2537033929903057 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | accuracy | 0.1484771573604061 | 0.124999353403683 | 0.1715089160375215 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | obstructed_recall | 0.1621129326047359 | 0.1331496467639449 | 0.190800013342073 | supported | native sensitivity | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | obstructed_false_negative_rate | -0.1621129326047359 | -0.190800013342073 | -0.133149646763945 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | accuracy | 0.0348984771573605 | 0.0154493911039593 | 0.0541241217414143 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_recall | 0.0651183970856101 | 0.0410886544021577 | 0.0890455754352031 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_false_negative_rate | -0.0651183970856102 | -0.089045575435203 | -0.0410886544021577 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | accuracy | 0.1008883248730965 | 0.0813515652930865 | 0.1208468692180347 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_recall | 0.0091074681238615 | -0.0133333333333333 | 0.0313513953299161 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_false_negative_rate | -0.0091074681238616 | -0.0313513953299161 | 0.0133333333333333 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | accuracy | 0.0556260575296108 | 0.0355104629042486 | 0.0754221463187707 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | obstructed_recall | 0.0833333333333333 | 0.056241286638112 | 0.111111111111111 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | obstructed_false_negative_rate | -0.0833333333333333 | -0.111111111111111 | -0.056241286638112 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | accuracy | 0.1548223350253806 | 0.1300601709320129 | 0.1798167842191789 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | obstructed_recall | 0.2695810564663023 | 0.2346331173165811 | 0.3044077134986226 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | obstructed_false_negative_rate | -0.2695810564663023 | -0.3044077134986226 | -0.2346331173165812 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | accuracy | 0.1522842639593908 | 0.1280417436761261 | 0.1766839767958289 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | obstructed_recall | 0.145719489981785 | 0.1143644028723908 | 0.177871324793994 | supported | native sensitivity | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | obstructed_false_negative_rate | -0.145719489981785 | -0.1778713247939941 | -0.1143644028723907 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | accuracy | 0.0298223350253807 | 0.0122122572363412 | 0.0474350925421436 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_recall | 0.0856102003642986 | 0.0640142661179696 | 0.1074535666193838 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_false_negative_rate | -0.0856102003642987 | -0.1074535666193838 | -0.0640142661179697 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | accuracy | 0.0958121827411168 | 0.0769551192682283 | 0.1153684460819281 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_recall | 0.02959927140255 | 0.0091443219071531 | 0.0494353556328805 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_false_negative_rate | -0.0295992714025501 | -0.0494353556328805 | -0.0091443219071531 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | accuracy | 0.050549915397631 | 0.0326796345527389 | 0.0687199331338328 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | obstructed_recall | 0.1038251366120218 | 0.0814916977395243 | 0.1259880821578267 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | obstructed_false_negative_rate | -0.1038251366120218 | -0.1259880821578267 | -0.0814916977395244 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | accuracy | 0.1497461928934009 | 0.125990151571212 | 0.1733598734315751 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | obstructed_recall | 0.2900728597449908 | 0.2571671628306524 | 0.322428961490234 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | obstructed_false_negative_rate | -0.2900728597449908 | -0.3224289614902341 | -0.2571671628306525 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | accuracy | 0.1472081218274111 | 0.1239044636481751 | 0.1706301388749783 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | obstructed_recall | 0.1662112932604735 | 0.136300731392514 | 0.1956734597296387 | supported | native sensitivity | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | obstructed_false_negative_rate | -0.1662112932604735 | -0.1956734597296387 | -0.136300731392514 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | accuracy | 0.0374365482233503 | 0.0211274936061381 | 0.0539395013969482 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_recall | 0.0869763205828779 | 0.0665748233819725 | 0.1081795110133609 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | obstructed_false_negative_rate | -0.0869763205828779 | -0.1081795110133609 | -0.0665748233819725 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | accuracy | 0.1034263959390863 | 0.0856776433003623 | 0.1217616850849056 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_recall | 0.0309653916211293 | 0.0111549893387523 | 0.0497041062345536 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | obstructed_false_negative_rate | -0.0309653916211293 | -0.0497041062345534 | -0.0111549893387524 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | accuracy | 0.0581641285956006 | 0.0407073395586747 | 0.0754914261815139 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | obstructed_recall | 0.1051912568306011 | 0.0830037752443979 | 0.1270492283630382 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | obstructed_false_negative_rate | -0.1051912568306011 | -0.1270492283630382 | -0.0830037752443979 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | accuracy | 0.1573604060913704 | 0.1343123180724923 | 0.1806634871985829 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | obstructed_recall | 0.2914389799635701 | 0.258948303618105 | 0.3231907909762457 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | obstructed_false_negative_rate | -0.2914389799635701 | -0.3231907909762456 | -0.258948303618105 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | accuracy | 0.1548223350253806 | 0.1318942931168785 | 0.1778056759135952 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | obstructed_recall | 0.1675774134790528 | 0.1382333705445238 | 0.1966175141211475 | supported | native sensitivity | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | obstructed_false_negative_rate | -0.1675774134790528 | -0.1966175141211476 | -0.1382333705445238 | supported | native sensitivity | exploratory/post hoc |

**3. Finalistas nos folds agrupados — exploratory/post hoc**

| model | endpoint | estimate | ci_low | ci_high | bootstrap_valid | bootstrap_total | design | mean_fold | sd_fold | method | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | accuracy | 0.935672514619883 | 0.9093567251461988 | 0.9619883040935672 | 10000.0 | 10000.0 | grouped | 0.9354575163398692 | 0.0586075373675194 | — | exploratory/post hoc |
| a_resnet101v2_logistic | balanced_accuracy | 0.9349529512403764 | 0.9081402908468776 | 0.9610778443113772 | 10000.0 | 10000.0 | grouped | 0.9346813725490196 | 0.060067758748664 | — | exploratory/post hoc |
| a_resnet101v2_logistic | mcc | 0.8725972644018329 | 0.8195304139213871 | 0.9239403952755224 | 10000.0 | 10000.0 | grouped | 0.8746066072975054 | 0.1146369335888974 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_recall | 0.9041916167664672 | 0.8562874251497006 | 0.9461077844311376 | 10000.0 | 10000.0 | grouped | 0.9036764705882352 | 0.0888594482194777 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_false_negative_rate | 0.0958083832335329 | 0.0538922155688622 | 0.1437125748502994 | 10000.0 | 10000.0 | grouped | 0.0963235294117647 | 0.0888594482194777 | — | exploratory/post hoc |
| a_resnet101v2_logistic | roc_auc | 0.9795722840034218 | 0.9658169375534644 | 0.9906928999144567 | 10000.0 | 10000.0 | grouped | 0.9768546131042716 | 0.0251078645281081 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_recall | 0.9041916167664672 | 0.850042076143469 | 0.9401642410578412 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_false_negative_rate | 0.0958083832335329 | 0.0598357589421588 | 0.1499579238565309 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| a_resnet101v2_logistic | accuracy | 0.9415204678362572 | 0.9152046783625732 | 0.9649122807017544 | 10000.0 | 10000.0 | original | 0.9415966386554622 | 0.0307492351301036 | — | exploratory/post hoc |
| a_resnet101v2_logistic | balanced_accuracy | 0.9409409751924722 | 0.9144054747647564 | 0.9646193327630452 | 10000.0 | 10000.0 | original | 0.9407475490196078 | 0.0315282051374323 | — | exploratory/post hoc |
| a_resnet101v2_logistic | mcc | 0.8838195559623587 | 0.8324343504758879 | 0.9303028515446488 | 10000.0 | 10000.0 | original | 0.8866451120243889 | 0.0592432658906188 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_recall | 0.9161676646706588 | 0.874251497005988 | 0.9580838323353292 | 10000.0 | 10000.0 | original | 0.9158088235294116 | 0.0655940693937996 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_false_negative_rate | 0.0838323353293413 | 0.0419161676646706 | 0.1257485029940119 | 10000.0 | 10000.0 | original | 0.0841911764705883 | 0.0655940693937996 | — | exploratory/post hoc |
| a_resnet101v2_logistic | roc_auc | 0.9775534644995724 | 0.962497005988024 | 0.989803250641574 | 10000.0 | 10000.0 | original | 0.97755430603614 | 0.016192691815752 | — | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_recall | 0.9161676646706588 | 0.8642123148783147 | 0.949407522832664 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_false_negative_rate | 0.0838323353293413 | 0.0505924771673359 | 0.1357876851216853 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | accuracy | 0.9239766081871345 | 0.8947368421052632 | 0.9502923976608189 | 10000.0 | 10000.0 | grouped | 0.9243464052287582 | 0.0471568740686374 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | balanced_accuracy | 0.92366124893071 | 0.8941317365269461 | 0.950196749358426 | 10000.0 | 10000.0 | grouped | 0.924434554523564 | 0.0471347321639099 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | mcc | 0.8480191898818489 | 0.7896728871725085 | 0.9008483787740186 | 10000.0 | 10000.0 | grouped | 0.8515319208026331 | 0.0948858966270235 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_recall | 0.9101796407185628 | 0.8622754491017964 | 0.9520958083832336 | 10000.0 | 10000.0 | grouped | 0.911029411764706 | 0.0567654678400552 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_false_negative_rate | 0.0898203592814371 | 0.0479041916167665 | 0.1377245508982035 | 10000.0 | 10000.0 | grouped | 0.0889705882352941 | 0.0567654678400552 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | roc_auc | 0.975431993156544 | 0.9603763900769888 | 0.9873053892215568 | 10000.0 | 10000.0 | grouped | 0.9760919383233168 | 0.0220319199799619 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_recall | 0.9101796407185628 | 0.8571009047696807 | 0.9448121726864636 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_false_negative_rate | 0.0898203592814371 | 0.0551878273135363 | 0.1428990952303192 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | accuracy | 0.9385964912280702 | 0.912280701754386 | 0.9619883040935672 | 10000.0 | 10000.0 | original | 0.9387394957983194 | 0.0373200290032887 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | balanced_accuracy | 0.9383575705731396 | 0.9118220701454234 | 0.9618990590248075 | 10000.0 | 10000.0 | original | 0.938500816993464 | 0.0371969429586922 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | mcc | 0.8772105410481054 | 0.8245953266229875 | 0.924047785951419 | 10000.0 | 10000.0 | original | 0.8812500581720467 | 0.0714468280341902 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_recall | 0.9281437125748504 | 0.8862275449101796 | 0.9640718562874252 | 10000.0 | 10000.0 | original | 0.9283088235294118 | 0.072816189629272 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_false_negative_rate | 0.0718562874251497 | 0.0359281437125748 | 0.1137724550898203 | 10000.0 | 10000.0 | original | 0.0716911764705882 | 0.072816189629272 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | roc_auc | 0.9771770744225834 | 0.9627373823781008 | 0.9884687767322498 | 10000.0 | 10000.0 | original | 0.9792495674740485 | 0.0175502534075419 | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_recall | 0.9281437125748504 | 0.8786129749579787 | 0.9584203832626684 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_false_negative_rate | 0.0718562874251497 | 0.0415796167373316 | 0.1213870250420212 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.9298245614035088 | 0.9005847953216374 | 0.956140350877193 | 10000.0 | 10000.0 | grouped | 0.9299019607843138 | 0.055653988154843 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | 0.9289649272882806 | 0.8997091531223268 | 0.9553635585970917 | 10000.0 | 10000.0 | grouped | 0.9294558393532852 | 0.0563287802813331 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | mcc | 0.8614744240714539 | 0.8060836655278135 | 0.912563723877856 | 10000.0 | 10000.0 | grouped | 0.8651598395689073 | 0.1073872230012118 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.8922155688622755 | 0.844311377245509 | 0.934131736526946 | 10000.0 | 10000.0 | grouped | 0.8922794117647058 | 0.0933313573382028 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.1077844311377245 | 0.0658682634730538 | 0.155688622754491 | 10000.0 | 10000.0 | grouped | 0.1077205882352941 | 0.0933313573382028 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.9812489307100084 | 0.967971770744226 | 0.9914798973481608 | 10000.0 | 10000.0 | grouped | 0.9791516800218538 | 0.0197692096867613 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.8922155688622755 | 0.8360662655768416 | 0.9307265311148004 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.1077844311377245 | 0.0692734688851998 | 0.1639337344231584 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.9385964912280702 | 0.912280701754386 | 0.9619883040935672 | 10000.0 | 10000.0 | original | 0.938655462184874 | 0.0350285684001392 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | 0.9378100940975193 | 0.9111377245508984 | 0.9614884516680924 | 10000.0 | 10000.0 | original | 0.9381740196078432 | 0.0350845149815376 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | mcc | 0.8787680421174565 | 0.8269181418095051 | 0.9251279781398244 | 10000.0 | 10000.0 | original | 0.8818521795774235 | 0.0653961769407612 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.9041916167664672 | 0.8562874251497006 | 0.9461077844311376 | 10000.0 | 10000.0 | original | 0.9047794117647058 | 0.0691604390548353 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0958083832335329 | 0.0538922155688622 | 0.1437125748502994 | 10000.0 | 10000.0 | original | 0.0952205882352941 | 0.0691604390548353 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.9805645851154832 | 0.966398631308811 | 0.9914798973481608 | 10000.0 | 10000.0 | original | 0.9817870530565168 | 0.0213269355582394 | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.9041916167664672 | 0.850042076143469 | 0.9401642410578412 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0958083832335329 | 0.0598357589421588 | 0.1499579238565309 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.9298245614035088 | 0.9005847953216374 | 0.956140350877193 | 10000.0 | 10000.0 | grouped | 0.9299019607843138 | 0.055653988154843 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | 0.9289649272882806 | 0.8997091531223268 | 0.9553635585970917 | 10000.0 | 10000.0 | grouped | 0.9294558393532852 | 0.0563287802813331 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | mcc | 0.8614744240714539 | 0.8060836655278135 | 0.912563723877856 | 10000.0 | 10000.0 | grouped | 0.8651598395689073 | 0.1073872230012118 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.8922155688622755 | 0.844311377245509 | 0.934131736526946 | 10000.0 | 10000.0 | grouped | 0.8922794117647058 | 0.0933313573382028 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.1077844311377245 | 0.0658682634730538 | 0.155688622754491 | 10000.0 | 10000.0 | grouped | 0.1077205882352941 | 0.0933313573382028 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.9815568862275448 | 0.968349016253208 | 0.9917878528656972 | 10000.0 | 10000.0 | grouped | 0.9791516800218538 | 0.0197692096867613 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.8922155688622755 | 0.8360662655768416 | 0.9307265311148004 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.1077844311377245 | 0.0692734688851998 | 0.1639337344231584 | — | — | grouped | — | — | Wilson pooled proportion | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.9385964912280702 | 0.912280701754386 | 0.9619883040935672 | 10000.0 | 10000.0 | original | 0.938655462184874 | 0.0350285684001392 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | 0.9378100940975193 | 0.9111377245508984 | 0.9614884516680924 | 10000.0 | 10000.0 | original | 0.9381740196078432 | 0.0350845149815376 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | mcc | 0.8787680421174565 | 0.8269181418095051 | 0.9251279781398244 | 10000.0 | 10000.0 | original | 0.8818521795774235 | 0.0653961769407612 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.9041916167664672 | 0.8562874251497006 | 0.9461077844311376 | 10000.0 | 10000.0 | original | 0.9047794117647058 | 0.0691604390548353 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0958083832335329 | 0.0538922155688622 | 0.1437125748502994 | 10000.0 | 10000.0 | original | 0.0952205882352941 | 0.0691604390548353 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.9809751924721984 | 0.9670479041916168 | 0.9917194183062448 | 10000.0 | 10000.0 | original | 0.9821342752787392 | 0.0213082242087238 | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | 0.9041916167664672 | 0.850042076143469 | 0.9401642410578412 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0958083832335329 | 0.0598357589421588 | 0.1499579238565309 | — | — | original | — | — | Wilson pooled proportion | exploratory/post hoc |

mean_fold e sd_fold descrevem os dez folds; estimate e seu IC usam as predições OOF agregadas. Bootstrap estratificado por imagem (10.000, seed 1980), pareado entre desenhos; Wilson para recall/FNR. AUC agrupada de margens OOF é descritiva: classificadores de folds diferentes têm escalas próprias. Nenhuma seleção nova; extração canônica de src/fixed_refit.py e fábrica de pipelines original reutilizadas.

| model | endpoint | difference | ci_low | ci_high | analysis_scope |
| --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | accuracy | -0.0058479532163743 | -0.0204678362573099 | 0.0087719298245614 | exploratory/post hoc |
| a_resnet101v2_logistic | balanced_accuracy | -0.0059880239520957 | -0.0205474764756201 | 0.0084345594525234 | exploratory/post hoc |
| a_resnet101v2_logistic | mcc | -0.0112222915605257 | -0.041033184162084 | 0.0180667049363448 | exploratory/post hoc |
| a_resnet101v2_logistic | roc_auc | 0.0020188195038494 | -0.0023267750213857 | 0.0068776732249786 | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_recall | -0.0119760479041916 | -0.029940119760479 | 0.0 | exploratory/post hoc |
| a_resnet101v2_logistic | obstructed_false_negative_rate | 0.0119760479041916 | 0.0 | 0.029940119760479 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | accuracy | -0.0146198830409357 | -0.0321637426900585 | 0.0029239766081871 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | balanced_accuracy | -0.0146963216424295 | -0.0322497861420016 | 0.0025834046193327 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | mcc | -0.0291913511662564 | -0.0644020248473866 | 0.0057086028142579 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | roc_auc | -0.0017450812660393 | -0.0054747647562019 | 0.0015055603079555 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_recall | -0.0179640718562874 | -0.0419161676646706 | 0.0 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | obstructed_false_negative_rate | 0.0179640718562874 | 0.0 | 0.0419161676646706 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | accuracy | -0.0087719298245614 | -0.023391812865497 | 0.0029239766081871 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | -0.0088451668092386 | -0.023541488451668 | 0.0029940119760479 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | mcc | -0.0172936180460026 | -0.0451015326588125 | 0.0055502778660806 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.0006843455945252 | -0.0022925577416594 | 0.003798118049615 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | -0.0119760479041916 | -0.0359281437125749 | 0.0119760479041916 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0119760479041916 | -0.0119760479041916 | 0.0359281437125749 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | accuracy | -0.0087719298245614 | -0.023391812865497 | 0.0029239766081871 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | balanced_accuracy | -0.0088451668092386 | -0.023541488451668 | 0.0029940119760479 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | mcc | -0.0172936180460026 | -0.0451015326588125 | 0.0055502778660806 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | roc_auc | 0.0005816937553464 | -0.002497861420017 | 0.0037639007698887 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_recall | -0.0119760479041916 | -0.0359281437125749 | 0.0119760479041916 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | obstructed_false_negative_rate | 0.0119760479041916 | -0.0119760479041916 | 0.0359281437125749 | exploratory/post hoc |

| model | group_id | filename | peer | original_test_fold | peer_original_test_fold | opposite_original_train_test | grouped_test_fold | true_label | original_prediction | grouped_prediction | prediction_changed | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | scene_0133 | clear.132.jpg | nonclear.151.jpg | 6 | 8 | True | 3 | 1 | 1 | 1 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0133 | nonclear.151.jpg | clear.132.jpg | 8 | 6 | True | 3 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0229 | nonclear.053.jpg | nonclear.109.jpg | 5 | 8 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0229 | nonclear.109.jpg | nonclear.053.jpg | 8 | 5 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0278 | nonclear.102.jpg | nonclear.104.jpg | 7 | 2 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0278 | nonclear.104.jpg | nonclear.102.jpg | 2 | 7 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0292 | nonclear.116.jpg | nonclear.117.jpg | 8 | 7 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| a_resnet101v2_logistic | scene_0292 | nonclear.117.jpg | nonclear.116.jpg | 7 | 8 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0133 | clear.132.jpg | nonclear.151.jpg | 6 | 8 | True | 3 | 1 | 1 | 1 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0133 | nonclear.151.jpg | clear.132.jpg | 8 | 6 | True | 3 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0229 | nonclear.053.jpg | nonclear.109.jpg | 5 | 8 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0229 | nonclear.109.jpg | nonclear.053.jpg | 8 | 5 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0278 | nonclear.102.jpg | nonclear.104.jpg | 7 | 2 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0278 | nonclear.104.jpg | nonclear.102.jpg | 2 | 7 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0292 | nonclear.116.jpg | nonclear.117.jpg | 8 | 7 | True | 1 | 0 | 0 | 1 | True | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | scene_0292 | nonclear.117.jpg | nonclear.116.jpg | 7 | 8 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0133 | clear.132.jpg | nonclear.151.jpg | 6 | 8 | True | 3 | 1 | 1 | 1 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0133 | nonclear.151.jpg | clear.132.jpg | 8 | 6 | True | 3 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0229 | nonclear.053.jpg | nonclear.109.jpg | 5 | 8 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0229 | nonclear.109.jpg | nonclear.053.jpg | 8 | 5 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0278 | nonclear.102.jpg | nonclear.104.jpg | 7 | 2 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0278 | nonclear.104.jpg | nonclear.102.jpg | 2 | 7 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0292 | nonclear.116.jpg | nonclear.117.jpg | 8 | 7 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | scene_0292 | nonclear.117.jpg | nonclear.116.jpg | 7 | 8 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0133 | clear.132.jpg | nonclear.151.jpg | 6 | 8 | True | 3 | 1 | 1 | 1 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0133 | nonclear.151.jpg | clear.132.jpg | 8 | 6 | True | 3 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0229 | nonclear.053.jpg | nonclear.109.jpg | 5 | 8 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0229 | nonclear.109.jpg | nonclear.053.jpg | 8 | 5 | True | 6 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0278 | nonclear.102.jpg | nonclear.104.jpg | 7 | 2 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0278 | nonclear.104.jpg | nonclear.102.jpg | 2 | 7 | True | 2 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0292 | nonclear.116.jpg | nonclear.117.jpg | 8 | 7 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | scene_0292 | nonclear.117.jpg | nonclear.116.jpg | 7 | 8 | True | 1 | 0 | 0 | 0 | False | exploratory/post hoc |

Os pares são colocados juntos no teste agrupado. opposite_original_train_test indica se cada imagem teve sua parceira no treino quando foi testada no desenho original. Mudança de predição não é causalmente atribuída ao agrupamento, pois outros membros dos folds também mudam.

Fine-tuning agrupado opcional: [7.487080183645138, 20.733452816248075] horas, [45, 100] GiB, 150 jobs. Somente estimativa; não executado.

**2. Separabilidade e recall igualado — exploratory/post hoc**

| model | analysis | endpoint | estimate | ci_low | ci_high | seed | recall_target | oracle | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | single fixed | roc_auc | 0.9029358635692644 | 0.8850296244648541 | 0.9197091217129852 | — | — | — | exploratory/post hoc |
| a_resnet101v2_logistic | — | clear_specificity | 0.7428909952606635 | 0.7003609035143824 | 0.7953494037333937 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | — | clear_specificity | 0.6516587677725119 | 0.6009732360097324 | 0.7081914748012198 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | — | clear_specificity | 0.5592417061611374 | 0.4208616782511113 | 0.6559270028494713 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | single fixed | roc_auc | 0.9132384171134074 | 0.8979558177387521 | 0.9280554852857412 | — | — | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | — | clear_specificity | 0.7511848341232228 | 0.6989244061317628 | 0.8025855758626341 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | — | clear_specificity | 0.6599526066350712 | 0.6070123871657152 | 0.719047619047619 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | — | clear_specificity | 0.5864928909952607 | 0.4957670623167252 | 0.663622061576867 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | single fixed | roc_auc | 0.910057817315412 | 0.8942124523911842 | 0.9255929793991704 | — | — | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.783175355450237 | 0.7299169926454766 | 0.81965413915345 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.6540284360189573 | 0.5895233649184399 | 0.7189010741248771 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.5781990521327014 | 0.5091733341528009 | 0.6450089510323382 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | single fixed | roc_auc | 0.9085444021443556 | 0.8926418568880754 | 0.9241751998144844 | — | — | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.773696682464455 | 0.7330897354720697 | 0.8129373696719656 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.6516587677725119 | 0.5841076671373182 | 0.7318501170960188 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | — | clear_specificity | 0.5556872037914692 | 0.5055207982201847 | 0.641774894443846 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB3 | individual-seed metric mean | roc_auc | 0.8557105659579244 | 0.8378713633329696 | 0.8733488120999727 | — | — | — | exploratory/post hoc |
| ft_EfficientNetB3 | — | clear_specificity | 0.6623222748815166 | 0.6159070736327114 | 0.7029178290925795 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB3 | — | clear_specificity | 0.5351500789889415 | 0.4654748723498723 | 0.592927366883548 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB3 | — | clear_specificity | 0.4135071090047393 | 0.3408973660768942 | 0.5045983597465441 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB4 | individual-seed metric mean | roc_auc | 0.7619762126744879 | 0.7382530529841524 | 0.7848832039901116 | — | — | — | exploratory/post hoc |
| ft_EfficientNetB4 | — | clear_specificity | 0.5548973143759873 | 0.5069310325998438 | 0.6019912189931658 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB4 | — | clear_specificity | 0.433649289099526 | 0.3786279032455012 | 0.4905278790923186 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| ft_EfficientNetB4 | — | clear_specificity | 0.3345181674565561 | 0.2901289887421685 | 0.4054479568582323 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| ft_MobileNet | individual-seed metric mean | roc_auc | 0.8480431352998559 | 0.8294165415133492 | 0.8665121242405581 | — | — | — | exploratory/post hoc |
| ft_MobileNet | — | clear_specificity | 0.6287519747235387 | 0.5895241195306298 | 0.6751218575243085 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| ft_MobileNet | — | clear_specificity | 0.533175355450237 | 0.4705876382221586 | 0.5873397067091919 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| ft_MobileNet | — | clear_specificity | 0.4474723538704581 | 0.3983449077002217 | 0.5042050654677133 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet101V2 | individual-seed metric mean | roc_auc | 0.6995741395384973 | 0.6746727851133918 | 0.7245887751045927 | — | — | — | exploratory/post hoc |
| ft_ResNet101V2 | — | clear_specificity | 0.3376777251184834 | 0.2882437146430743 | 0.3786068308371735 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet101V2 | — | clear_specificity | 0.2191943127962085 | 0.1794864226682408 | 0.256079875180476 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet101V2 | — | clear_specificity | 0.1563981042654028 | 0.1279300754866905 | 0.1956146397746985 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet50 | individual-seed metric mean | roc_auc | 0.7266529407194469 | 0.7012964843257697 | 0.7518473871259563 | — | — | — | exploratory/post hoc |
| ft_ResNet50 | — | clear_specificity | 0.3850710900473933 | 0.3409544440780339 | 0.4287423945794932 | — | 0.9 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet50 | — | clear_specificity | 0.2752764612954186 | 0.2312329384363282 | 0.3188230670744719 | — | 0.95 | external threshold, not deployable | exploratory/post hoc |
| ft_ResNet50 | — | clear_specificity | 0.1872037914691943 | 0.1602407877122525 | 0.248920550362634 | — | 0.97 | external threshold, not deployable | exploratory/post hoc |

| fixed | fine_tuning | difference | ci_low | ci_high | p_value | p_holm | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 0.0472252976113398 | 0.0323629158616421 | 0.0619245908028339 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 0.1409596508947763 | 0.1203671287737553 | 0.1620418292004002 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | 0.0548927282694083 | 0.0382554084232816 | 0.0713040311054679 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | 0.2033617240307669 | 0.1781019881988279 | 0.2291333023051429 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | 0.1762829228498174 | 0.1513088977023925 | 0.2020329163837506 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 0.0575278511554829 | 0.0420856318433033 | 0.0735301447121324 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 0.1512622044389194 | 0.1309227470334311 | 0.1723873398482723 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 0.0651952818135515 | 0.0489679000231716 | 0.0819044093178239 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 0.21366427757491 | 0.189251536712281 | 0.2388754297127534 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 0.1865854763939605 | 0.1625225626615296 | 0.2117279662365354 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.0543472513574875 | 0.0405999562219178 | 0.0681716486190475 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.148081604640924 | 0.1283901691514417 | 0.1677975431577295 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.062014682015556 | 0.0467483114943553 | 0.0769806006370067 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.2104836777769145 | 0.1869740395549544 | 0.2344785844131248 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.183404876595965 | 0.159752554226288 | 0.2081085630242868 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.0528338361864311 | 0.0397522948916022 | 0.0662896322876544 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.1465681894698676 | 0.1273501014310009 | 0.1660719925574741 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.0605012668444996 | 0.0456584400313822 | 0.0751438817330815 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.2089702626058581 | 0.1855809323060097 | 0.2328042614854887 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.1818914614249086 | 0.1585254472990426 | 0.2064452923095595 | 9.999000099990002e-05 | 0.001999800019998 | exploratory/post hoc |

Bootstrap pareado por cenas usa as mesmas 10.000 reamostragens para os 20 contrastes fixo − média das AUCs individuais FT. P bilateral do bootstrap centrado no nulo, correção Monte Carlo (1+contagem)/(B+1), Holm em 20 pares. DeLong multivariado: `delong_sensitivity.csv/.tex`, sensibilidade com independência entre imagens, contraste [1,−1/3,−1/3,−1/3]. [Referência](https://pubmed.ncbi.nlm.nih.gov/3203132/).

| fixed | fine_tuning | recall_target | difference | ci_low | ci_high | substitution_descriptive | oracle | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 0.9 | 0.0805687203791468 | 0.0455735663274241 | 0.1306100335249043 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 0.95 | 0.1165086887835703 | 0.0617569865986334 | 0.1819281616393106 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB3 | 0.97 | 0.145734597156398 | 0.0174184845372036 | 0.2397947451521122 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 0.9 | 0.1879936808846761 | 0.1452589896521493 | 0.2414208126797625 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 0.95 | 0.2180094786729858 | 0.1615077428790115 | 0.278833604706384 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_EfficientNetB4 | 0.97 | 0.2247235387045813 | 0.089190163201898 | 0.3126918228726194 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | 0.9 | 0.1141390205371247 | 0.0711970394480176 | 0.1608796168043508 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | 0.95 | 0.1184834123222748 | 0.0675611501431174 | 0.1798301342140828 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_MobileNet | 0.97 | 0.1117693522906793 | -0.0142631487740592 | 0.1984143657462985 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | 0.9 | 0.40521327014218 | 0.3571717051323726 | 0.4710850066934404 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | 0.95 | 0.4324644549763033 | 0.3787135764629443 | 0.4979919678714859 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet101V2 | 0.97 | 0.4028436018957346 | 0.2630926910623569 | 0.4983600739683561 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | 0.9 | 0.3578199052132702 | 0.3093835297602674 | 0.4184163928441037 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | 0.95 | 0.3763823064770932 | 0.3205266666377044 | 0.438317554585937 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| a_resnet101v2_logistic | ft_ResNet50 | 0.97 | 0.3720379146919431 | 0.225520919383285 | 0.4588389786391526 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 0.9 | 0.0888625592417061 | 0.042315369261477 | 0.1413908037723203 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 0.95 | 0.1248025276461296 | 0.0659376675913083 | 0.1955673649135836 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB3 | 0.97 | 0.1729857819905213 | 0.0645678449226432 | 0.2581292255162264 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 0.9 | 0.1962875197472354 | 0.1468230234677603 | 0.2510929668580137 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 0.95 | 0.2263033175355451 | 0.1657184019519101 | 0.293842288585622 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_EfficientNetB4 | 0.97 | 0.2519747235387046 | 0.1503943012116088 | 0.3219580574548706 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 0.9 | 0.122432859399684 | 0.0701965503409547 | 0.169721104404203 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 0.95 | 0.1267772511848341 | 0.0731875223069887 | 0.1925597855193926 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_MobileNet | 0.97 | 0.1390205371248025 | 0.0491517574771844 | 0.2096842009531084 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 0.9 | 0.4135071090047393 | 0.3570260617440246 | 0.4794698358651998 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 0.95 | 0.4407582938388626 | 0.3849878934624697 | 0.5078077345672545 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet101V2 | 0.97 | 0.4300947867298579 | 0.3345321599862967 | 0.5038850158021937 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 0.9 | 0.3661137440758295 | 0.3110563132143758 | 0.4259919848315091 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 0.95 | 0.3846761453396525 | 0.3279997533908754 | 0.4527421307506051 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | ft_ResNet50 | 0.97 | 0.3992890995260664 | 0.2962094336687353 | 0.4633576834989314 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.9 | 0.1208530805687203 | 0.0763708758827198 | 0.1603497190128002 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.95 | 0.1188783570300158 | 0.055286472174649 | 0.1901952891699899 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.97 | 0.164691943127962 | 0.076766801680168 | 0.2438374144602869 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.9 | 0.2282780410742496 | 0.1770323496919948 | 0.2713367658276125 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.95 | 0.2203791469194312 | 0.1552887382497909 | 0.2871928855551767 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.97 | 0.2436808846761453 | 0.1627601951880811 | 0.3045870398495696 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.9 | 0.1544233807266982 | 0.1039501680994607 | 0.1863229425654933 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.95 | 0.1208530805687203 | 0.0618876817457915 | 0.1863358880292518 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.97 | 0.1307266982622432 | 0.0706003287899838 | 0.1886801823074789 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.9 | 0.4454976303317535 | 0.3902611687950034 | 0.4982121573301549 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.95 | 0.4348341232227488 | 0.3680692899228712 | 0.5078635070380354 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.97 | 0.4218009478672986 | 0.3488994187605763 | 0.4869189682020994 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.9 | 0.3981042654028436 | 0.3415379940383689 | 0.4463015618006961 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.95 | 0.3787519747235386 | 0.3117558185594381 | 0.450101880269517 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.97 | 0.3909952606635071 | 0.3042942060347223 | 0.447264093975025 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.9 | 0.1113744075829383 | 0.0754054517529835 | 0.152615144418423 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.95 | 0.1165086887835703 | 0.052386212333691 | 0.1955355401590967 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB3 | 0.97 | 0.1421800947867298 | 0.0689095172201722 | 0.2348211041056103 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.9 | 0.2187993680884676 | 0.1758821081295614 | 0.262806129231765 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.95 | 0.2180094786729858 | 0.1536635323182415 | 0.2948289484975773 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_EfficientNetB4 | 0.97 | 0.221169036334913 | 0.1540562378797673 | 0.2990914974113898 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.9 | 0.1449447077409162 | 0.103602415428324 | 0.1784848800874478 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.95 | 0.1184834123222748 | 0.0584474112969807 | 0.1937016710934195 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_MobileNet | 0.97 | 0.108214849921011 | 0.0598799972869795 | 0.1843052232712061 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.9 | 0.4360189573459715 | 0.3889516544354319 | 0.4908753158125995 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.95 | 0.4324644549763033 | 0.3643948035081874 | 0.5198149376981495 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet101V2 | 0.97 | 0.3992890995260664 | 0.3441240548231271 | 0.4835041851925676 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.9 | 0.3886255924170617 | 0.3409265890352417 | 0.4387840715581991 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.95 | 0.3763823064770932 | 0.3086562202550923 | 0.4606726452263149 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | ft_ResNet50 | 0.97 | 0.3684834123222749 | 0.2987937586732766 | 0.4429178349812719 | supported | external recall-matched thresholds, not deployable | exploratory/post hoc |

Os limiares com recall 0,90/0,95/0,97 são oráculos externos não implantáveis, reestimados em cada bootstrap. Substituição com Δ=3 p.p. é apenas leitura descritiva pós-teste. Ensemble é sensibilidade separada (`ensemble_auc_sensitivity.csv/.tex`); as curvas principais FT são médias verticais de ROC por seed.

| model | seed | partial_auc_mcclish | ci_low | ci_high | focus | oracle | analysis | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | mean | 0.777895568304648 | 0.7444610636594348 | 0.811175391278993 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | mean | 0.792884043688586 | 0.7632340746263389 | 0.8237528959196723 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | mean | 0.7971299548693037 | 0.7656539969146433 | 0.8282134956396564 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | mean | 0.7997998356157663 | 0.7694367482341717 | 0.8293453748967607 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| ft_EfficientNetB3 | mean | 0.7262435135698776 | 0.6993223463194189 | 0.7546550615824864 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| ft_EfficientNetB4 | mean | 0.6799936936065074 | 0.6557224127145013 | 0.7065158303563349 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| ft_MobileNet | mean | 0.732121681135478 | 0.7082844260825119 | 0.7571800764417738 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| ft_ResNet101V2 | mean | 0.5820327532632633 | 0.5663190215931309 | 0.60085022259109 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |
| ft_ResNet50 | mean | 0.6066557440658337 | 0.5883014061756019 | 0.6275833941060461 | obstructed sensitivity [0.90,1.00] | post-test descriptive, not deployable | individual-seed metric mean | exploratory/post hoc |

AUC parcial integra especificidade em sensibilidade de obstruídos [0,90;1,00], corrigida McClish com referência aleatória 0,005 e máximo 0,10, conforme [pROC](https://rdrr.io/cran/pROC/man/auc.html). Valores abaixo da diagonal são preservados na correção. ROC: `results/stage3/roc_analysis/roc.png/.pdf`, com zoom; coordenadas e curvas por seed em CSV/LaTeX.

**1. Limiar, atributos, clusters e erros — exploratory/post hoc**

| model | decision | estimate | ci_low | ci_high | analysis_scope |
| --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | frozen default | 0.7893401015228426 | 0.7695250015126168 | 0.8097622027534418 | exploratory/post hoc |
| a_resnet101v2_logistic | external accuracy oracle, not deployable | 0.8286802030456852 | 0.8109128427038081 | 0.8508771929824561 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | frozen default | 0.809010152284264 | 0.787974587415599 | 0.8297738693467337 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | external accuracy oracle, not deployable | 0.8350253807106599 | 0.8177214383103001 | 0.8567709173387097 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | frozen default | 0.7963197969543148 | 0.7760343810828578 | 0.8165149301700462 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | external accuracy oracle, not deployable | 0.8413705583756346 | 0.8250946902546672 | 0.8616555863216728 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | frozen default | 0.800761421319797 | 0.7809692728965084 | 0.8202738513785374 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | external accuracy oracle, not deployable | 0.8362944162436549 | 0.8212244781488675 | 0.8586426344686394 | exploratory/post hoc |
| ft_EfficientNetB3 | frozen default | 0.7779187817258882 | 0.7588210543048559 | 0.7969697454390353 | exploratory/post hoc |
| ft_EfficientNetB3 | external accuracy oracle, not deployable | 0.7855329949238579 | 0.7689790542437088 | 0.8055379573788192 | exploratory/post hoc |
| ft_EfficientNetB4 | frozen default | 0.7119289340101522 | 0.6925012424895942 | 0.7314933991946841 | exploratory/post hoc |
| ft_EfficientNetB4 | external accuracy oracle, not deployable | 0.7309644670050761 | 0.713337378137316 | 0.7526886760752687 | exploratory/post hoc |
| ft_MobileNet | frozen default | 0.7571912013536379 | 0.7372721020868617 | 0.776937618147448 | exploratory/post hoc |
| ft_MobileNet | external accuracy oracle, not deployable | 0.7707275803722503 | 0.7550072480275429 | 0.7930673636926071 | exploratory/post hoc |
| ft_ResNet101V2 | frozen default | 0.6579949238578681 | 0.6363438770948635 | 0.679225218647968 | exploratory/post hoc |
| ft_ResNet101V2 | external accuracy oracle, not deployable | 0.6681472081218275 | 0.6495668320905964 | 0.6907152887611214 | exploratory/post hoc |
| ft_ResNet50 | frozen default | 0.6605329949238579 | 0.6392426957721024 | 0.6820417528046719 | exploratory/post hoc |
| ft_ResNet50 | external accuracy oracle, not deployable | 0.6715313028764806 | 0.6548990115100513 | 0.6962488165832038 | exploratory/post hoc |

Oráculo maximiza a acurácia nas próprias externas: estimativa de treino descritiva, não implantável. Seus ICs reescolhem o limiar em cada sorteio de cenas; métricas por seed e médias estão no CSV completo. Taxa prevista de obstruídos, distribuição/quantis por classe/domínio e KS: `predicted_obstructed_fraction.csv`, `score_distributions.csv`, `kolmogorov_smirnov.csv`. KS é descritivo e sua inferência usual não corrige correlação de cenas. Platt FT usa exclusivamente 342 OOF VIA por seed, sem ajustar em externas; adaptadores fixos existentes são reutilizados, `via_only_platt.csv/.tex`.

| model | endpoint | estimate | ci_low | ci_high | bootstrap_valid | bootstrap_total | seed | analysis | unrecalibrated_brier | fit_source | platt_intercept | platt_slope | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | accuracy | 0.7893401015228426 | 0.7695250015126168 | 0.8097622027534418 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | accuracy | 0.809010152284264 | 0.787974587415599 | 0.8297738693467337 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.7963197969543148 | 0.7760343810828578 | 0.8165149301700462 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | accuracy | 0.800761421319797 | 0.7809692728965084 | 0.8202738513785374 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| ft_EfficientNetB3 | accuracy | 0.7764382402707276 | 0.7573382957678151 | 0.7954695319264606 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| ft_EfficientNetB4 | accuracy | 0.7034686971235194 | 0.6840999111974594 | 0.7227084251679261 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| ft_MobileNet | accuracy | 0.7561336717428088 | 0.7362353638763213 | 0.7752643184933548 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| ft_ResNet101V2 | accuracy | 0.6571489001692048 | 0.6354163366075201 | 0.6785038634068438 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |
| ft_ResNet50 | accuracy | 0.6607445008460237 | 0.6393574183366121 | 0.6822297239398897 | 10000.0 | 10000.0 | mean | individual-seed metric mean; VIA-only calibration sensitivity | — | — | — | — | exploratory/post hoc |

Atributos: Y médio/p10, saturação HSV, contraste RMS, densidade Canny, variância de Laplaciano, distâncias ao VIA em DINOv2 e quatro extratores. VIA exclui seu próprio vizinho. Quintis externos ficam fixos nos sorteios; a reamostragem por cenas é global antes de restringir subgrupos. Recalls de classes ausentes são não estimáveis (NaN), nunca zero. Em estratos pequenos, os ICs usam apenas sorteios com a classe necessária presente; bootstrap_valid/total registram essa cobertura. Ranking pós-teste: média da amplitude de acurácia entre quintis nos nove modelos; apenas descritivo, sem interpretação causal. Todos os atributos em `attribute_quintiles.csv/.tex`.

A classificação binária de baixa luz usa o limiar VIA pré-fixado; seus dois estratos estão em `low_light_metrics.csv/.tex` e não são divididos artificialmente em quintis.

| attribute | mean_accuracy_range | analysis_scope |
| --- | --- | --- |
| luminance_p10 | 0.1888419203309616 | exploratory/post hoc |
| distance_ResNet101V2 | 0.1772553747237291 | exploratory/post hoc |
| distance_ResNet50 | 0.1717251207405919 | exploratory/post hoc |
| contrast_rms | 0.16638351230475 | exploratory/post hoc |
| distance_dinov2 | 0.1645309162890035 | exploratory/post hoc |
| distance_EfficientNetB3 | 0.1608026551767761 | exploratory/post hoc |
| distance_MobileNet | 0.1498050290596001 | exploratory/post hoc |
| luminance_mean | 0.1421520475669561 | exploratory/post hoc |
| edge_density | 0.1389499103282507 | exploratory/post hoc |
| sharpness_laplacian | 0.1360845072518771 | exploratory/post hoc |
| saturation_mean | 0.0973247307987111 | exploratory/post hoc |

| model | attribute | quintile | images | clear | nonclear | endpoint | estimate | ci_low | ci_high | bootstrap_valid | bootstrap_total | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.8722741433021807 | 0.8353658536585366 | 0.908080962601032 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.9595141700404858 | 0.9327731092436976 | 0.9826105778279692 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.0404858299595142 | 0.0173894221720308 | 0.0672268907563025 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.581081081081081 | 0.4626819029850746 | 0.6941176470588235 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7911392405063291 | 0.7470219947735192 | 0.8358662613981763 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.9536082474226804 | 0.922328839478847 | 0.9809523809523808 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.0463917525773195 | 0.019047619047619 | 0.0776711605211529 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5327868852459017 | 0.4414414414414414 | 0.625 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.691131498470948 | 0.6396085019631531 | 0.742029216283578 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.967741935483871 | 0.9338235294117648 | 0.9927007299270072 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.032258064516129 | 0.0072992700729926 | 0.0661764705882352 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5221674876847291 | 0.4480843409242638 | 0.5951248548199767 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.7919463087248322 | 0.7428545548654245 | 0.8384191805733846 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.9256198347107438 | 0.8728813559322034 | 0.9685051750606644 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.0743801652892561 | 0.0314948249393354 | 0.1271186440677966 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.7005649717514124 | 0.6285714285714286 | 0.7692307692307693 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.802547770700637 | 0.7507503104277298 | 0.8509966329339981 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.9347826086956522 | 0.8536585365853658 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.0652173913043477 | 0.0 | 0.1463414634146341 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.7798507462686567 | 0.7211895910780669 | 0.8352059925093633 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.8575949367088608 | 0.8185053380782918 | 0.8953846153846153 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.9508196721311476 | 0.8888888888888888 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.0491803278688525 | 0.0 | 0.1111111111111111 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.8352941176470589 | 0.7884615384615384 | 0.8791686320754717 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.726984126984127 | 0.6747666408877538 | 0.7777777777777778 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.9649122807017544 | 0.9285714285714286 | 0.9919370967741936 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.0350877192982456 | 0.0080629032258064 | 0.0714285714285714 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.5920398009950248 | 0.5188679245283019 | 0.6635082163263498 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7396825396825397 | 0.6903897371139938 | 0.7875816993464052 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.9867549668874172 | 0.9657093596059114 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.0132450331125827 | 0.0 | 0.0342906403940885 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5121951219512195 | 0.4342838345864661 | 0.5894039735099338 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7746031746031746 | 0.728124291918429 | 0.8198772742705358 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.949748743718593 | 0.9170731707317074 | 0.9783783783783784 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.050251256281407 | 0.0216216216216216 | 0.0829268292682926 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.4741379310344827 | 0.3773584905660377 | 0.5714285714285714 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8476190476190476 | 0.8058245391867173 | 0.8858024691358025 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.9227053140096618 | 0.8842592592592593 | 0.9574493175431552 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.0772946859903381 | 0.0425506824568445 | 0.1157407407407407 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.7037037037037037 | 0.6041666666666666 | 0.7913744004796162 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.8670886075949367 | 0.8260869565217391 | 0.9047619047619048 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.9493670886075948 | 0.8928571428571429 | 0.9891304347826086 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.0506329113924051 | 0.0108695652173913 | 0.1071428571428571 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8396624472573839 | 0.7874975393700787 | 0.8875513682637527 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.7777777777777778 | 0.730766253869969 | 0.8231520574118085 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.976 | 0.946969696969697 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.024 | 0.0 | 0.0530303030303029 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.6473684210526316 | 0.576271186440678 | 0.7164948453608248 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.6984126984126984 | 0.6468840411721068 | 0.7483443708609272 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.9785714285714284 | 0.9513888888888888 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.0214285714285714 | 0.0 | 0.0486111111111111 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.4742857142857143 | 0.3989892447794002 | 0.5513554463554463 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7682539682539683 | 0.7204968944099379 | 0.8151515151515152 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.9405405405405406 | 0.9036129288728518 | 0.9719642507025684 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.0594594594594594 | 0.0280357492974315 | 0.0963870711271481 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5230769230769231 | 0.4354838709677419 | 0.6129032258064516 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.834920634920635 | 0.7943908663087426 | 0.875 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.9310344827586208 | 0.8949711607786589 | 0.9634703196347032 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.0689655172413793 | 0.0365296803652968 | 0.1050288392213409 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.6607142857142857 | 0.5702461215021023 | 0.7474747474747475 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.838006230529595 | 0.7908424431719591 | 0.8819875776397516 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.9473684210526316 | 0.9180311859631148 | 0.9738814081765088 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.0526315789473684 | 0.0261185918234912 | 0.0819688140368852 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.4729729729729729 | 0.3478260869565217 | 0.6027397260273972 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7816455696202531 | 0.7313432835820896 | 0.830620193358379 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.9123711340206184 | 0.8602150537634409 | 0.9575471698113208 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.0876288659793814 | 0.0424528301886792 | 0.1397849462365591 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5737704918032787 | 0.4796747967479675 | 0.6643356643356644 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.7492354740061162 | 0.6997082359327257 | 0.796876201923077 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.9516129032258064 | 0.9098360655737704 | 0.9847328244274808 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.0483870967741935 | 0.0152671755725191 | 0.0901639344262295 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.625615763546798 | 0.554904098791382 | 0.6933980041718024 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.8288590604026845 | 0.7824672476646161 | 0.8710822769927962 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.8760330578512396 | 0.8108108108108109 | 0.9318181818181818 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.1239669421487603 | 0.0681818181818182 | 0.1891891891891891 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.7966101694915254 | 0.7321382547864507 | 0.8553478063196182 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8503184713375797 | 0.803025974025974 | 0.8942307692307693 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.8260869565217391 | 0.6923076923076923 | 0.9393939393939394 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.1739130434782608 | 0.0606060606060605 | 0.3076923076923077 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8544776119402985 | 0.8039196832579186 | 0.9008305375992144 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.8955696202531646 | 0.860066552901024 | 0.9285767274268758 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.8688524590163934 | 0.7777777777777778 | 0.9491525423728814 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.1311475409836066 | 0.0508474576271186 | 0.2222222222222222 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.9019607843137256 | 0.8624988382899629 | 0.9372549019607844 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.7904761904761904 | 0.742671009771987 | 0.8357782203352738 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.9298245614035088 | 0.8782555673382821 | 0.9724770642201837 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.0701754385964912 | 0.0275229357798164 | 0.1217444326617178 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.7114427860696517 | 0.6448585363362442 | 0.7747265122265122 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7714285714285715 | 0.7222222222222222 | 0.819078947368421 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.9205298013245032 | 0.8705035971223022 | 0.9645390070921984 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.0794701986754966 | 0.0354609929078014 | 0.1294964028776978 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.6341463414634146 | 0.5578231292517006 | 0.7094972067039106 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7619047619047619 | 0.7111783988957902 | 0.808641975308642 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.9095477386934674 | 0.8627450980392157 | 0.9513513513513514 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.0904522613065326 | 0.0486486486486485 | 0.1372549019607842 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.5086206896551724 | 0.4095238095238095 | 0.6055045871559633 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8253968253968254 | 0.7770028576441982 | 0.8700022658610271 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.9371980676328504 | 0.9010974702380952 | 0.96875 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.0628019323671497 | 0.03125 | 0.0989025297619047 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.6111111111111112 | 0.4957264957264957 | 0.71875 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.879746835443038 | 0.8411214953271028 | 0.9155844155844156 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.8987341772151899 | 0.8266666666666667 | 0.961038961038961 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.1012658227848101 | 0.0389610389610389 | 0.1733333333333333 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8734177215189873 | 0.8271604938271605 | 0.9155569620253164 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.8476190476190476 | 0.8061538461538461 | 0.8866279069767442 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.936 | 0.890625 | 0.9754098360655736 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.0639999999999999 | 0.0245901639344262 | 0.109375 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.7894736842105263 | 0.7282038834951456 | 0.8474602108309218 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.7428571428571429 | 0.6923076923076923 | 0.791669952681388 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.9428571428571428 | 0.900709219858156 | 0.9782608695652174 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.0571428571428571 | 0.0217391304347825 | 0.099290780141844 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.5828571428571429 | 0.5058823529411764 | 0.6589628218602207 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7746031746031746 | 0.727563854489164 | 0.8206912817176317 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.918918918918919 | 0.8770053475935828 | 0.955976570599212 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.081081081081081 | 0.0440234294007878 | 0.1229946524064171 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5692307692307692 | 0.4769230769230769 | 0.656934306569343 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.8 | 0.75 | 0.8473898842121733 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.9014778325123152 | 0.8585858585858586 | 0.940300219088046 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.0985221674876847 | 0.0596997809119539 | 0.1414141414141414 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.6160714285714286 | 0.5132743362831859 | 0.7169858967028777 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.8598130841121495 | 0.8150470219435737 | 0.9015396825396824 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.97165991902834 | 0.9486144210648686 | 0.9914893617021276 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.0283400809716599 | 0.0085106382978723 | 0.0513855789351313 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.4864864864864865 | 0.3623188405797101 | 0.6153846153846154 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7879746835443038 | 0.7421383647798742 | 0.832765064622615 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.963917525773196 | 0.935483870967742 | 0.989071038251366 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.0360824742268041 | 0.0109289617486338 | 0.0645161290322581 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5081967213114754 | 0.4117647058823529 | 0.6033057851239669 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.6972477064220184 | 0.644775137470542 | 0.7485384342358786 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.9435483870967742 | 0.9008264462809916 | 0.9821428571428572 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.0564516129032257 | 0.0178571428571429 | 0.0991735537190082 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5467980295566502 | 0.4714285714285714 | 0.6208874458874458 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.8120805369127517 | 0.7651001384894003 | 0.8566433566433567 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.9090909090909092 | 0.8521061554512258 | 0.9572649572649572 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.0909090909090909 | 0.0427350427350426 | 0.147893844548774 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.7457627118644068 | 0.6779661016949152 | 0.8089943081017149 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8280254777070064 | 0.7781350482315113 | 0.8733344155844155 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.8478260869565217 | 0.735279111644658 | 0.9444444444444444 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.1521739130434782 | 0.0555555555555555 | 0.2647208883553418 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8246268656716418 | 0.7700729927007299 | 0.8754455113140401 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.9082278481012658 | 0.8749904434250765 | 0.9386503067484664 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.918032786885246 | 0.8409090909090909 | 0.9824561403508772 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.081967213114754 | 0.0175438596491228 | 0.159090909090909 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.9058823529411764 | 0.8677669871691803 | 0.9399145422284232 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.7047619047619048 | 0.6510067114093959 | 0.756185401943463 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.9298245614035088 | 0.880274647887324 | 0.9722222222222222 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.0701754385964912 | 0.0277777777777777 | 0.1197253521126759 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.5771144278606966 | 0.5047158790336802 | 0.6467734855136084 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7523809523809524 | 0.7037037037037037 | 0.8 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.9801324503311258 | 0.9545454545454546 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.0198675496688741 | 0.0 | 0.0454545454545454 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5426829268292683 | 0.4649636673544451 | 0.6222295805739514 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7587301587301587 | 0.710608817058724 | 0.8054732353934481 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.9346733668341708 | 0.8990384615384616 | 0.9668508287292816 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.0653266331658291 | 0.0331491712707182 | 0.1009615384615384 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.4568965517241379 | 0.3611039518900343 | 0.5545473644003056 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8571428571428571 | 0.8114478114478114 | 0.9 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.9516908212560388 | 0.9205580926083264 | 0.9790575916230366 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.0483091787439613 | 0.0209424083769633 | 0.0794419073916736 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.6759259259259259 | 0.5652173913043478 | 0.7788461538461539 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.8955696202531646 | 0.8580645161290322 | 0.929768690349947 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.9493670886075948 | 0.8928571428571429 | 0.9891304347826086 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.0506329113924051 | 0.0108695652173913 | 0.1071428571428571 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8776371308016878 | 0.8312236286919831 | 0.9198113207547168 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.7746031746031746 | 0.7264944275582573 | 0.819620253164557 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.944 | 0.9013979919118672 | 0.9809523809523808 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.056 | 0.019047619047619 | 0.0986020080881325 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.6631578947368421 | 0.5923859910128734 | 0.732242507037589 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.7015873015873015 | 0.6506410256410257 | 0.7526214388194301 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.95 | 0.9097211021505376 | 0.983336776859504 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.05 | 0.0166632231404959 | 0.0902788978494623 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.5028571428571429 | 0.4259259259259259 | 0.5801155068273551 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7714285714285715 | 0.72413513210927 | 0.8190184049079755 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.9513513513513514 | 0.9181286549707602 | 0.9797979797979798 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.0486486486486485 | 0.0202020202020202 | 0.0818713450292397 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5153846153846153 | 0.4230691056910569 | 0.6076999999999998 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.8380952380952381 | 0.7916635196374622 | 0.8821598336304218 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.9408866995073892 | 0.9061032863849764 | 0.971291866028708 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.0591133004926108 | 0.0287081339712919 | 0.0938967136150235 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.6517857142857143 | 0.5483870967741935 | 0.75 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.8785046728971962 | 0.841771619145037 | 0.9136916647649476 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.9676113360323888 | 0.944 | 0.9878542510121456 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.0323886639676113 | 0.0121457489878542 | 0.056 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.581081081081081 | 0.4666438356164384 | 0.6923076923076923 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7911392405063291 | 0.7452830188679245 | 0.8350886625165064 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.9587628865979382 | 0.9289014895057548 | 0.9845380650277558 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.0412371134020618 | 0.0154619349722441 | 0.0710985104942449 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5245901639344263 | 0.4274176316515026 | 0.6185588852000699 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.6880733944954128 | 0.6354515050167224 | 0.7406366741381151 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.9435483870967742 | 0.8939393939393939 | 0.984251968503937 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.0564516129032257 | 0.0157480314960629 | 0.106060606060606 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5320197044334976 | 0.4560418208734271 | 0.6055045871559633 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.8154362416107382 | 0.7683791433239963 | 0.859107739218818 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.9256198347107438 | 0.8739495798319328 | 0.9689922480620154 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.0743801652892561 | 0.0310077519379845 | 0.1260504201680672 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.7401129943502824 | 0.6721289441212762 | 0.8044692737430168 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8343949044585988 | 0.7852564102564102 | 0.8794818232327083 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.8695652173913043 | 0.7631578947368421 | 0.9574468085106383 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.1304347826086956 | 0.0425531914893616 | 0.2368421052631578 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8283582089552238 | 0.7740860711209548 | 0.8788935093953171 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.9082278481012658 | 0.875 | 0.9384615384615383 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.9344262295081968 | 0.865664466130884 | 0.9857142857142858 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.0655737704918032 | 0.0142857142857142 | 0.1343355338691159 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.9019607843137256 | 0.863449501129518 | 0.937007874015748 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.7047619047619048 | 0.6520117540687161 | 0.7553516819571865 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.9298245614035088 | 0.880274647887324 | 0.9722222222222222 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.0701754385964912 | 0.0277777777777777 | 0.1197253521126759 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.5771144278606966 | 0.5053749142072752 | 0.6455109126984125 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7587301587301587 | 0.7099992447129909 | 0.8060621521335807 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.9933774834437086 | 0.9779411764705882 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.0066225165562914 | 0.0 | 0.0220588235294117 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5426829268292683 | 0.4647058823529412 | 0.6227570573139435 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.765079365079365 | 0.71672212741752 | 0.8123212976539588 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.9346733668341708 | 0.8960396039603961 | 0.9692307692307692 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.0653266331658291 | 0.0307692307692307 | 0.1039603960396039 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.4741379310344827 | 0.3771893874029335 | 0.5724651510122801 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8666666666666667 | 0.828123629385965 | 0.90282131661442 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.9420289855072465 | 0.9086287593471972 | 0.9714285714285714 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.0579710144927536 | 0.0285714285714285 | 0.0913712406528028 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.7222222222222222 | 0.6293103448275862 | 0.8062015503875969 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.9018987341772152 | 0.8664495114006515 | 0.9343762807377048 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.9620253164556962 | 0.912280701754386 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.0379746835443037 | 0.0 | 0.087719298245614 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8818565400843882 | 0.8368200836820083 | 0.9224489795918368 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.7746031746031746 | 0.7269938650306749 | 0.8201892744479495 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.944 | 0.9013979919118672 | 0.9809523809523808 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.056 | 0.019047619047619 | 0.0986020080881325 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.6631578947368421 | 0.5913978494623656 | 0.732620320855615 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.707936507936508 | 0.6567644451012264 | 0.7586206896551724 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.9571428571428572 | 0.92 | 0.9861111111111112 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.0428571428571428 | 0.0138888888888888 | 0.0799999999999999 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.5085714285714286 | 0.4315771449170872 | 0.5868296016662327 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7619047619047619 | 0.7142857142857143 | 0.8093787202380951 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.945945945945946 | 0.9112426035502958 | 0.9763313609467456 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.054054054054054 | 0.0236686390532544 | 0.0887573964497041 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5 | 0.4077651232262883 | 0.5934959349593496 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.8571428571428571 | 0.8170731707317073 | 0.8950617283950617 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.9408866995073892 | 0.9032258064516128 | 0.9731206797235022 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.0591133004926108 | 0.0268793202764976 | 0.0967741935483871 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.7053571428571429 | 0.616822429906542 | 0.7868852459016393 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.7902388369678089 | 0.7502609962527662 | 0.8287193515704154 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.8178137651821862 | 0.7756154447133796 | 0.8583697709940087 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.1821862348178138 | 0.1416302290059913 | 0.2243845552866203 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.6981981981981983 | 0.6027397260273973 | 0.7901284348864993 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7869198312236287 | 0.7461289054654007 | 0.8274769680680223 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.8642611683848798 | 0.8225806451612904 | 0.9022456395348836 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.1357388316151202 | 0.0977543604651163 | 0.1774193548387096 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.6639344262295083 | 0.5819660553857553 | 0.7450980392156863 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.6911314984709479 | 0.6440851492371618 | 0.737074444051548 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.8844086021505376 | 0.8318035668352912 | 0.9333333333333332 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.1155913978494623 | 0.0666666666666666 | 0.1681964331647089 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5730706075533663 | 0.5058626465661642 | 0.63915857605178 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.7583892617449665 | 0.7129612733171127 | 0.8006541467938528 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.8732782369146005 | 0.8209366391184574 | 0.92 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.1267217630853994 | 0.08 | 0.1790633608815426 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.6798493408662901 | 0.6178848842628553 | 0.7387387387387387 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8651804670912951 | 0.8283482667203597 | 0.8988289257419342 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.8115942028985508 | 0.7210844803037493 | 0.8916666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.1884057971014492 | 0.1083333333333333 | 0.2789155196962505 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8743781094527363 | 0.8333333333333334 | 0.9102053898482472 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.8934599156118144 | 0.8655362361192285 | 0.920290465036842 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.8688524590163934 | 0.7959183673469389 | 0.9333333333333332 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.1311475409836065 | 0.0666666666666666 | 0.2040816326530612 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.8993464052287582 | 0.8684546615581098 | 0.9290780141843972 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.6952380952380951 | 0.6480244883040936 | 0.74120571404142 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.8421052631578947 | 0.7864056924736537 | 0.8927536231884058 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.1578947368421052 | 0.1072463768115942 | 0.2135943075263463 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.6119402985074626 | 0.5482214702011656 | 0.6743128073958202 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7312169312169313 | 0.6866265708020579 | 0.7737714676034348 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.891832229580574 | 0.8530701754385964 | 0.9271542611074836 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.108167770419426 | 0.0728457388925162 | 0.1469298245614035 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5833333333333334 | 0.5140562248995985 | 0.6516007532956686 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7544973544973544 | 0.7123933895373519 | 0.7967219778755165 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.845896147403685 | 0.8007443729252692 | 0.8899680360929106 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.1541038525963149 | 0.1100319639070894 | 0.1992556270747307 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.5977011494252874 | 0.5151515151515151 | 0.6782608695652174 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8148148148148149 | 0.776073223929546 | 0.8513520686291387 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.8228663446054751 | 0.778294395846668 | 0.8650416182733256 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.1771336553945249 | 0.1349583817266744 | 0.2217056041533319 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.7993827160493828 | 0.7239057239057239 | 0.8666666666666667 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.8987341772151899 | 0.8695163905921762 | 0.9262723160795592 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.919831223628692 | 0.8647342995169082 | 0.9636981632945903 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.080168776371308 | 0.0363018367054097 | 0.1352657004830917 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8917018284106892 | 0.8567375886524823 | 0.9241707856179106 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.7671957671957671 | 0.7258229935964021 | 0.8067333692953519 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.8586666666666667 | 0.8095238095238096 | 0.9043484200522688 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.1413333333333333 | 0.095651579947731 | 0.1904761904761904 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.7070175438596492 | 0.6470561880666245 | 0.7636070168808264 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.674074074074074 | 0.625514403292181 | 0.7218815445832409 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.888095238095238 | 0.8482582008655907 | 0.9250660038197956 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.1119047619047619 | 0.0749339961802044 | 0.1517417991344092 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.5028571428571429 | 0.4312477355072463 | 0.5746542744252874 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.744973544973545 | 0.7022988505747128 | 0.7865863368028585 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.8378378378378378 | 0.7921108823305116 | 0.879919254658385 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.1621621621621621 | 0.1200807453416149 | 0.2078891176694882 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.6128205128205129 | 0.5356080489938758 | 0.6853333333333333 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.8042328042328043 | 0.7672394493051982 | 0.8403451995685005 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.8029556650246304 | 0.7564060205580029 | 0.8473699485030985 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.1970443349753694 | 0.1526300514969014 | 0.243593979441997 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.8065476190476191 | 0.7410708136560409 | 0.8666666666666667 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.8068535825545172 | 0.7656238408753709 | 0.8462381859874191 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.8717948717948718 | 0.8349032253576001 | 0.906610632183908 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.1282051282051282 | 0.0933893678160919 | 0.1650967746423999 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.5900900900900902 | 0.4838709677419355 | 0.6933333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.75 | 0.704882674274593 | 0.7945367436592123 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.915807560137457 | 0.8833333333333333 | 0.9465932253392696 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.0841924398625429 | 0.0534067746607305 | 0.1166666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.4863387978142077 | 0.3972222222222222 | 0.5732093127537052 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.5891946992864424 | 0.5379921767407435 | 0.6401277405148479 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.9408602150537634 | 0.9054373522458627 | 0.9716312056737588 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.0591397849462365 | 0.0283687943262411 | 0.0945626477541371 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.3743842364532019 | 0.3096536850572799 | 0.4385964912280702 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.6554809843400448 | 0.6048565121412803 | 0.7049954344703222 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.9366391184573004 | 0.9009009009009008 | 0.9666666666666668 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.0633608815426997 | 0.0333333333333333 | 0.099099099099099 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.4632768361581921 | 0.3983739837398374 | 0.5286224312086379 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.7579617834394905 | 0.7146549891770667 | 0.7986128300330033 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.8768115942028984 | 0.7948717948717948 | 0.950354609929078 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.1231884057971014 | 0.0496453900709219 | 0.2051282051282051 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.7375621890547263 | 0.6894208978772232 | 0.7833349546044098 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.7141350210970464 | 0.6677630680184067 | 0.7582139759916016 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.918032786885246 | 0.8571428571428571 | 0.9682539682539684 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.081967213114754 | 0.0317460317460317 | 0.1428571428571428 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.665359477124183 | 0.6106960315609452 | 0.717577540106952 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.6137566137566138 | 0.563655884502924 | 0.6624737945492662 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.8859649122807017 | 0.8373983739837398 | 0.9303030303030304 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.1140350877192982 | 0.0696969696969697 | 0.1626016260162601 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.4593698175787728 | 0.3973046527450672 | 0.5216666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.6793650793650793 | 0.6306864006864006 | 0.7263700494488046 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.93598233995585 | 0.9037995758492712 | 0.9644444444444444 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.0640176600441501 | 0.0355555555555555 | 0.0962004241507286 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.4430894308943089 | 0.3706463312806596 | 0.5147067160197412 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7333333333333334 | 0.6894629087763853 | 0.7760420697884417 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.8877721943048575 | 0.8481670544890441 | 0.9244003182503004 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.1122278056951423 | 0.0755996817496995 | 0.1518329455109559 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.4683908045977011 | 0.3879781420765027 | 0.5482866043613707 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.8190476190476191 | 0.7805636142967478 | 0.8550724637681159 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.9098228663446054 | 0.8767772511848341 | 0.9405405405405406 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.0901771336553945 | 0.0594594594594594 | 0.1232227488151658 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.6450617283950617 | 0.559748427672956 | 0.7243622448979591 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.7310126582278481 | 0.6843839152787252 | 0.774835556835269 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.940928270042194 | 0.893719806763285 | 0.9786324786324788 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.0590717299578058 | 0.0213675213675213 | 0.1062801932367149 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.6610407876230661 | 0.6043472729284213 | 0.7148602922327608 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.6783068783068783 | 0.6318215253581108 | 0.7243368781012491 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.8933333333333334 | 0.8482384823848239 | 0.9328362682987938 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.1066666666666666 | 0.0671637317012061 | 0.1517615176151761 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.5368421052631579 | 0.4700854700854701 | 0.6024972576443165 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.6328042328042328 | 0.5827953428754541 | 0.6819607260165363 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.926190476190476 | 0.8888888888888888 | 0.9591836734693876 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.0738095238095238 | 0.0408163265306122 | 0.1111111111111111 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.3980952380952381 | 0.3298429319371728 | 0.4663209219858155 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7248677248677248 | 0.6798780487804877 | 0.7695769879670301 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.9135135135135136 | 0.8785976080246913 | 0.94585448392555 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.0864864864864864 | 0.05414551607445 | 0.1214023919753086 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.4564102564102564 | 0.3796260683760683 | 0.531328320802005 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.7925925925925926 | 0.7524108099831572 | 0.8299098327779836 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.8801313628899835 | 0.8406416857360792 | 0.915841584158416 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.1198686371100164 | 0.0841584158415841 | 0.1593583142639206 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.6339285714285714 | 0.556574240934906 | 0.70798951459628 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.7601246105919003 | 0.7201421820872431 | 0.7978948972483456 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.7989203778677463 | 0.7586193468686219 | 0.8370130215328139 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.2010796221322537 | 0.1629869784671859 | 0.2413806531313781 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.6306306306306306 | 0.5277777777777778 | 0.7313467385295743 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.7521097046413502 | 0.7087150537634408 | 0.7967967967967967 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.865979381443299 | 0.825320298573975 | 0.9052631578947368 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.134020618556701 | 0.0947368421052631 | 0.1746797014260249 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5710382513661202 | 0.4850943453464384 | 0.6547619047619048 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.6605504587155964 | 0.6118006223577689 | 0.707463194232917 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.8844086021505376 | 0.8358190929410917 | 0.9276755765199162 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.1155913978494623 | 0.0723244234800838 | 0.1641809070589083 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5238095238095237 | 0.4570446735395189 | 0.5895121082621082 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.7527964205816554 | 0.7067088744588745 | 0.796910750623782 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.8319559228650139 | 0.7614610434068191 | 0.894308943089431 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.1680440771349862 | 0.105691056910569 | 0.2385389565931808 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.6986817325800377 | 0.6395833333333334 | 0.7547907924344707 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8641188959660298 | 0.8260869565217391 | 0.8982106824832843 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.7246376811594203 | 0.6222222222222222 | 0.8205128205128206 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.2753623188405797 | 0.1794871794871795 | 0.3777777777777777 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8880597014925374 | 0.8478513356562137 | 0.9234765367523547 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.8565400843881857 | 0.8213943859774403 | 0.8892581216972507 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.8852459016393442 | 0.8121212121212121 | 0.9487179487179486 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.1147540983606557 | 0.0512820512820512 | 0.1878787878787878 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.8496732026143791 | 0.8084275675927219 | 0.8876543209876543 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.6962962962962962 | 0.6480329560313829 | 0.7429955586722767 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.8304093567251462 | 0.7627627627627628 | 0.8906666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.1695906432748538 | 0.1093333333333333 | 0.2372372372372372 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.6202321724709785 | 0.5566666666666666 | 0.6829631642512077 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.7301587301587302 | 0.685356652006597 | 0.7733333333333333 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.8852097130242825 | 0.8447488584474886 | 0.9216870839060942 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.1147902869757174 | 0.0783129160939058 | 0.1552511415525114 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5873983739837398 | 0.5153371294925135 | 0.6563178752429218 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7312169312169313 | 0.6870310825294749 | 0.7733740675896859 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.812395309882747 | 0.7669867543813111 | 0.8568510164802117 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.1876046901172529 | 0.1431489835197883 | 0.2330132456186888 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.5919540229885057 | 0.5071225071225071 | 0.6754405371836671 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.7714285714285714 | 0.73128125 | 0.8095238095238096 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.7971014492753623 | 0.7526315789473684 | 0.839379820777835 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.2028985507246377 | 0.1606201792221649 | 0.2473684210526315 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.7222222222222223 | 0.644257029734971 | 0.7948717948717948 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.8702531645569619 | 0.8342907818689527 | 0.9030883919062832 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.919831223628692 | 0.8630124302384575 | 0.9687534722222222 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.080168776371308 | 0.0312465277777777 | 0.1369875697615423 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.8537271448663853 | 0.8095238095238096 | 0.8934426229508197 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.7608465608465608 | 0.7199548179700546 | 0.8001873304526858 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.8586666666666666 | 0.8057971014492754 | 0.9054736161711128 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.1413333333333333 | 0.094526383828887 | 0.1942028985507246 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.6964912280701755 | 0.636751497005988 | 0.7526935216391161 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.6603174603174603 | 0.6111111111111112 | 0.7081853450790502 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.8642857142857143 | 0.8166638702460851 | 0.907859970047069 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.1357142857142856 | 0.0921400299529311 | 0.1833361297539149 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.4971428571428571 | 0.4261597382048331 | 0.5663295873905118 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.7238095238095239 | 0.6803162299222923 | 0.7657344276094276 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.8144144144144144 | 0.7689594356261024 | 0.8564102564102564 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.1855855855855855 | 0.1435897435897436 | 0.2310405643738977 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5948717948717949 | 0.5130424504060027 | 0.6720901084010841 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.7703703703703703 | 0.7319092010934035 | 0.8085368530713993 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.7750410509031198 | 0.7277770506108203 | 0.8225125680770841 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.2249589490968801 | 0.1774874319229158 | 0.2722229493891797 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.7619047619047619 | 0.6955100574712645 | 0.8247162965127605 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.5887850467289719 | 0.5454545454545454 | 0.6317475879240585 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.5560053981106612 | 0.5058992712129099 | 0.6052305357289358 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.4439946018893387 | 0.3947694642710641 | 0.49410072878709 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.6981981981981983 | 0.6104417670682731 | 0.7843137254901961 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.6392405063291139 | 0.5930470347648262 | 0.685249862579623 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.6804123711340205 | 0.6236559139784946 | 0.734885611646814 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.3195876288659793 | 0.2651143883531858 | 0.3763440860215053 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.5737704918032787 | 0.4945332422586521 | 0.6520499618611746 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.6116207951070337 | 0.5618506541711951 | 0.6608703174364374 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.7473118279569894 | 0.6694438705234159 | 0.8215229725072245 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.2526881720430107 | 0.1784770274927755 | 0.330556129476584 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.5287356321839081 | 0.4626858875023788 | 0.5947712418300654 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.639821029082774 | 0.5938967136150235 | 0.6830414150129422 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.7107438016528925 | 0.6519157653228448 | 0.7686579855822926 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.2892561983471074 | 0.2313420144177073 | 0.348084234677155 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.5913370998116761 | 0.528406410806175 | 0.6528599605522682 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.8131634819532909 | 0.7734791583914105 | 0.8490746996996996 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.536231884057971 | 0.4166666666666667 | 0.6507936507936508 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.463768115942029 | 0.3492063492063492 | 0.5833333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8606965174129354 | 0.8239235396171737 | 0.8938775510204081 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.7964135021097046 | 0.7580063344412733 | 0.8327975132606693 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.7540983606557377 | 0.6717152872003618 | 0.8309178743961353 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.2459016393442623 | 0.1690821256038647 | 0.3282847127996381 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.8065359477124182 | 0.7627887268427277 | 0.8472788236367507 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.6095238095238096 | 0.5627082981399103 | 0.65637982977289 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.6578947368421053 | 0.5841269841269842 | 0.7313958816622275 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.3421052631578947 | 0.2686041183377724 | 0.4158730158730159 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.582089552238806 | 0.5198757834499046 | 0.6424876723986883 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.6148148148148148 | 0.567179104477612 | 0.6612200435729847 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.7152317880794702 | 0.6562483621593291 | 0.7716921320688443 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.2847682119205298 | 0.2283078679311556 | 0.3437516378406708 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.5223577235772358 | 0.4514265010351967 | 0.5919548081150682 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.617989417989418 | 0.5720393211047861 | 0.6634416331718712 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.6164154103852596 | 0.5594054664087099 | 0.6727842904066146 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.3835845896147403 | 0.3272157095933854 | 0.4405945335912901 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.6206896551724138 | 0.5426943229082267 | 0.6979215801886791 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.6507936507936508 | 0.6049362276183469 | 0.6958336840628507 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.5845410628019323 | 0.5295032367013791 | 0.6402651776805587 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.4154589371980676 | 0.3597348223194412 | 0.4704967632986209 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.7777777777777777 | 0.7023780628756355 | 0.8469417767106843 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.7869198312236287 | 0.75 | 0.8225108225108224 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.7468354430379747 | 0.6722199453551913 | 0.8162393162393161 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.2531645569620253 | 0.1837606837606837 | 0.3277800546448087 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.80028129395218 | 0.7565093761891172 | 0.8410498710736052 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.6402116402116402 | 0.5947478618540263 | 0.68497195565283 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.6373333333333333 | 0.5727688860435339 | 0.7016129032258065 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.3626666666666667 | 0.2983870967741935 | 0.427231113956466 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.6421052631578947 | 0.5794859813084112 | 0.7022682616736015 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.5947089947089946 | 0.5448341399475701 | 0.6439399802983394 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.6785714285714285 | 0.6106859537708214 | 0.7440011904761905 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.3214285714285714 | 0.2559988095238095 | 0.3893140462291787 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.5276190476190477 | 0.458539551607445 | 0.5965909090909091 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.617989417989418 | 0.5725575098460436 | 0.6634204370897919 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.6324324324324325 | 0.5747747747747748 | 0.6885245901639344 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.3675675675675675 | 0.3114754098360656 | 0.4252252252252251 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5974358974358974 | 0.5219611767696022 | 0.6717181355540135 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.6497354497354498 | 0.6054416646176543 | 0.692923184679429 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.6009852216748769 | 0.5449975266362254 | 0.6551738224025242 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.3990147783251231 | 0.3448261775974757 | 0.4550024733637746 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.7380952380952381 | 0.6666666666666666 | 0.8078078078078077 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 1 | 321 | 74 | 247 | accuracy | 0.6645898234683281 | 0.6180344478216818 | 0.7102918105081086 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_recall | 0.6909581646423751 | 0.6413497890295359 | 0.7407442348008386 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 1 | 321 | 74 | 247 | obstructed_false_negative_rate | 0.3090418353576248 | 0.2592557651991614 | 0.3586502109704641 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 1 | 321 | 74 | 247 | clear_recall | 0.5765765765765766 | 0.4687474216171617 | 0.6807511737089201 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 2 | 316 | 122 | 194 | accuracy | 0.669831223628692 | 0.6213221018694216 | 0.7188225623582766 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_recall | 0.8487972508591065 | 0.8045435500253936 | 0.8905730380730381 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 2 | 316 | 122 | 194 | obstructed_false_negative_rate | 0.1512027491408935 | 0.1094269619269618 | 0.1954564499746063 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 2 | 316 | 122 | 194 | clear_recall | 0.3852459016393442 | 0.3050847457627119 | 0.4672364672364672 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 3 | 327 | 203 | 124 | accuracy | 0.5514780835881753 | 0.5032446983396349 | 0.6 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_recall | 0.836021505376344 | 0.783712780679434 | 0.8861788617886178 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 3 | 327 | 203 | 124 | obstructed_false_negative_rate | 0.1639784946236559 | 0.1138211382113821 | 0.216287219320566 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 3 | 327 | 203 | 124 | clear_recall | 0.3776683087027915 | 0.3170346083788706 | 0.4391799549104792 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 4 | 298 | 177 | 121 | accuracy | 0.6230425055928411 | 0.5762691686844228 | 0.6666666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_recall | 0.7933884297520661 | 0.7418278572690338 | 0.842948717948718 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 4 | 298 | 177 | 121 | obstructed_false_negative_rate | 0.2066115702479338 | 0.157051282051282 | 0.2581721427309663 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 4 | 298 | 177 | 121 | clear_recall | 0.5065913370998117 | 0.4438202247191011 | 0.567010776184073 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 5 | 314 | 268 | 46 | accuracy | 0.7961783439490446 | 0.7533304093567251 | 0.8360367744793974 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_recall | 0.6159420289855072 | 0.4841269841269842 | 0.7451069518716577 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 5 | 314 | 268 | 46 | obstructed_false_negative_rate | 0.3840579710144928 | 0.2548930481283422 | 0.5158730158730158 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | luminance_p10 | 5 | 314 | 268 | 46 | clear_recall | 0.8271144278606966 | 0.7848596905908702 | 0.8664172041266107 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 1 | 316 | 255 | 61 | accuracy | 0.7099156118143459 | 0.6647496863662563 | 0.7536669343942817 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_recall | 0.8579234972677595 | 0.7857061317677756 | 0.9245283018867924 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 1 | 316 | 255 | 61 | obstructed_false_negative_rate | 0.1420765027322404 | 0.0754716981132075 | 0.2142938682322243 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 1 | 316 | 255 | 61 | clear_recall | 0.6745098039215686 | 0.6205781426737669 | 0.7246927609427609 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 2 | 315 | 201 | 114 | accuracy | 0.5862433862433863 | 0.5393580724215152 | 0.634379078815261 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_recall | 0.7865497076023392 | 0.7249190938511326 | 0.8433333333333333 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 2 | 315 | 201 | 114 | obstructed_false_negative_rate | 0.2134502923976608 | 0.1566666666666666 | 0.2750809061488673 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 2 | 315 | 201 | 114 | clear_recall | 0.4726368159203979 | 0.4126984126984127 | 0.5330118161639901 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 3 | 315 | 164 | 151 | accuracy | 0.6169312169312169 | 0.5692620272808951 | 0.6644810528947462 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_recall | 0.8123620309050773 | 0.7612284323065525 | 0.8604178779069767 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 3 | 315 | 164 | 151 | obstructed_false_negative_rate | 0.1876379690949227 | 0.1395821220930232 | 0.2387715676934476 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 3 | 315 | 164 | 151 | clear_recall | 0.4369918699186992 | 0.3691965423498429 | 0.5046302762259466 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 4 | 315 | 116 | 199 | accuracy | 0.7005291005291006 | 0.6552768258727779 | 0.744346870079809 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_recall | 0.778894472361809 | 0.7298835450695705 | 0.8260886901047084 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 4 | 315 | 116 | 199 | obstructed_false_negative_rate | 0.2211055276381909 | 0.1739113098952915 | 0.2701164549304295 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 4 | 315 | 116 | 199 | clear_recall | 0.5660919540229884 | 0.48 | 0.6504867418310889 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 5 | 315 | 108 | 207 | accuracy | 0.6888888888888888 | 0.6405895691609977 | 0.7374385382059799 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_recall | 0.6940418679549114 | 0.6398104265402843 | 0.7467963836477987 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 5 | 315 | 108 | 207 | obstructed_false_negative_rate | 0.3059581320450886 | 0.2532036163522013 | 0.3601895734597156 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet50 | 5 | 315 | 108 | 207 | clear_recall | 0.6790123456790124 | 0.5781710914454278 | 0.7710858358808369 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | accuracy | 0.7141350210970464 | 0.6698924731182796 | 0.7551916837431695 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_recall | 0.8481012658227849 | 0.7851827485380117 | 0.9047619047619048 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | obstructed_false_negative_rate | 0.1518987341772151 | 0.0952380952380952 | 0.2148172514619882 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 1 | 316 | 237 | 79 | clear_recall | 0.669479606188467 | 0.6160827195664258 | 0.7193677296938166 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | accuracy | 0.6486772486772486 | 0.6026821559817765 | 0.693568137515506 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_recall | 0.7840000000000001 | 0.7222222222222222 | 0.8421052631578947 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | obstructed_false_negative_rate | 0.2159999999999999 | 0.1578947368421052 | 0.2777777777777777 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 2 | 315 | 190 | 125 | clear_recall | 0.5596491228070175 | 0.4962110327412052 | 0.6215791878835358 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | accuracy | 0.598941798941799 | 0.5506329113924051 | 0.6463699494949494 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_recall | 0.8238095238095239 | 0.7733310657596373 | 0.8703703703703702 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | obstructed_false_negative_rate | 0.1761904761904762 | 0.1296296296296296 | 0.2266689342403628 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 3 | 315 | 175 | 140 | clear_recall | 0.419047619047619 | 0.3499976979742173 | 0.4886389230139229 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | accuracy | 0.653968253968254 | 0.6080808080808081 | 0.7006374664767012 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_recall | 0.7513513513513513 | 0.6969696969696969 | 0.8058608058608058 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | obstructed_false_negative_rate | 0.2486486486486486 | 0.1941391941391941 | 0.303030303030303 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 4 | 315 | 130 | 185 | clear_recall | 0.5153846153846154 | 0.4371571811526962 | 0.5925988068605518 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | accuracy | 0.6867724867724867 | 0.6387094970051237 | 0.7333333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_recall | 0.7093596059113301 | 0.6552176527946817 | 0.7616487455197133 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | obstructed_false_negative_rate | 0.29064039408867 | 0.2383512544802867 | 0.3447823472053182 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | distance_ResNet101V2 | 5 | 315 | 112 | 203 | clear_recall | 0.6458333333333334 | 0.5519125683060109 | 0.7345146400087403 | 10000 | 10000 | exploratory/post hoc |

Regressões do acerto: por modelo/seed, atributos padronizados externamente e classe binária, máxima verossimilhança não penalizada, covariância sandwich com correção finita por cena. Coeficientes e SE/IC robustos: `cluster_robust_logistic.csv/.tex`; convergência/rank em `logistic_diagnostics.csv/.tex`. P-valores exploratórios não corrigidos e associações descritivas.

| model | seed | term | coefficient | robust_se | ci_low | ci_high | odds_ratio | p_value | notice | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | -1 | intercept | 3.779401861106295 | 0.2540082713274379 | 3.2815547975292523 | 4.277248924683338 | 43.78984149331683 | 4.508534625827656e-50 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | luminance_mean | 0.0309437538554453 | 0.2023238111842945 | -0.3656036292806432 | 0.427491136991534 | 1.0314274884353531 | 0.8784442266313417 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | luminance_p10 | -0.2171474301182342 | 0.2505044435207539 | -0.7081271173861466 | 0.273832257149678 | 0.8048113071318606 | 0.3860293358291282 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | saturation_mean | -0.352256076872364 | 0.0843186839299251 | -0.5175176605988291 | -0.186994493145899 | 0.7031000512592664 | 2.945037657537759e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | contrast_rms | -0.5814016568213799 | 0.146016415190145 | -0.8675885717457035 | -0.2952147418970562 | 0.5591141309436257 | 6.84081460167285e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | edge_density | -0.0456237671167734 | 0.1728754780587692 | -0.3844534779220961 | 0.2932059436885492 | 0.9554013479838892 | 0.7918483797091831 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | sharpness_laplacian | 0.07216826610515 | 0.1572896766781931 | -0.2361138353240495 | 0.3804503675343497 | 1.0748361873474088 | 0.6463606157324844 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | distance_dinov2 | -0.0463425618639256 | 0.1197082047193187 | -0.2809663317677316 | 0.1882812080398803 | 0.9547148572660544 | 0.6986604355285815 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | distance_EfficientNetB3 | -0.1778560621792974 | 0.1125975629161346 | -0.3985432302418979 | 0.0428311058833031 | 0.8370628998276014 | 0.1142047029879819 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | distance_MobileNet | -0.2368174626659292 | 0.1456119983623004 | -0.5222117351729356 | 0.048576809841077 | 0.7891353215292761 | 0.1038731926179769 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | distance_ResNet50 | -0.1099173441117671 | 0.1585308339438954 | -0.4206320690808934 | 0.2007973808573593 | 0.8959081843229424 | 0.4880899701206866 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | distance_ResNet101V2 | -0.1640356112309952 | 0.1158929053335756 | -0.3911815317485072 | 0.0631103092865168 | 0.8487117976701741 | 0.1569492445055403 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| a_resnet101v2_logistic | -1 | class_clear | -3.478872409948797 | 0.2971296482131802 | -4.06123581918567 | -2.896509000711924 | 0.0308421687554833 | 1.1562251051364092e-31 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | intercept | 3.073209941217161 | 0.2321569588566196 | 2.618190663097852 | 3.52822921933647 | 21.611162016032623 | 5.3195126792575165e-40 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | luminance_mean | 0.1123598200662136 | 0.1867042865889347 | -0.253573857407333 | 0.4782934975397603 | 1.1189153964325071 | 0.5473030662161593 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | luminance_p10 | -0.0968851817571607 | 0.234915478628892 | -0.5573110592807651 | 0.3635406957664437 | 0.9076602161016634 | 0.6800274260266868 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | saturation_mean | -0.1339130164334286 | 0.0995133104314958 | -0.3289555208615091 | 0.0611294879946519 | 0.8746661428751729 | 0.1784059178964941 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | contrast_rms | -0.4981638508774696 | 0.1393861572879664 | -0.7713556991053115 | -0.2249720026496277 | 0.6076453635192807 | 0.0003515908580014 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | edge_density | -0.0029369827442859 | 0.1953983183548868 | -0.3859106493595454 | 0.3800366838709734 | 0.9970673259702952 | 0.9880076503595144 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | sharpness_laplacian | -0.0542310236982964 | 0.1805388827290294 | -0.4080807316562848 | 0.2996186842596919 | 0.9472132525138716 | 0.7638841316905954 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | distance_dinov2 | 0.0344049050728849 | 0.1064571607821094 | -0.1742472959564337 | 0.2430571061022036 | 1.0350036001042582 | 0.7465583585568103 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | distance_EfficientNetB3 | -0.1937151536798111 | 0.1241617017173115 | -0.4370676173049401 | 0.0496373099453179 | 0.8238925536096163 | 0.1187163042289868 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | distance_MobileNet | -0.2920203702202635 | 0.1520835481367397 | -0.5900986472093288 | 0.0060579067688018 | 0.7467533242887268 | 0.0548413255295693 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | distance_ResNet50 | -0.0444862837025493 | 0.1544355210435591 | -0.3471743428815945 | 0.2582017754764957 | 0.9564887194874304 | 0.7733028552765697 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | distance_ResNet101V2 | -0.1729591958345282 | 0.1119075908252582 | -0.3922940434486733 | 0.0463756517796168 | 0.841171937463292 | 0.1222123800748486 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | class_clear | -2.389616784700811 | 0.2744097121229803 | -2.927449937469842 | -1.8517836319317804 | 0.0916648045031772 | 3.0871707750922106e-18 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | intercept | 3.64115174428378 | 0.2543437661105143 | 3.142647123014907 | 4.1396563655526535 | 38.135734055002864 | 1.7417615043980004e-46 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | luminance_mean | -0.0451746662529751 | 0.1905932644441356 | -0.4187306002593893 | 0.3283812677534389 | 0.9558305159171744 | 0.8126402540327741 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | luminance_p10 | 0.0680144371808842 | 0.2322613975709701 | -0.3872095370571435 | 0.523238411418912 | 1.0703807616478949 | 0.7696477120114046 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | saturation_mean | -0.1898134458762094 | 0.0947544331681626 | -0.3755287222613107 | -0.0040981694911081 | 0.8271134209709989 | 0.0451542796799999 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | contrast_rms | -0.5542668280928545 | 0.1414678967982358 | -0.8315388107860182 | -0.2769948453996908 | 0.5744933091824332 | 8.929820263647768e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | edge_density | -0.1010803784639893 | 0.1690379111240577 | -0.4323885962890158 | 0.2302278393610372 | 0.9038603790570722 | 0.5498568766743868 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | sharpness_laplacian | 0.209496779255762 | 0.1511351762791485 | -0.0867227230484732 | 0.5057162815599974 | 1.2330574037419004 | 0.1656995810391759 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_dinov2 | 0.0687392360326635 | 0.1148672052405059 | -0.1563963492434924 | 0.2938748213088196 | 1.0711568536162122 | 0.5495573832778538 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_EfficientNetB3 | -0.1040326522173631 | 0.1105996181896636 | -0.320803920572979 | 0.1127386161382526 | 0.9011958708973234 | 0.3468975924792815 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_MobileNet | -0.2601172883318957 | 0.1440666745495786 | -0.5424827818215152 | 0.0222482051577238 | 0.7709611557525604 | 0.0709911500893132 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_ResNet50 | -0.0853652762203594 | 0.1572001584360398 | -0.3934719251189793 | 0.2227413726782604 | 0.9181768349754366 | 0.5871053183538617 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_ResNet101V2 | -0.2334293253390338 | 0.1117974363918981 | -0.4525482742310558 | -0.0143103764470119 | 0.7918135549173183 | 0.0368008378229991 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | class_clear | -3.2489774501415445 | 0.296932695359903 | -3.830954838879342 | -2.667000061403747 | 0.0388138766706783 | 7.274309727598559e-28 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | intercept | 3.604030323234888 | 0.2484285264599444 | 3.117119358641054 | 4.090941287828721 | 36.74603479985074 | 1.0890411468785724e-47 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | luminance_mean | 0.0750976274978312 | 0.1885475987590555 | -0.2944488754414163 | 0.4446441304370788 | 1.0779893871541275 | 0.6904125046814166 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | luminance_p10 | -0.1229510077957227 | 0.2315240801577651 | -0.5767298664586944 | 0.330827850867249 | 0.8843069856480501 | 0.5953836971705573 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | saturation_mean | -0.205724325132446 | 0.1027146499421991 | -0.4070413397037898 | -0.0044073105611021 | 0.8140574605355646 | 0.0451909991072055 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | contrast_rms | -0.6101547090906568 | 0.1424762765749484 | -0.8894030798289159 | -0.3309063383523978 | 0.5432668142578274 | 1.8480481340616185e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | edge_density | -0.0588367747639797 | 0.1686171578177731 | -0.3893203312623123 | 0.2716467817343528 | 0.9428606552506986 | 0.7271365643034634 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | sharpness_laplacian | 0.1602509564687819 | 0.1494724258565205 | -0.1327096148918238 | 0.4532115278293877 | 1.173805408092521 | 0.2836704055197327 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_dinov2 | 0.0657286080082391 | 0.1133254540199275 | -0.1563852004024626 | 0.287842416418941 | 1.0679368483253897 | 0.5619155420718946 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_EfficientNetB3 | -0.1017691500361275 | 0.1099484756629297 | -0.3172642024905424 | 0.1137259024182874 | 0.9032380400720996 | 0.3546499170018357 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_MobileNet | -0.2853056167179567 | 0.1457569780306209 | -0.5709840441533618 | 0.0003728107174484 | 0.7517844613072026 | 0.0502997267289158 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_ResNet50 | -0.0435408795442891 | 0.1628965821531875 | -0.362812313769198 | 0.2757305546806198 | 0.9573934154844578 | 0.7892446941278939 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | distance_ResNet101V2 | -0.2264319718897162 | 0.1136117558987252 | -0.4491069216715677 | -0.0037570221078648 | 0.7973735842861058 | 0.0462578402055563 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | class_clear | -3.1490111655910225 | 0.2900653874245182 | -3.71752887810472 | -2.5804934530773247 | 0.0428945214817566 | 1.8631969877293027e-27 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | intercept | 2.474666335428317 | 0.1714259981638301 | 2.13867755301339 | 2.810655117843244 | 11.877743269718472 | 3.08183175004364e-47 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | luminance_mean | 0.3799443318294164 | 0.1830560012757009 | 0.0211611621751343 | 0.7387275014836985 | 1.462203188991986 | 0.037934351748864 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | luminance_p10 | -0.4847520403788634 | 0.2311608520881842 | -0.9378189851072825 | -0.0316850956504442 | 0.6158498837372494 | 0.0359906138935861 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | saturation_mean | -0.3064432517993228 | 0.088538078891375 | -0.4799746976867792 | -0.1329118059118665 | 0.7360602870910758 | 0.0005378802436492 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | contrast_rms | -0.5765533495107361 | 0.1294860046670176 | -0.8303412551600691 | -0.3227654438614031 | 0.5618314699967312 | 8.482460584109713e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | edge_density | 0.3071849250953296 | 0.1641477858782999 | -0.0145388233681219 | 0.628908673558781 | 1.3595923675744088 | 0.061290706379532 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | sharpness_laplacian | -0.2167259307592144 | 0.13771652892058 | -0.4866453675194126 | 0.0531935060009837 | 0.8051506060840706 | 0.1155544215196331 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | distance_dinov2 | 0.0129831351362761 | 0.0962624570860821 | -0.1756878138157721 | 0.2016540840883245 | 1.0130677819656522 | 0.8927128724657767 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | distance_EfficientNetB3 | -0.1909047783129675 | 0.117442821314267 | -0.4210884783316975 | 0.0392789217057624 | 0.8262112576354111 | 0.1040533948994772 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | distance_MobileNet | -0.176629271297153 | 0.1340206278998937 | -0.4393048751663815 | 0.0860463325720753 | 0.8380904311150276 | 0.1875284179376664 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | distance_ResNet50 | 0.1188244426262729 | 0.1487289205147247 | -0.1726788850420998 | 0.4103277702946458 | 1.1261721931002182 | 0.4243292628531236 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | distance_ResNet101V2 | -0.332387469305253 | 0.1061221722475089 | -0.5403831048715207 | -0.1243918337389852 | 0.7172093723920427 | 0.0017354841779755 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | class_clear | -1.769066225132009 | 0.2188413619457248 | -2.1979874128733123 | -1.3401450373907058 | 0.1704921157723739 | 6.278767460495163e-16 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | intercept | 1.8016547286248588 | 0.1476261596913193 | 1.512312772453922 | 2.0909966847957957 | 6.059666276163846 | 2.953049126066758e-34 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | luminance_mean | 0.5300779574966402 | 0.1824259570222491 | 0.17252965188779 | 0.8876262631054904 | 1.699064758290947 | 0.0036641426680841 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | luminance_p10 | -0.5571127323613584 | 0.2275878664869381 | -1.0031767539940553 | -0.1110487107286615 | 0.5728606804760178 | 0.0143690839615373 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | saturation_mean | -0.3245047869799555 | 0.0768141412100555 | -0.4750577372550342 | -0.1739518367048768 | 0.722885247204396 | 2.3942413653609784e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | contrast_rms | -0.5053576412125327 | 0.1302518738093309 | -0.7606466227976703 | -0.2500686596273951 | 0.6032897755530391 | 0.0001045212034355 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | edge_density | 0.2908156002891155 | 0.1618628978358959 | -0.026429849902518 | 0.6080610504807491 | 1.3375179232902994 | 0.0723866625203161 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | sharpness_laplacian | -0.1540801488291938 | 0.1442741095388046 | -0.4368522074268298 | 0.1286919097684421 | 0.8572033144160861 | 0.2855348947624763 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | distance_dinov2 | 0.0139936670548597 | 0.092264710878726 | -0.1668418433114393 | 0.1948291774211588 | 1.0140920367289208 | 0.879448259838398 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | distance_EfficientNetB3 | -0.1375635523815964 | 0.1009939925470497 | -0.3355081404287151 | 0.0603810356655223 | 0.8714789636735575 | 0.1731674683234458 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | distance_MobileNet | -0.4022365525201293 | 0.1420331515032835 | -0.6806164140772786 | -0.1238566909629801 | 0.6688225153246694 | 0.0046259205937621 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | distance_ResNet50 | 0.3759046980212064 | 0.1440970521103689 | 0.0934796656064996 | 0.6583297304359131 | 1.456308338091086 | 0.0090889245638593 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | distance_ResNet101V2 | -0.2351954970970155 | 0.0983380009972227 | -0.4279344373632308 | -0.0424565568308002 | 0.7904163104291226 | 0.016770311695301 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | class_clear | -0.9659449165510364 | 0.1979199407491736 | -1.3538608722417076 | -0.5780289608603655 | 0.3806233724383959 | 1.0582630841126965e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | intercept | 2.368331228074434 | 0.1622663924823002 | 2.050294942907893 | 2.686367513240975 | 10.679555662956162 | 3.0076254516532454e-48 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | luminance_mean | 0.490772879817286 | 0.1794035547970357 | 0.1391483737166476 | 0.8423973859179243 | 1.633578291869342 | 0.0062270338943799 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | luminance_p10 | -0.4939976350431918 | 0.2156348155926564 | -0.9166341074177228 | -0.0713611626686607 | 0.6101822261435834 | 0.0219692419904771 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | saturation_mean | -0.2199629758956541 | 0.0897934580534247 | -0.3959549197276698 | -0.0439710320636385 | 0.8025485110522407 | 0.0142993228108148 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | contrast_rms | -0.6193770689052046 | 0.1290740473286817 | -0.8723575530082321 | -0.366396584802177 | 0.5382796443066269 | 1.5976403641990212e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | edge_density | 0.2674258087906103 | 0.1628574884274865 | -0.0517690031399031 | 0.5866206207211238 | 1.30659668825154 | 0.1005724281251773 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | sharpness_laplacian | -0.104837281805277 | 0.1325565239354387 | -0.3646432946345515 | 0.1549687310239973 | 0.9004710336869333 | 0.4290097138676026 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | distance_dinov2 | 0.0676684212760696 | 0.0972047974177133 | -0.1228494807871553 | 0.2581863233392946 | 1.0700104569495097 | 0.4863393959304271 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | distance_EfficientNetB3 | -0.3384765844417093 | 0.1059970445641523 | -0.5462269742551293 | -0.1307261946282893 | 0.7128554711023005 | 0.0014067409822156 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | distance_MobileNet | -0.2302597562563585 | 0.1320924421232393 | -0.489156185447842 | 0.0286366729351249 | 0.794327244231783 | 0.0813036941801705 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | distance_ResNet50 | 0.26519300920249 | 0.1440971130106162 | -0.0172321425745079 | 0.5476181609794881 | 1.303682574231132 | 0.0657129111850806 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | distance_ResNet101V2 | -0.2466654666233269 | 0.1010569902132592 | -0.4447335278273263 | -0.0485974054193276 | 0.7814020548798499 | 0.0146525339405803 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | class_clear | -1.651934813340564 | 0.2045392186843954 | -2.05282431538793 | -1.2510453112931978 | 0.1916786871334492 | 6.67221433740202e-16 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | intercept | 2.8688413991670285 | 0.1973389659773581 | 2.4820641331050424 | 3.2556186652290147 | 17.616595768867136 | 6.997085305347149e-48 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | luminance_mean | 0.3118588596943051 | 0.1866093357394531 | -0.053888717533956 | 0.6776064369225663 | 1.36596188714514 | 0.0946849972186535 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | luminance_p10 | -0.3198177792577839 | 0.2257496077391924 | -0.7622788799506335 | 0.1226433214350656 | 0.7262813685465894 | 0.1565728467625572 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | saturation_mean | -0.1757129608944519 | 0.0797553553346529 | -0.3320305849245617 | -0.019395336864342 | 0.8388587340436884 | 0.0275842224249994 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | contrast_rms | -0.6189598085416063 | 0.1329567390375911 | -0.8795502285571684 | -0.3583693885260441 | 0.5385042939321607 | 3.2343489669014862e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | edge_density | -0.1673398161282972 | 0.1532895789252317 | -0.4677818700270532 | 0.1331022377704587 | 0.8459121079134139 | 0.2749833922370505 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | sharpness_laplacian | -0.0937632106137829 | 0.1365480355939351 | -0.3613924425375817 | 0.1738660213100159 | 0.9104983330587556 | 0.4922917629831063 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | distance_dinov2 | -0.0453882848049728 | 0.1016811062487739 | -0.244679590960755 | 0.1539030213508093 | 0.9556263545935084 | 0.6553236821703496 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | distance_EfficientNetB3 | -0.1727482626294687 | 0.1077186633996219 | -0.3838729633555149 | 0.0383764380965774 | 0.8413493872704376 | 0.1087805191035475 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | distance_MobileNet | -0.2870229506735787 | 0.1331657709564713 | -0.5480230657217653 | -0.0260228356253921 | 0.7504945042850207 | 0.0311320447922834 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | distance_ResNet50 | 0.2008318680603388 | 0.1313707881503146 | -0.056650145334912 | 0.4583138814555896 | 1.222419226828786 | 0.1263287431526675 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | distance_ResNet101V2 | -0.1394613575638063 | 0.1027269817227784 | -0.3408025419809508 | 0.0618798268533382 | 0.8698266347752117 | 0.1745930932513698 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | class_clear | -2.8872420871976567 | 0.2415389528544836 | -3.360649735655949 | -2.413834438739364 | 0.0557296985128306 | 6.222810488440639e-33 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | intercept | 2.8019938343484827 | 0.1894321109577005 | 2.430713719356005 | 3.1732739493409605 | 16.47746738123499 | 1.6609377249597372e-49 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | luminance_mean | 0.1434918675628772 | 0.1853511400717166 | -0.2197896914711161 | 0.5067734265968705 | 1.1542974235177192 | 0.4388348109884036 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | luminance_p10 | 0.0201844394297555 | 0.2214178511751605 | -0.4137865744077968 | 0.454155453267308 | 1.0203895227337685 | 0.9273655005881228 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | saturation_mean | -0.3115423850270466 | 0.0864701377511953 | -0.4810207407576021 | -0.142064029296491 | 0.7323165705899091 | 0.0003146988609179 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | contrast_rms | -0.4401356653241256 | 0.1316004232499882 | -0.6980677552443231 | -0.1822035754039282 | 0.6439490535998293 | 0.0008243543361299 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | edge_density | -0.6967346826942079 | 0.1655931804999996 | -1.0212913525596388 | -0.3721780128287771 | 0.4982094626315606 | 2.5820191359988477e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | sharpness_laplacian | 0.5066572011743793 | 0.1644987590237637 | 0.184245557986278 | 0.8290688443624805 | 1.6597337553713885 | 0.0020699627116534 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | distance_dinov2 | -0.1522437207715826 | 0.0967502653065324 | -0.341870756267076 | 0.0373833147239107 | 0.8587789529643606 | 0.1155859230926016 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | distance_EfficientNetB3 | -0.2223827199417956 | 0.1098847269287837 | -0.4377528271732244 | -0.0070126127103668 | 0.8006088967024158 | 0.0429925987646472 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | distance_MobileNet | -0.2004690925517672 | 0.1271506523167949 | -0.4496797917034529 | 0.0487416065999185 | 0.8183467826456862 | 0.1148814831279144 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | distance_ResNet50 | 0.319827530284191 | 0.1362401282068933 | 0.0528017857495677 | 0.5868532748188142 | 1.3768902719825171 | 0.0188984472314096 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | distance_ResNet101V2 | -0.0015431538706703 | 0.107455365895374 | -0.2121518009711713 | 0.2090654932298307 | 0.998458036179042 | 0.9885420675089144 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | class_clear | -3.154588122338513 | 0.2346349107136819 | -3.614464096853089 | -2.6947121478239384 | 0.0426559664146975 | 3.3089426250344284e-41 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | intercept | 2.905348385887482 | 0.2139794360302905 | 2.485956397835932 | 3.3247403739390324 | 18.27160811390898 | 5.430577125185471e-42 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | luminance_mean | 0.5980408226060713 | 0.1932985991164346 | 0.2191825300758238 | 0.9768991151363188 | 1.8185524411337008 | 0.0019756373878998 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | luminance_p10 | -0.8172546203104077 | 0.2298708144784684 | -1.267793137785082 | -0.3667161028357336 | 0.4416424679357297 | 0.0003775798262354 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | saturation_mean | -0.2146927353152135 | 0.0903194275732967 | -0.391715560463144 | -0.0376699101672829 | 0.8067892999560385 | 0.0174522776227157 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | contrast_rms | -0.7812186495197816 | 0.1436436564457311 | -1.0627550427610517 | -0.4996822562785114 | 0.4578477152923595 | 5.370451762727839e-08 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | edge_density | -0.0152964240729671 | 0.1610977307841919 | -0.3310421744011043 | 0.30044932625517 | 0.98481997198485 | 0.9243535824676912 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | sharpness_laplacian | -0.1486053207730699 | 0.1435720212475776 | -0.4300013116059336 | 0.1327906700597937 | 0.8619092254440419 | 0.3006419828206086 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | distance_dinov2 | -0.0095009031301144 | 0.0971255005665085 | -0.1998633862208904 | 0.1808615799606616 | 0.9905440878523012 | 0.9220745172922188 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | distance_EfficientNetB3 | -0.3121106392827728 | 0.111662079236767 | -0.530964293025688 | -0.0932569855398575 | 0.7319005467969728 | 0.0051877969733616 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | distance_MobileNet | -0.4493927776861393 | 0.134090294006917 | -0.7122049246160765 | -0.186580630756202 | 0.6380154512399686 | 0.0008039851477379 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | distance_ResNet50 | 0.298174264389231 | 0.1410595619580728 | 0.0217026032764197 | 0.5746459255020424 | 1.3473965706622593 | 0.0345308008894211 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | distance_ResNet101V2 | -0.1162561345907948 | 0.1056804013571685 | -0.3233859151225773 | 0.0908736459409876 | 0.8902471709978284 | 0.2713004056803946 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | class_clear | -2.9260682126540924 | 0.2599284103609581 | -3.435518535520304 | -2.4166178897878807 | 0.0536073972166587 | 2.134130159624621e-29 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | intercept | 2.0344117954664864 | 0.1521761185754459 | 1.736152083751524 | 2.332671507181449 | 7.647752359607924 | 9.201573225486368e-41 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | luminance_mean | 0.2793798683899757 | 0.1744275001610164 | -0.0624917498389616 | 0.621251486618913 | 1.3223095520775856 | 0.1092228005500312 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | luminance_p10 | -0.1499201400957703 | 0.2180673399126811 | -0.5773242725290675 | 0.2774839923375267 | 0.8607767152263254 | 0.4917710404604574 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | saturation_mean | -0.2354664938768917 | 0.085135116409848 | -0.4023282558598143 | -0.0686047318939692 | 0.7902021391754055 | 0.0056783780177503 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | contrast_rms | -0.4867063943800135 | 0.1250980327243707 | -0.7318940330565864 | -0.2415187557034405 | 0.6146474704062299 | 9.999669477567224e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | edge_density | 0.5341472188496819 | 0.2002994537427702 | 0.1415675033908165 | 0.9267269343085472 | 1.705992783260276 | 0.0076590155321374 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | sharpness_laplacian | -0.3580370821623644 | 0.1922104608123343 | -0.7347626628063767 | 0.0186884984816478 | 0.6990471523458908 | 0.062499610722326 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | distance_dinov2 | -0.0622927075457774 | 0.0951934183512798 | -0.248868379079535 | 0.1242829639879802 | 0.9396078162375554 | 0.5128667562399577 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | distance_EfficientNetB3 | -0.1020278664966329 | 0.1062825146909746 | -0.3103377674772868 | 0.1062820344840209 | 0.9030043877495324 | 0.3370710673816813 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | distance_MobileNet | -0.2700414385179355 | 0.1442565905487297 | -0.5527791605259791 | 0.012696283490108 | 0.7633478616773952 | 0.061213220220047 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | distance_ResNet50 | 0.1990668816671674 | 0.1507812067832987 | -0.0964588531735763 | 0.4945926165079112 | 1.2202635764330934 | 0.1867560049896006 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | distance_ResNet101V2 | -0.216805170691599 | 0.1037138297473749 | -0.4200805416951672 | -0.0135297996880308 | 0.8050868085321754 | 0.0365803416536626 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1980 | class_clear | -1.144059764723142 | 0.2014778447490441 | -1.53894908411401 | -0.7491704453322736 | 0.3185232638496752 | 1.3600793292564368e-08 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | intercept | 3.630664375977709 | 0.258677565785965 | 3.123665663428741 | 4.137663088526677 | 37.73788043157013 | 9.455109097862045e-45 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | luminance_mean | 0.1150203682797403 | 0.1949568291420046 | -0.267087995378707 | 0.4971287319381878 | 1.1218962884366637 | 0.5552049583372156 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | luminance_p10 | -0.1019866935498722 | 0.2407531780375427 | -0.5738542516670024 | 0.369880864567258 | 0.9030415678665163 | 0.6718465049280196 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | saturation_mean | -0.3507258241516434 | 0.0908781373352585 | -0.5288437003108302 | -0.1726079479924566 | 0.7041767956610546 | 0.0001137131523532 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | contrast_rms | -0.5211551989967431 | 0.142831889306236 | -0.8011005578807697 | -0.2412098401127164 | 0.5938341549662225 | 0.0002635384665202 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | edge_density | -0.2015955518813529 | 0.1569525100678609 | -0.5092168188975121 | 0.1060257151348062 | 0.8174254672871013 | 0.1989892258772688 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | sharpness_laplacian | -0.0765884547659214 | 0.1359056909421848 | -0.3429587143066279 | 0.189781804774785 | 0.9262709775825276 | 0.5730664204263441 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | distance_dinov2 | -0.1100378796699882 | 0.1074583975987696 | -0.3206524687999563 | 0.1005767094599798 | 0.8958002020378134 | 0.3058331627678762 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | distance_EfficientNetB3 | -0.3575376555810698 | 0.1150520326204353 | -0.5830354958652444 | -0.1320398152968953 | 0.6993963622704514 | 0.0018860229702414 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | distance_MobileNet | -0.3479284233001597 | 0.148871589656081 | -0.6397113773472962 | -0.0561454692530232 | 0.7061494182504993 | 0.0194337646453038 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | distance_ResNet50 | 0.1212171565457534 | 0.1450550309618396 | -0.1630854799157869 | 0.4055197930072938 | 1.1288700272679542 | 0.403344383215278 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | distance_ResNet101V2 | -0.1006716980159444 | 0.1087414909655053 | -0.3138011039335166 | 0.1124577079016278 | 0.9042298446132876 | 0.354555540454337 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1981 | class_clear | -3.616089924111774 | 0.3052852932989932 | -4.214438103987531 | -3.0177417442360164 | 0.0268876037933851 | 2.2854668636713843e-32 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | intercept | 1.4473916624582577 | 0.1258617396787532 | 1.200707185656352 | 1.6940761392601629 | 4.252009362823128 | 1.3213836386737588e-30 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | luminance_mean | 0.6118417033048767 | 0.1776934603969893 | 0.2635689206384927 | 0.9601144859712608 | 1.8438240505411985 | 0.0005747833253776 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | luminance_p10 | -0.5902669447876815 | 0.2236058870999901 | -1.0285264302347796 | -0.1520074593405834 | 0.5541793297039682 | 0.0082963587717105 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | saturation_mean | -0.1397699010015386 | 0.0804950524077556 | -0.2975373046543994 | 0.0179975026513221 | 0.8695582968741651 | 0.0824968815623335 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | contrast_rms | -0.5932748602207192 | 0.1261253642536689 | -0.840476031694899 | -0.3460736887465393 | 0.5525149096178241 | 2.553001962735594e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | edge_density | 0.5014393390289144 | 0.1762728180487324 | 0.1559509642000263 | 0.8469277138578024 | 1.6510960482178465 | 0.0044456393126358 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | sharpness_laplacian | -0.4140757226693514 | 0.1617290603254962 | -0.7310588561608211 | -0.0970925891778817 | 0.6609509004044551 | 0.0104580333843232 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | distance_dinov2 | -0.108993306300081 | 0.0935274822031732 | -0.2923038029830065 | 0.0743171903828443 | 0.8967364199626476 | 0.2438727838919806 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | distance_EfficientNetB3 | -0.1207406088440476 | 0.1030920072274785 | -0.322797230103843 | 0.0813160124157477 | 0.8862638187762982 | 0.2415213189640764 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | distance_MobileNet | -0.1543472518736257 | 0.1286499466282993 | -0.4064965138780855 | 0.097802010130834 | 0.8569743833765583 | 0.2302380231133329 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | distance_ResNet50 | 0.2351876436870976 | 0.1326087310401794 | -0.0247206931872056 | 0.495095980561401 | 1.2651461431483078 | 0.0761383275494674 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | distance_ResNet101V2 | -0.2852843208176918 | 0.0973822672921753 | -0.476150057443203 | -0.0944185841921805 | 0.7518004714045854 | 0.0033947451120998 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_MobileNet | 1982 | class_clear | -0.6339517552531972 | 0.1772054187827929 | -0.9812679939327994 | -0.2866355165735951 | 0.5304912816621757 | 0.0003469009394957 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | intercept | 0.8339199336379246 | 0.1128308104558453 | 0.6127756087980084 | 1.055064258477841 | 2.3023260400115064 | 1.4585094513455042e-13 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | luminance_mean | 1.037235276866528 | 0.1708081859436434 | 0.7024573841523754 | 1.3720131695806808 | 2.821405815498987 | 1.2592230635964218e-09 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | luminance_p10 | -0.8648332608218993 | 0.2183726959585888 | -1.292835880107637 | -0.4368306415361616 | 0.4211217642704974 | 7.483862177716119e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | saturation_mean | -0.0704489012773799 | 0.0712020762382711 | -0.2100024063288627 | 0.0691046037741027 | 0.9319753610595536 | 0.3224567109655581 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | contrast_rms | -0.643938093249563 | 0.1252354327680092 | -0.8893950310631416 | -0.3984811554359843 | 0.5252199807232896 | 2.720892692120021e-07 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | edge_density | -0.1229796994665081 | 0.1579888094508192 | -0.4326320759504667 | 0.1866726770174505 | 0.8842816137671272 | 0.4363287244291651 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | sharpness_laplacian | -0.0294522334007778 | 0.1442530187002021 | -0.3121829547143491 | 0.2532784879127933 | 0.9709772568149778 | 0.8382201417764578 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | distance_dinov2 | -0.1402646268249218 | 0.0856997052753843 | -0.3082329626503678 | 0.027703709000524 | 0.8691282103259972 | 0.1016933660804227 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | distance_EfficientNetB3 | -0.1957873997917558 | 0.0956144244562269 | -0.3831882281284812 | -0.0083865714550305 | 0.8221870132287782 | 0.0405917184570224 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | distance_MobileNet | -0.0883204324375717 | 0.1262434472783942 | -0.3357530423873986 | 0.1591121775122552 | 0.9154674842421088 | 0.4841745857037214 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | distance_ResNet50 | 0.0408543869484575 | 0.130772496131983 | -0.2154549956386257 | 0.2971637695355407 | 1.041700409325878 | 0.7547303958911036 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | distance_ResNet101V2 | -0.0417522389981179 | 0.0947991123762883 | -0.2275550850220032 | 0.1440506070257672 | 0.9591073805427816 | 0.6596267248672639 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | class_clear | -0.327316308972175 | 0.1659639379533495 | -0.6525996500931712 | -0.0020329678511787 | 0.7208556938379018 | 0.048585246053272 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | intercept | 1.3215777316607096 | 0.1196536168320026 | 1.0870609520500352 | 1.556094511271384 | 3.7493321532702586 | 2.316914621412169e-28 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | luminance_mean | 0.5977384623372117 | 0.1725496139189766 | 0.2595474335097356 | 0.9359294911646878 | 1.818002666247894 | 0.0005319015996049 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | luminance_p10 | -0.5054799679112548 | 0.2177476251857575 | -0.9322574709944546 | -0.0787024648280551 | 0.603215981619999 | 0.0202651093380681 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | saturation_mean | -0.2063674277968714 | 0.0731285441307576 | -0.3496967405350004 | -0.0630381150587424 | 0.8135341063169824 | 0.004772787112443 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | contrast_rms | -0.4560991824745206 | 0.1267151300953414 | -0.7044562737576905 | -0.2077420911913507 | 0.6337509769911622 | 0.0003189450628874 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | edge_density | 0.2227575812732488 | 0.1596122839055838 | -0.0900767466718689 | 0.5355919092183667 | 1.2495176305448663 | 0.1628299274883472 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | sharpness_laplacian | -0.2929372479418757 | 0.1507638918240277 | -0.5884290460860547 | 0.0025545502023032 | 0.7460689565908379 | 0.0520137492050811 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | distance_dinov2 | -0.0942709186165956 | 0.0880231966741143 | -0.2667932139019408 | 0.0782513766687494 | 0.9100361830970582 | 0.284179263114145 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | distance_EfficientNetB3 | -0.1683826464478349 | 0.0942265235522742 | -0.3530632389987025 | 0.0162979461030327 | 0.8450304249223974 | 0.0739377478183205 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | distance_MobileNet | -0.0508605974777917 | 0.1228799731058756 | -0.2917009191865518 | 0.1899797242309683 | 0.9504111510097416 | 0.6789439072029368 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | distance_ResNet50 | -0.0152192741986631 | 0.1273902550927918 | -0.2648995861618985 | 0.2344610377645722 | 0.9848959536528512 | 0.904903252436992 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | distance_ResNet101V2 | -0.2528223202028599 | 0.0987180681552524 | -0.4463061784105199 | -0.0593384619951999 | 0.7766058567367803 | 0.0104355083152381 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | class_clear | -0.856870623531271 | 0.1668318745502594 | -1.183855089123075 | -0.5298861579394669 | 0.4244883899561923 | 2.8044986955127297e-07 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | intercept | 0.790998351377075 | 0.1174519286950058 | 0.5607968012201033 | 1.0211999015340467 | 2.20559728854648 | 1.6431807776558663e-11 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | luminance_mean | 0.3822329464506757 | 0.1671011990796854 | 0.0547206144810437 | 0.7097452784203077 | 1.4655534408449358 | 0.0221705098553977 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | luminance_p10 | -0.3150570619690431 | 0.2100356769003851 | -0.726719424162278 | 0.0966053002241916 | 0.7297472322670039 | 0.1336100285791026 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | saturation_mean | -0.0836248759591136 | 0.0746030795646946 | -0.2298442250416871 | 0.0625944731234598 | 0.9197762215616028 | 0.2623173974727204 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | contrast_rms | -0.4638326859041415 | 0.1267965742210758 | -0.7123494047405032 | -0.2153159670677799 | 0.6288687642740193 | 0.0002541064706349 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | edge_density | 0.1339151727848421 | 0.1448212234021285 | -0.1499292092803512 | 0.4177595548500356 | 1.1432958329295395 | 0.3551256171273912 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | sharpness_laplacian | -0.1697775454047766 | 0.1348008195708618 | -0.4339822968501405 | 0.0944272060405873 | 0.8438525145880302 | 0.2078607341884797 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | distance_dinov2 | -0.2168164910533224 | 0.0839299275934907 | -0.3813161263616141 | -0.0523168557450307 | 0.80507769470987 | 0.0097859142747358 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | distance_EfficientNetB3 | -0.0631217971481088 | 0.0994736157079371 | -0.258086501347638 | 0.1318429070514203 | 0.9388291200159194 | 0.5257166064879562 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | distance_MobileNet | 0.0304702184688147 | 0.1468146020111008 | -0.2572811138775167 | 0.3182215508151462 | 1.0309391866440487 | 0.8355864765716786 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | distance_ResNet50 | 0.1050712358653427 | 0.1397472221535928 | -0.1688282864952096 | 0.3789707582258951 | 1.1107897356054235 | 0.4521314273487242 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | distance_ResNet101V2 | -0.0528594669598424 | 0.0914579383161327 | -0.2321137321597434 | 0.1263947982400585 | 0.948513300574148 | 0.5632878997084521 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | class_clear | -0.3136248265397275 | 0.1642348684031607 | -0.635519253615589 | 0.008269600536134 | 0.7307931509295931 | 0.0561832273299116 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | intercept | 1.9730559522244973 | 0.1527510711348662 | 1.673669354200252 | 2.272442550248743 | 7.192623250160341 | 3.618530243217068e-38 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | luminance_mean | 0.4962712418356604 | 0.1822101713132476 | 0.1391458684448316 | 0.8533966152264894 | 1.642585035180096 | 0.0064570780058731 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | luminance_p10 | -0.2953826364881324 | 0.2299882249542008 | -0.7461512742666498 | 0.155386001290385 | 0.7442467570107154 | 0.1990236863921287 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | saturation_mean | -0.1278555170423573 | 0.0798477218074132 | -0.2843541760324564 | 0.0286431419477416 | 0.8799805121472094 | 0.1093233538779433 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | contrast_rms | -0.6641519639054034 | 0.1404601263986248 | -0.9394487529106444 | -0.3888551749001626 | 0.5147098351967245 | 2.2629363475991663e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | edge_density | -0.2183642449437725 | 0.1432998382298914 | -0.4992267668647679 | 0.0624982769772229 | 0.8038325963772195 | 0.1275517893531388 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | sharpness_laplacian | 0.0934456262375202 | 0.1311922876485628 | -0.1636865326030747 | 0.3505777850781152 | 1.0979509019966611 | 0.4762914837132695 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | distance_dinov2 | -0.1910416445560275 | 0.0925570357823809 | -0.3724501012052742 | -0.0096331879067808 | 0.8260981849426986 | 0.0390136982144946 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | distance_EfficientNetB3 | 2.219583011851001e-05 | 0.102873589023146 | -0.2016063336156171 | 0.2016507252758541 | 1.0000221960764475 | 0.9998278497892752 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | distance_MobileNet | 0.1339414588637054 | 0.1306208202842083 | -0.1220706445244147 | 0.3899535622518256 | 1.143325886088956 | 0.305164126095912 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | distance_ResNet50 | -0.0654007063432003 | 0.1338996139596918 | -0.3278391272480058 | 0.1970377145616051 | 0.9366920497211614 | 0.6252446731330386 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | distance_ResNet101V2 | -0.1846342606895705 | 0.1010865089705178 | -0.382760177594665 | 0.013491656215524 | 0.8314083069530074 | 0.0677753503880075 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1980 | class_clear | -2.3493142967325307 | 0.2015995934140532 | -2.7444422391219825 | -1.954186354343079 | 0.0954345795875863 | 2.2056514426642097e-31 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | intercept | 1.6808139884384912 | 0.1394538467556219 | 1.407489471291912 | 1.9541385055850704 | 5.369925249678412 | 1.8739543566593258e-33 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | luminance_mean | 0.3178004724822013 | 0.1698753981258233 | -0.0151491897038062 | 0.6507501346682087 | 1.3741020626996223 | 0.0613747848574692 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | luminance_p10 | -0.0340478086013952 | 0.2158429242048059 | -0.4570921663606119 | 0.3889965491578213 | 0.9665252953103318 | 0.8746589570027459 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | saturation_mean | -0.0977868780764174 | 0.0697379073921143 | -0.2344706649221474 | 0.0388969087693124 | 0.906842151104195 | 0.1608538864268388 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | contrast_rms | -0.4298734022606307 | 0.1266920057527389 | -0.6781851706651336 | -0.1815616338561278 | 0.6505914529172183 | 0.0006911684546802 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | edge_density | -0.1069572193705595 | 0.1393377966879367 | -0.3800542825640723 | 0.1661398438229533 | 0.8985641133064091 | 0.4427183512443348 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | sharpness_laplacian | -0.0132917786516999 | 0.1274252502337919 | -0.2630406798309294 | 0.2364571225275295 | 0.9867961669557668 | 0.9169230259923074 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | distance_dinov2 | -0.1521803409473131 | 0.0864613377546406 | -0.3216414490015573 | 0.017280767106931 | 0.8588333839483807 | 0.0783913618560815 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | distance_EfficientNetB3 | -0.0376596739048891 | 0.0979434377326294 | -0.229625284382879 | 0.1543059365731007 | 0.9630406329850082 | 0.7006047023523989 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | distance_MobileNet | -0.0345615618912515 | 0.1128885441818227 | -0.2558190427547767 | 0.1866959189722736 | 0.9660288672918198 | 0.7594854889495355 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | distance_ResNet50 | 0.0542858512680576 | 0.124762903276464 | -0.1902449457604594 | 0.2988166482965748 | 1.055786356893454 | 0.6634810468961467 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | distance_ResNet101V2 | -0.0471103880512068 | 0.0945352183852549 | -0.2323960113569302 | 0.1381752352545165 | 0.9539820835546826 | 0.618246600790926 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1981 | class_clear | -1.5448929861285243 | 0.1851431627230626 | -1.907766917049556 | -1.1820190552074927 | 0.2133346997735505 | 7.162745645068011e-17 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | intercept | 1.4601081130066242 | 0.1303684807988652 | 1.204590585921654 | 1.7156256400915946 | 4.306425083742061 | 4.084002863210623e-29 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | luminance_mean | 0.6633273619866784 | 0.1682560738626054 | 0.3335515170358696 | 0.9931032069374872 | 1.9412408111263624 | 8.06810550125437e-05 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | luminance_p10 | -0.4395664331955515 | 0.2119086027951475 | -0.854899662688233 | -0.0242332037028699 | 0.6443157144379674 | 0.038049522626531 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | saturation_mean | 0.0134009275127164 | 0.0724772774046575 | -0.1286519258979271 | 0.1554537809233599 | 1.0134911223898178 | 0.8533087813330738 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | contrast_rms | -0.5608085681851628 | 0.1230532227588015 | -0.8019884529739917 | -0.3196286833963339 | 0.5707473890456419 | 5.177909586292064e-06 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | edge_density | 0.1265725519840509 | 0.1418395262558989 | -0.1514278110617266 | 0.4045729150298286 | 1.1349317897422304 | 0.3721976150460712 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | sharpness_laplacian | -0.2736410076528081 | 0.1279464152515157 | -0.5244113734967784 | -0.0228706418088378 | 0.7606050776571788 | 0.0324586999417038 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | distance_dinov2 | -0.1968411543308184 | 0.085574365967439 | -0.3645638296268443 | -0.0291184790347924 | 0.8213210862489998 | 0.021434908193074 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | distance_EfficientNetB3 | -0.1075432009799043 | 0.1012711616230686 | -0.3060310304336482 | 0.0909446284738395 | 0.8980377255030427 | 0.288266046537928 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | distance_MobileNet | 0.0250261259107148 | 0.1136324067526578 | -0.1976892988010944 | 0.2477415506225241 | 1.0253419081656463 | 0.8256861329320722 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | distance_ResNet50 | 0.1864361688029343 | 0.1223845693589047 | -0.0534331794039565 | 0.4263055170098253 | 1.204947706376412 | 0.1276678234418636 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | distance_ResNet101V2 | -0.2240298614620571 | 0.0939306060356275 | -0.4081304663379026 | -0.0399292565862116 | 0.7992912660071824 | 0.0170764693315989 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |
| ft_ResNet50 | 1982 | class_clear | -1.3605551210634026 | 0.1789751440607964 | -1.7113399575504218 | -1.0097702845763834 | 0.2565183386890179 | 2.9176412804687567e-14 | Descriptive association; attributes standardised externally, class unscaled; not causal | exploratory/post hoc |

| model | seed | converged | max_abs_gradient | design_rank | parameters | clusters | observations | separation_warning | status | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | -1 | True | 2.256992950311343e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | -1 | True | 8.21915237203541e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | -1 | True | 4.836942260055821e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | -1 | True | 2.09474223300532e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB3 | 1980 | True | 4.80977503158897e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB3 | 1981 | True | 1.524163710797666e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB3 | 1982 | True | 1.46644854037703e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB4 | 1980 | True | 3.864689768207086e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB4 | 1981 | True | 1.76369967519463e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_EfficientNetB4 | 1982 | True | 7.629180380774869e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_MobileNet | 1980 | True | 3.146495552996953e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_MobileNet | 1981 | True | 1.311683966775945e-06 | 13 | 13 | 1378 | 1576 | False | Desired error not necessarily achieved due to precision loss. | exploratory/post hoc |
| ft_MobileNet | 1982 | True | 8.413084069402998e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet101V2 | 1980 | True | 2.713655282704508e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet101V2 | 1981 | True | 9.349150045245836e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet101V2 | 1982 | True | 4.296244742185706e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet50 | 1980 | True | 2.050618261861814e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet50 | 1981 | True | 5.907468274557459e-07 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |
| ft_ResNet50 | 1982 | True | 5.2997863875248186e-08 | 13 | 13 | 1378 | 1576 | False | Optimization terminated successfully. | exploratory/post hoc |

| cluster | external_images | via_images | external_clear_fraction | via_clear_fraction | analysis_scope |
| --- | --- | --- | --- | --- | --- |
| 0 | 127 | 68 | 0.84251968503937 | 0.6470588235294118 | exploratory/post hoc |
| 1 | 153 | 19 | 0.0 | 0.0 | exploratory/post hoc |
| 2 | 188 | 23 | 0.925531914893617 | 0.6956521739130435 | exploratory/post hoc |
| 3 | 85 | 12 | 0.5764705882352941 | 0.1666666666666666 | exploratory/post hoc |
| 4 | 207 | 37 | 0.9516908212560388 | 0.8108108108108109 | exploratory/post hoc |
| 5 | 51 | 1 | 0.6078431372549019 | 0.0 | exploratory/post hoc |
| 6 | 179 | 50 | 0.0726256983240223 | 0.12 | exploratory/post hoc |
| 7 | 115 | 43 | 0.4260869565217391 | 0.8604651162790697 | exploratory/post hoc |
| 8 | 167 | 50 | 0.8263473053892215 | 0.66 | exploratory/post hoc |
| 9 | 108 | 24 | 0.75 | 0.2916666666666667 | exploratory/post hoc |
| 10 | 61 | 2 | 0.0327868852459016 | 0.0 | exploratory/post hoc |
| 11 | 135 | 13 | 0.0222222222222222 | 0.0 | exploratory/post hoc |

| model | cluster | endpoint | estimate | ci_low | ci_high | bootstrap_valid | bootstrap_total | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_resnet101v2_logistic | 0 | accuracy | 0.8740157480314961 | 0.8102189781021898 | 0.9281162464985993 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 0 | obstructed_recall | 0.95 | 0.8333333333333334 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 0 | clear_recall | 0.8598130841121495 | 0.7876078539823009 | 0.9223321894877804 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 0 | accuracy | 0.8503937007874016 | 0.7802990006447453 | 0.9130434782608696 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 0 | obstructed_recall | 0.9 | 0.75 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 0 | clear_recall | 0.8411214953271028 | 0.7619047619047619 | 0.910002808988764 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 0 | accuracy | 0.8503937007874016 | 0.7851851851851852 | 0.9097744360902256 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 0 | obstructed_recall | 0.85 | 0.6666666666666666 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 0 | clear_recall | 0.8504672897196262 | 0.7787592485129842 | 0.9150983370642788 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 0 | accuracy | 0.8582677165354331 | 0.7930972906403941 | 0.9155920014564872 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 0 | obstructed_recall | 0.9 | 0.75 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 0 | clear_recall | 0.8504672897196262 | 0.776595744680851 | 0.9148936170212766 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 0 | accuracy | 0.8976377952755904 | 0.8503937007874015 | 0.9393939393939394 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 0 | obstructed_recall | 0.75 | 0.5641025641025642 | 0.9102564102564102 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 0 | clear_recall | 0.9252336448598132 | 0.8814102564102564 | 0.9629629629629628 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 0 | accuracy | 0.7559055118110236 | 0.6904761904761906 | 0.8170442699669528 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 0 | obstructed_recall | 0.6166666666666667 | 0.4333333333333333 | 0.7916666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 0 | clear_recall | 0.7819314641744549 | 0.715210355987055 | 0.8448844884488449 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 0 | accuracy | 0.8451443569553806 | 0.7912978585334198 | 0.8933333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 0 | obstructed_recall | 0.5833333333333334 | 0.4222222222222222 | 0.7391304347826088 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 0 | clear_recall | 0.8940809968847351 | 0.8433333333333334 | 0.9371069182389936 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 0 | accuracy | 0.7795275590551182 | 0.7119993527508092 | 0.8416666666666667 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 0 | obstructed_recall | 0.3666666666666666 | 0.2352941176470588 | 0.5 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 0 | clear_recall | 0.8566978193146418 | 0.7916666666666666 | 0.915204678362573 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 0 | accuracy | 0.7034120734908136 | 0.637034677990092 | 0.7715067862245282 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 0 | obstructed_recall | 0.5833333333333334 | 0.4333333333333333 | 0.7291666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 0 | clear_recall | 0.7258566978193146 | 0.6513383354618988 | 0.8012247118325101 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 1 | accuracy | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 1 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 1 | accuracy | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 1 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 1 | accuracy | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 1 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 1 | accuracy | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 1 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 1 | accuracy | 0.9477124183006536 | 0.9176942628903412 | 0.9738955823293172 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 1 | obstructed_recall | 0.9477124183006536 | 0.9176942628903412 | 0.9738955823293172 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 1 | accuracy | 0.9847494553376906 | 0.9703196347031964 | 0.9957264957264956 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 1 | obstructed_recall | 0.9847494553376906 | 0.9703196347031964 | 0.9957264957264956 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| ft_MobileNet | 1 | accuracy | 0.9411764705882352 | 0.9106753812636166 | 0.9686282889224066 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 1 | obstructed_recall | 0.9411764705882352 | 0.9106753812636166 | 0.9686282889224066 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 1 | accuracy | 0.6884531590413943 | 0.617997634968438 | 0.7577639751552795 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 1 | obstructed_recall | 0.6884531590413943 | 0.617997634968438 | 0.7577639751552795 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 1 | accuracy | 0.8278867102396514 | 0.779166027607362 | 0.8760698198198198 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 1 | obstructed_recall | 0.8278867102396514 | 0.779166027607362 | 0.8760698198198198 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 1 | clear_recall | — | — | — | 0 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 2 | accuracy | 0.3563829787234042 | 0.2886597938144329 | 0.4254173033972023 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 2 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 2 | clear_recall | 0.3045977011494253 | 0.2378378378378378 | 0.3742331288343558 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 2 | accuracy | 0.5372340425531915 | 0.4659615285904938 | 0.6086956521739131 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 2 | obstructed_recall | 0.9285714285714286 | 0.7647058823529411 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 2 | clear_recall | 0.5057471264367817 | 0.43125 | 0.5798816568047337 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 2 | accuracy | 0.4148936170212766 | 0.3442578882425524 | 0.4874371859296482 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 2 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 2 | clear_recall | 0.367816091954023 | 0.2954507321537523 | 0.441340782122905 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 2 | accuracy | 0.4042553191489361 | 0.3333333333333333 | 0.4769230769230769 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 2 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 2 | clear_recall | 0.3563218390804598 | 0.2848484848484848 | 0.4285714285714285 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 2 | accuracy | 0.3049645390070922 | 0.2509353227820063 | 0.3621621621621622 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 2 | obstructed_recall | 0.9761904761904764 | 0.9215686274509804 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 2 | clear_recall | 0.2509578544061303 | 0.2012578616352201 | 0.303923110124553 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 2 | accuracy | 0.1914893617021277 | 0.1417624521072797 | 0.2455197132616487 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 2 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 2 | clear_recall | 0.1264367816091954 | 0.0859646364338519 | 0.1717171717171717 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 2 | accuracy | 0.3333333333333333 | 0.2771518024344569 | 0.3913061504073833 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 2 | obstructed_recall | 0.9047619047619048 | 0.7619047619047619 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 2 | clear_recall | 0.2873563218390804 | 0.2339159465828751 | 0.3434969719595155 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 2 | accuracy | 0.2943262411347517 | 0.2422833181542198 | 0.346592803030303 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 2 | obstructed_recall | 0.7619047619047619 | 0.5757575757575758 | 0.923076923076923 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 2 | clear_recall | 0.2567049808429119 | 0.2067039106145251 | 0.3075541861453463 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 2 | accuracy | 0.2358156028368794 | 0.1853930080714614 | 0.2887629198966407 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 2 | obstructed_recall | 0.8571428571428572 | 0.7142857142857143 | 0.9761904761904764 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 2 | clear_recall | 0.185823754789272 | 0.1402550091074681 | 0.2339544513457556 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 3 | accuracy | 0.7647058823529411 | 0.662918082048602 | 0.8552631578947368 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 3 | obstructed_recall | 0.9722222222222222 | 0.9047619047619048 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 3 | clear_recall | 0.6122448979591837 | 0.4545454545454545 | 0.7555555555555555 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 3 | accuracy | 0.7529411764705882 | 0.6527777777777778 | 0.8461538461538461 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 3 | obstructed_recall | 0.9722222222222222 | 0.9090488215488216 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 3 | clear_recall | 0.5918367346938775 | 0.44 | 0.7333333333333333 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 3 | accuracy | 0.788235294117647 | 0.6911764705882353 | 0.875 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 3 | obstructed_recall | 0.9444444444444444 | 0.8571428571428571 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 3 | clear_recall | 0.673469387755102 | 0.5227272727272727 | 0.8048780487804879 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 3 | accuracy | 0.788235294117647 | 0.6911764705882353 | 0.875 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 3 | obstructed_recall | 0.9444444444444444 | 0.8571428571428571 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 3 | clear_recall | 0.673469387755102 | 0.5227272727272727 | 0.8048780487804879 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 3 | accuracy | 0.784313725490196 | 0.6984126984126985 | 0.8631578947368421 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 3 | obstructed_recall | 0.8333333333333334 | 0.6979106104651163 | 0.9494949494949496 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 3 | clear_recall | 0.7482993197278911 | 0.6326530612244898 | 0.8518518518518517 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 3 | accuracy | 0.5333333333333333 | 0.4272300469483568 | 0.6413070926657883 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 3 | obstructed_recall | 0.9814814814814816 | 0.9523809523809524 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 3 | clear_recall | 0.2040816326530612 | 0.1171171171171171 | 0.3 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 3 | accuracy | 0.6235294117647059 | 0.524904214559387 | 0.7176470588235294 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 3 | obstructed_recall | 0.9351851851851852 | 0.8518518518518517 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 3 | clear_recall | 0.3945578231292517 | 0.2866666666666667 | 0.5 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 3 | accuracy | 0.5294117647058824 | 0.44 | 0.6197968749999999 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 3 | obstructed_recall | 0.5555555555555556 | 0.4065040650406504 | 0.7058823529411765 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 3 | clear_recall | 0.5102040816326531 | 0.3999999999999999 | 0.6203703703703703 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 3 | accuracy | 0.4941176470588235 | 0.3991194843168527 | 0.5910664375715922 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 3 | obstructed_recall | 0.7777777777777777 | 0.6666666666666666 | 0.8759689922480621 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 3 | clear_recall | 0.2857142857142857 | 0.1886724090784796 | 0.3904761904761905 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 4 | accuracy | 0.855072463768116 | 0.7990430622009569 | 0.90625 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 4 | obstructed_recall | 0.8 | 0.5 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 4 | clear_recall | 0.8578680203045685 | 0.8011363636363636 | 0.9099537914691944 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 4 | accuracy | 0.9130434782608696 | 0.868020304568528 | 0.9509803921568628 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 4 | obstructed_recall | 0.5 | 0.1818181818181818 | 0.8333333333333334 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 4 | clear_recall | 0.934010152284264 | 0.8917525773195877 | 0.9680851063829788 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 4 | accuracy | 0.8840579710144928 | 0.8341452174890412 | 0.9295154185022028 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 4 | obstructed_recall | 0.8 | 0.5 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 4 | clear_recall | 0.8883248730964467 | 0.837696335078534 | 0.9343434343434344 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 4 | accuracy | 0.8888888888888888 | 0.8394477587108428 | 0.9336283185840708 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 4 | obstructed_recall | 0.8 | 0.5 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 4 | clear_recall | 0.8934010152284264 | 0.8430232558139535 | 0.9381443298969072 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 4 | accuracy | 0.9162640901771336 | 0.8859180035650623 | 0.943396947666032 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 4 | obstructed_recall | 0.5666666666666668 | 0.2857142857142857 | 0.8333333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 4 | clear_recall | 0.934010152284264 | 0.9070789369185386 | 0.9573070607553366 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 4 | accuracy | 0.8727858293075684 | 0.84 | 0.9045457782957784 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 4 | obstructed_recall | 0.7666666666666666 | 0.5277777777777778 | 0.9629629629629628 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 4 | clear_recall | 0.8781725888324873 | 0.8454853219696971 | 0.9111111111111112 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 4 | accuracy | 0.927536231884058 | 0.8981926199598612 | 0.9534054092125652 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 4 | obstructed_recall | 0.4666666666666666 | 0.2222222222222222 | 0.7142857142857143 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 4 | clear_recall | 0.9509306260575296 | 0.9277777777777776 | 0.9711938832772168 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 4 | accuracy | 0.9146537842190016 | 0.883719920865633 | 0.9430894308943087 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 4 | obstructed_recall | 0.2666666666666666 | 0.0476190476190476 | 0.5185185185185185 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 4 | clear_recall | 0.9475465313028764 | 0.9247863247863248 | 0.9680851063829788 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 4 | accuracy | 0.9194847020933976 | 0.8884239888423989 | 0.9472099673202616 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 4 | obstructed_recall | 0.5 | 0.2857142857142857 | 0.7272727272727272 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 4 | clear_recall | 0.9407783417935702 | 0.9145299145299144 | 0.9644679842702262 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 5 | accuracy | 0.6274509803921569 | 0.4693877551020408 | 0.7647288602941172 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 5 | obstructed_recall | 0.6 | 0.3333333333333333 | 0.8235294117647058 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 5 | clear_recall | 0.6451612903225806 | 0.4444444444444444 | 0.8148148148148148 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 5 | accuracy | 0.4509803921568627 | 0.2954545454545454 | 0.6046664626682984 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 5 | obstructed_recall | 0.65 | 0.391304347826087 | 0.8823529411764706 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 5 | clear_recall | 0.3225806451612903 | 0.1515151515151515 | 0.5 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 5 | accuracy | 0.5098039215686274 | 0.3571428571428571 | 0.6590909090909091 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 5 | obstructed_recall | 0.55 | 0.3 | 0.7727272727272727 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 5 | clear_recall | 0.4838709677419355 | 0.2857142857142857 | 0.6842105263157895 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 5 | accuracy | 0.5686274509803921 | 0.4150943396226415 | 0.72 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 5 | obstructed_recall | 0.5 | 0.2307692307692307 | 0.75 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 5 | clear_recall | 0.6129032258064516 | 0.425 | 0.8 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 5 | accuracy | 0.7124183006535948 | 0.5925925925925926 | 0.8238993710691824 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 5 | obstructed_recall | 0.4333333333333333 | 0.2592592592592592 | 0.6190476190476191 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 5 | clear_recall | 0.8924731182795699 | 0.7849304237824161 | 0.9722222222222222 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 5 | accuracy | 0.607843137254902 | 0.497352328909706 | 0.7083333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 5 | obstructed_recall | 0.6833333333333332 | 0.4791666666666667 | 0.8636363636363636 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 5 | clear_recall | 0.5591397849462365 | 0.4333024691358024 | 0.6666666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 5 | accuracy | 0.7058823529411765 | 0.5909090909090909 | 0.8113275909215203 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 5 | obstructed_recall | 0.3833333333333333 | 0.1875 | 0.5740740740740741 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 5 | clear_recall | 0.913978494623656 | 0.85 | 0.96875 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 5 | accuracy | 0.6535947712418301 | 0.5447154471544716 | 0.7588652482269503 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 5 | obstructed_recall | 0.7333333333333334 | 0.5490196078431372 | 0.875 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 5 | clear_recall | 0.6021505376344086 | 0.4649056353003722 | 0.7444444444444445 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 5 | accuracy | 0.5294117647058824 | 0.3958333333333333 | 0.6607142857142857 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 5 | obstructed_recall | 0.6499999999999999 | 0.4666666666666666 | 0.8095238095238094 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 5 | clear_recall | 0.4516129032258065 | 0.28125 | 0.6363636363636364 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 6 | accuracy | 0.9162011173184358 | 0.875 | 0.9550561797752808 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 6 | obstructed_recall | 0.9397590361445785 | 0.9017259627126925 | 0.972972972972973 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 6 | clear_recall | 0.6153846153846154 | 0.2857142857142857 | 0.8571428571428571 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 6 | accuracy | 0.8659217877094972 | 0.7988505747126436 | 0.925925925925926 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 6 | obstructed_recall | 0.9156626506024096 | 0.8618386058230683 | 0.961038961038961 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 6 | clear_recall | 0.2307692307692307 | 0.0 | 0.6 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 6 | accuracy | 0.9441340782122905 | 0.8882971294425295 | 0.9844579616473478 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 6 | obstructed_recall | 0.993975903614458 | 0.98 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 6 | clear_recall | 0.3076923076923077 | 0.0625 | 0.7142857142857143 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 6 | accuracy | 0.9553072625698324 | 0.9226190476190476 | 0.9831460674157304 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 6 | obstructed_recall | 0.9879518072289156 | 0.96875 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 6 | clear_recall | 0.5384615384615384 | 0.1818181818181818 | 0.7857142857142857 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 6 | accuracy | 0.8268156424581005 | 0.7771317829457365 | 0.8740759637188209 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 6 | obstructed_recall | 0.8413654618473897 | 0.7900549496543972 | 0.8888888888888888 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 6 | clear_recall | 0.6410256410256411 | 0.5151515151515151 | 0.8333333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 6 | accuracy | 0.8547486033519552 | 0.8014971040297989 | 0.90414073813158 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 6 | obstructed_recall | 0.893574297188755 | 0.8520449593979006 | 0.9325153374233128 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 6 | clear_recall | 0.358974358974359 | 0.1333333333333333 | 0.6666666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 6 | accuracy | 0.851024208566108 | 0.8079089087417551 | 0.8908045977011495 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 6 | obstructed_recall | 0.8734939759036146 | 0.8353652414536193 | 0.9085740378343116 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 6 | clear_recall | 0.5641025641025641 | 0.4222222222222222 | 0.7878787878787877 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 6 | accuracy | 0.6275605214152701 | 0.5714285714285715 | 0.683230099630532 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 6 | obstructed_recall | 0.6305220883534136 | 0.5720423470065046 | 0.6873706004140786 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 6 | clear_recall | 0.5897435897435898 | 0.375 | 0.8888888888888888 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 6 | accuracy | 0.7355679702048418 | 0.6739526411657559 | 0.7971014492753623 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 6 | obstructed_recall | 0.7590361445783133 | 0.7015235581412052 | 0.815290522628858 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 6 | clear_recall | 0.4358974358974359 | 0.1818181818181818 | 0.7777777777777777 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 7 | accuracy | 0.8956521739130435 | 0.8347808120733021 | 0.9482758620689656 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 7 | obstructed_recall | 0.9090909090909092 | 0.8333333333333334 | 0.9696969696969696 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 7 | clear_recall | 0.8775510204081632 | 0.7727272727272727 | 0.9622641509433962 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 7 | accuracy | 0.8695652173913043 | 0.8016528925619835 | 0.9298245614035088 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 7 | obstructed_recall | 0.8333333333333334 | 0.7291666666666666 | 0.9193548387096774 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 7 | clear_recall | 0.9183673469387756 | 0.8285714285714286 | 0.9827586206896552 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 7 | accuracy | 0.8695652173913043 | 0.7982456140350878 | 0.9307692307692308 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 7 | obstructed_recall | 0.8939393939393939 | 0.8113207547169812 | 0.9615384615384616 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 7 | clear_recall | 0.8367346938775511 | 0.7105263157894737 | 0.9454545454545454 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 7 | accuracy | 0.8695652173913043 | 0.803921568627451 | 0.9285714285714286 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 7 | obstructed_recall | 0.8787878787878788 | 0.7924528301886793 | 0.9523809523809524 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 7 | clear_recall | 0.8571428571428571 | 0.75 | 0.9491525423728814 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 7 | accuracy | 0.8115942028985508 | 0.7507781145631612 | 0.8689484126984127 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 7 | obstructed_recall | 0.7424242424242425 | 0.6613756613756614 | 0.8186274509803922 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 7 | clear_recall | 0.9047619047619048 | 0.8119565217391306 | 0.975609756097561 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 7 | accuracy | 0.8318840579710146 | 0.7737003058103976 | 0.8830409356725147 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 7 | obstructed_recall | 0.8434343434343434 | 0.7692307692307692 | 0.9082125603864736 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 7 | clear_recall | 0.8163265306122449 | 0.7210884353741497 | 0.8965585659551176 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 7 | accuracy | 0.7652173913043478 | 0.6996996996996997 | 0.826560975609756 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 7 | obstructed_recall | 0.6767676767676768 | 0.5936004566210046 | 0.7586206896551723 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 7 | clear_recall | 0.8843537414965987 | 0.7982456140350878 | 0.9513888888888888 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 7 | accuracy | 0.6347826086956522 | 0.5555555555555556 | 0.710915030420354 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 7 | obstructed_recall | 0.4595959595959596 | 0.3725490196078431 | 0.547945205479452 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 7 | clear_recall | 0.8707482993197279 | 0.7749943693693693 | 0.9466666666666668 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 7 | accuracy | 0.6434782608695652 | 0.5677958341093934 | 0.7183908045977012 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 7 | obstructed_recall | 0.5858585858585859 | 0.48356220657277 | 0.6888888888888888 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 7 | clear_recall | 0.7210884353741497 | 0.6136363636363636 | 0.8217087677091687 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 8 | accuracy | 0.7664670658682635 | 0.6976708297444731 | 0.83125 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 8 | obstructed_recall | 0.9655172413793104 | 0.875 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 8 | clear_recall | 0.7246376811594203 | 0.647887323943662 | 0.8 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 8 | accuracy | 0.8203592814371258 | 0.7594902863331754 | 0.8779105218377765 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 8 | obstructed_recall | 0.7586206896551724 | 0.5769230769230769 | 0.9230769230769232 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 8 | clear_recall | 0.8333333333333334 | 0.7686567164179104 | 0.8933360655737704 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 8 | accuracy | 0.7305389221556886 | 0.6568025258323766 | 0.8 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 8 | obstructed_recall | 0.9310344827586208 | 0.8076923076923077 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 8 | clear_recall | 0.6884057971014492 | 0.6060606060606061 | 0.7659574468085106 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 8 | accuracy | 0.7305389221556886 | 0.6568047337278107 | 0.8 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 8 | obstructed_recall | 0.9655172413793104 | 0.875 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 8 | clear_recall | 0.6811594202898551 | 0.6 | 0.7594985771759395 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 8 | accuracy | 0.8263473053892216 | 0.7728119180633147 | 0.8752834467120181 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 8 | obstructed_recall | 0.8390804597701149 | 0.7111111111111111 | 0.9333333333333332 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 8 | clear_recall | 0.8236714975845411 | 0.7626245330012453 | 0.8774509803921569 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 8 | accuracy | 0.5988023952095808 | 0.5333333333333333 | 0.6625514403292181 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 8 | obstructed_recall | 0.896551724137931 | 0.8181818181818182 | 0.9696969696969696 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 8 | clear_recall | 0.5362318840579711 | 0.4677002583979328 | 0.6035370597870597 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 8 | accuracy | 0.8343313373253493 | 0.7829977628635346 | 0.8808080808080808 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 8 | obstructed_recall | 0.9080459770114944 | 0.8214285714285715 | 0.980406162464986 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 8 | clear_recall | 0.818840579710145 | 0.7589716312056737 | 0.8725490196078431 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 8 | accuracy | 0.6746506986027944 | 0.6144578313253012 | 0.7326216445173929 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 8 | obstructed_recall | 0.7471264367816092 | 0.5925925925925926 | 0.8939393939393939 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 8 | clear_recall | 0.6594202898550725 | 0.5941999766245909 | 0.7226282015564347 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 8 | accuracy | 0.5349301397205588 | 0.467077881274488 | 0.5996467391304346 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 8 | obstructed_recall | 0.8045977011494253 | 0.655905017921147 | 0.9310344827586208 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 8 | clear_recall | 0.4782608695652174 | 0.4107505376344085 | 0.5456828703703703 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 9 | accuracy | 0.4814814814814814 | 0.3773584905660377 | 0.5894736842105263 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 9 | obstructed_recall | 0.7777777777777778 | 0.59375 | 0.9310540752351094 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 9 | clear_recall | 0.382716049382716 | 0.2682926829268293 | 0.5053763440860215 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 9 | accuracy | 0.5370370370370371 | 0.4308943089430894 | 0.6435643564356436 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 9 | obstructed_recall | 0.7777777777777778 | 0.6 | 0.925925925925926 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 9 | clear_recall | 0.4567901234567901 | 0.3376623376623376 | 0.5814012814428097 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 9 | accuracy | 0.5555555555555556 | 0.4504504504504504 | 0.6631605800214821 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 9 | obstructed_recall | 0.6296296296296297 | 0.4230769230769231 | 0.8064516129032258 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 9 | clear_recall | 0.5308641975308642 | 0.4073966408268734 | 0.6567270788912578 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 9 | accuracy | 0.5648148148148148 | 0.4607843137254901 | 0.6722725297015358 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 9 | obstructed_recall | 0.6666666666666666 | 0.4666666666666667 | 0.8378378378378378 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 9 | clear_recall | 0.5308641975308642 | 0.4096385542168674 | 0.6571428571428571 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 9 | accuracy | 0.5802469135802469 | 0.4900277777777778 | 0.6720441035055511 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 9 | obstructed_recall | 0.6666666666666666 | 0.5263157894736842 | 0.7866759259259258 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 9 | clear_recall | 0.5514403292181069 | 0.4402495565231414 | 0.6666666666666666 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 9 | accuracy | 0.4907407407407407 | 0.3974358974358974 | 0.5852736812251843 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 9 | obstructed_recall | 0.7530864197530863 | 0.6000000000000001 | 0.8888888888888888 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 9 | clear_recall | 0.4032921810699588 | 0.2980377302436126 | 0.5145682274384498 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 9 | accuracy | 0.4876543209876543 | 0.3955536723163842 | 0.5815637602558753 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 9 | obstructed_recall | 0.5185185185185185 | 0.3333333333333333 | 0.6862745098039215 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 9 | clear_recall | 0.4773662551440329 | 0.3703703703703704 | 0.5894319640564827 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 9 | accuracy | 0.6450617283950617 | 0.5647382920110192 | 0.7272727272727272 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 9 | obstructed_recall | 0.5802469135802469 | 0.4533333333333333 | 0.7171911421911419 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 9 | clear_recall | 0.6666666666666666 | 0.5670480180470513 | 0.7637130801687763 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 9 | accuracy | 0.6018518518518519 | 0.5205992509363296 | 0.6820987654320988 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 9 | obstructed_recall | 0.7037037037037036 | 0.5679012345679012 | 0.8333333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 9 | clear_recall | 0.5679012345679012 | 0.4712605363984674 | 0.6634949759720401 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 10 | accuracy | 0.9836065573770492 | 0.9452054794520548 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 10 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 10 | accuracy | 0.9836065573770492 | 0.9452054794520548 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 10 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 10 | accuracy | 0.9672131147540984 | 0.9152542372881356 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 10 | obstructed_recall | 0.9830508474576272 | 0.9428434065934066 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 10 | accuracy | 0.9672131147540984 | 0.9152542372881356 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 10 | obstructed_recall | 0.9830508474576272 | 0.9428434065934066 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 10 | accuracy | 0.8087431693989071 | 0.7192982456140351 | 0.8908045977011495 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 10 | obstructed_recall | 0.8192090395480226 | 0.7309941520467836 | 0.8978494623655915 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 10 | accuracy | 0.8360655737704917 | 0.7543804824561404 | 0.9080459770114944 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 10 | obstructed_recall | 0.8474576271186441 | 0.768361581920904 | 0.917948717948718 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| ft_MobileNet | 10 | accuracy | 0.7158469945355191 | 0.626865671641791 | 0.798941798941799 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 10 | obstructed_recall | 0.7288135593220338 | 0.641784004505773 | 0.8111163522012579 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 10 | clear_recall | 0.3333333333333333 | 0.0 | 0.6666666666666666 | 8696 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 10 | accuracy | 0.4918032786885246 | 0.3933333333333333 | 0.5918396669012432 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 10 | obstructed_recall | 0.4802259887005649 | 0.3790849673202614 | 0.5833333333333334 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 10 | clear_recall | 0.8333333333333334 | 0.6666666666666666 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 10 | accuracy | 0.5901639344262294 | 0.4779874213836477 | 0.6965174129353233 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 10 | obstructed_recall | 0.5932203389830509 | 0.4787878787878788 | 0.6995422535211265 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 10 | clear_recall | 0.5 | 0.0 | 1.0 | 8696 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 11 | accuracy | 0.9777777777777776 | 0.9496387763738589 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 11 | obstructed_recall | 1.0 | 1.0 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| a_resnet101v2_logistic | 11 | clear_recall | 0.0 | 0.0 | 0.0 | 9495 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 11 | accuracy | 0.9407407407407408 | 0.8916666666666667 | 0.9807722929936304 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 11 | obstructed_recall | 0.9621212121212122 | 0.918032786885246 | 0.9930555555555556 | 10000 | 10000 | exploratory/post hoc |
| b_efficientnetb3_pca_300_linear_svm | 11 | clear_recall | 0.0 | 0.0 | 0.0 | 9495 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 11 | accuracy | 0.9629629629629628 | 0.9285714285714286 | 0.9923664122137404 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 11 | obstructed_recall | 0.9848484848484848 | 0.9612403100775194 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| c_mobilenet_resnet50_pca_300_linear_svm | 11 | clear_recall | 0.0 | 0.0 | 0.0 | 9495 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 11 | accuracy | 0.9703703703703704 | 0.9389274030222776 | 0.9933333333333332 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 11 | obstructed_recall | 0.9924242424242424 | 0.975206611570248 | 1.0 | 10000 | 10000 | exploratory/post hoc |
| d_mobilenet_resnet50_pca_300_linear_svm | 11 | clear_recall | 0.0 | 0.0 | 0.0 | 9495 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 11 | accuracy | 0.9308641975308642 | 0.8972222222222221 | 0.9615384615384616 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 11 | obstructed_recall | 0.9469696969696968 | 0.9193548387096774 | 0.9711286089238844 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB3 | 11 | clear_recall | 0.2222222222222222 | 0.0 | 0.6666666666666666 | 9495 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 11 | accuracy | 0.9604938271604938 | 0.9319715914534829 | 0.9844974074644488 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 11 | obstructed_recall | 0.9797979797979798 | 0.963302752293578 | 0.9928571428571428 | 10000 | 10000 | exploratory/post hoc |
| ft_EfficientNetB4 | 11 | clear_recall | 0.1111111111111111 | 0.0 | 0.3333333333333333 | 9495 | 10000 | exploratory/post hoc |
| ft_MobileNet | 11 | accuracy | 0.9061728395061728 | 0.8512396694214877 | 0.9516908212560388 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 11 | obstructed_recall | 0.9217171717171716 | 0.8693684684684686 | 0.9627104377104376 | 10000 | 10000 | exploratory/post hoc |
| ft_MobileNet | 11 | clear_recall | 0.2222222222222222 | 0.0 | 0.6666666666666666 | 9495 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 11 | accuracy | 0.8296296296296296 | 0.7756399782135075 | 0.8802101672535211 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 11 | obstructed_recall | 0.8434343434343434 | 0.7910041700295938 | 0.8915360046457608 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet101V2 | 11 | clear_recall | 0.2222222222222222 | 0.0 | 0.6666666666666666 | 9495 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 11 | accuracy | 0.928395061728395 | 0.8909795993869246 | 0.9613546532518866 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 11 | obstructed_recall | 0.9469696969696968 | 0.9166666666666666 | 0.97323600973236 | 10000 | 10000 | exploratory/post hoc |
| ft_ResNet50 | 11 | clear_recall | 0.1111111111111111 | 0.0 | 0.3333333333333333 | 9495 | 10000 | exploratory/post hoc |

K-means k=12, n_init=20, seed 1980 sobre DINOv2 normalizado; VIA atribuído ao centroide mais próximo. O cluster 1 não contém imagens livres externas: seu recall de livres é não estimável. Mosaicos sem classes/nomes originais: `results/stage3/error_analysis/mosaics/cluster_00.jpg` até `cluster_11.jpg`, 16 imagens aleatórias por cluster. Nomes do autor: `results/stage3/error_analysis/cluster_names.csv`, preenchidos em 2026-10-13.

Consenso: quatro fixos errados no padrão congelado e maioria das seeds EfficientNetB3 errada; mosaicos `consensus_clear_to_nonclear.jpg` e `consensus_nonclear_to_clear.jpg`, até 60 por tipo, ordenados pela confiança no erro, identificadores por hash e verdade/previsão visíveis. CSV do autor: `results/stage3/error_analysis/error_annotations.csv`, categorias preenchidas em 2026-10-13. Lista original e adendo CATEGORY_ADDENDUM.md validados. Nenhum rótulo alterado.

**Revisão do autor — exploratory/post hoc**

Jefferson Antonio Ribeiro Passerini revisou e preencheu os nomes dos 12 clusters e as categorias dos 148 erros em 2026-10-13. As planilhas foram lidas em UTF-8, com todos os campos de autoria/data preenchidos; filename/error_type preservam exatamente a ordem de consensus_errors.csv. Os nomes de clusters e as categorias de erro foram rascunhados com auxílio de IA e revisados, corrigidos e finalizados pelo autor em 2026-10-13; apenas os registros finais do autor (`cluster_names.csv`, `error_annotations.csv`) são distribuídos.

| cluster | name | reviewer | date |
| --- | --- | --- | --- |
| 0 | Asfalto e calçada em close com meio-fio, faixas e sombras | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 1 | Obstáculos verticais próximos (postes, pilares, lixeiras, bebedouros) | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 2 | Ruas e calçadas largas com horizonte | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 3 | Gramados e vegetação rasteira | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 4 | Piso de concreto liso em close, com juntas e manchas | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 5 | Área de piscina com piso estampado/sextavado e sombras | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 6 | Bordas com árvores, canteiros e meio-fio | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 7 | Ambientes internos com piso cerâmico, móveis e escadas | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 8 | Calçadas de bloquetes e pedra portuguesa | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 10 | Veículos estacionados próximos | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |
| 11 | Mobiliário e objetos diversos (cadeiras, placas, corrimãos, equipamentos) | Jefferson Antonio Ribeiro Passerini | 2026-10-13 |

**Tabela 4.1 — Categorias por tipo de erro — exploratory/post hoc**

n e percentual dentro de cada tipo: 137 clear_to_nonclear (livre → obstruído) e 11 nonclear_to_clear (obstruído → livre). Outro: zero em ambos.

| category | error_type | n | percent_within_type | analysis_scope |
| --- | --- | --- | --- | --- |
| bueiro/grelha | clear_to_nonclear | 24 | 17.518248 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 24 | 17.518248 | exploratory/post hoc |
| cena aberta/horizonte | clear_to_nonclear | 21 | 15.328467 | exploratory/post hoc |
| sem causa aparente | clear_to_nonclear | 16 | 11.678832 | exploratory/post hoc |
| grama/vegetação | clear_to_nonclear | 13 | 9.489051 | exploratory/post hoc |
| baixa luz | clear_to_nonclear | 10 | 7.29927 | exploratory/post hoc |
| junta ou rachadura | clear_to_nonclear | 8 | 5.839416 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 8 | 5.839416 | exploratory/post hoc |
| corredor estreito | clear_to_nonclear | 4 | 2.919708 | exploratory/post hoc |
| piso tátil | clear_to_nonclear | 4 | 2.919708 | exploratory/post hoc |
| faixa ou pintura | clear_to_nonclear | 3 | 2.189781 | exploratory/post hoc |
| sombra | clear_to_nonclear | 2 | 1.459854 | exploratory/post hoc |
| borda de piscina | clear_to_nonclear | 0 | 0.0 | exploratory/post hoc |
| desfoque | clear_to_nonclear | 0 | 0.0 | exploratory/post hoc |
| obstáculo pequeno/distante | clear_to_nonclear | 0 | 0.0 | exploratory/post hoc |
| outro | clear_to_nonclear | 0 | 0.0 | exploratory/post hoc |
| borda de piscina | nonclear_to_clear | 4 | 36.363636 | exploratory/post hoc |
| rótulo questionável | nonclear_to_clear | 3 | 27.272727 | exploratory/post hoc |
| baixa luz | nonclear_to_clear | 2 | 18.181818 | exploratory/post hoc |
| desfoque | nonclear_to_clear | 1 | 9.090909 | exploratory/post hoc |
| obstáculo pequeno/distante | nonclear_to_clear | 1 | 9.090909 | exploratory/post hoc |
| bueiro/grelha | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| cena aberta/horizonte | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| corredor estreito | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| faixa ou pintura | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| grama/vegetação | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| junta ou rachadura | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| mudança de piso/textura | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| outro | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| piso tátil | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| sem causa aparente | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |
| sombra | nonclear_to_clear | 0 | 0.0 | exploratory/post hoc |

**Categorias acrescentadas e motivo — exploratory/post hoc**

A lista válida é a união de error_categories em configs/stage3.json com a seção “Lista completa usada na anotação” do [CATEGORY_ADDENDUM.md](../results/stage3/error_analysis/CATEGORY_ADDENDUM.md). O adendo foi lido do disco; a configuração registrada permanece intacta. As seis categorias acrescentadas são exclusivas da anotação exploratória:

| Categoria | Motivo |
| --- | --- |
| `bueiro/grelha` | É o padrão de erro mais frequente entre as livres classificadas como obstruídas. Antes estava em `outro`. |
| `piso tátil` | Elemento de acessibilidade confundido com obstáculo; relevante para o usuário-alvo. Incluída na revisão do autor de 2026-10-13. |
| `sem causa aparente` | Registra separadamente os erros que não se explicam pelo conteúdo visível da imagem, em vez de diluí-los em `outro`. Incluída na revisão do autor de 2026-10-13. |
| `borda de piscina` | Padrão principal das obstruídas classificadas como livres; corresponde ao grupo visual 5, quase ausente no VIA (51 externas contra 1 no VIA). Incluída na revisão do autor de 2026-10-13. |
| `corredor estreito` | Padrão recorrente de erro ligado ao grupo visual 7 (ambientes internos). Incluída na revisão do autor de 2026-10-13. |
| `cena aberta/horizonte` | Corresponde ao grupo visual 2, sub-representado no VIA (188 externas contra 23 no VIA). Antes estava em `outro`. |

**Categorias por cluster — exploratory/post hoc**

Junção apenas com cluster_assignments.csv no domínio externo; cada erro de consenso contribui uma vez. Os nomes vêm da revisão do autor.

| category | error_type | cluster | name | n | analysis_scope |
| --- | --- | --- | --- | --- | --- |
| baixa luz | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 5 | exploratory/post hoc |
| baixa luz | clear_to_nonclear | 6 | Bordas com árvores, canteiros e meio-fio | 1 | exploratory/post hoc |
| baixa luz | clear_to_nonclear | 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 4 | exploratory/post hoc |
| baixa luz | nonclear_to_clear | 4 | Piso de concreto liso em close, com juntas e manchas | 1 | exploratory/post hoc |
| baixa luz | nonclear_to_clear | 6 | Bordas com árvores, canteiros e meio-fio | 1 | exploratory/post hoc |
| borda de piscina | nonclear_to_clear | 5 | Área de piscina com piso estampado/sextavado e sombras | 4 | exploratory/post hoc |
| bueiro/grelha | clear_to_nonclear | 4 | Piso de concreto liso em close, com juntas e manchas | 1 | exploratory/post hoc |
| bueiro/grelha | clear_to_nonclear | 6 | Bordas com árvores, canteiros e meio-fio | 1 | exploratory/post hoc |
| bueiro/grelha | clear_to_nonclear | 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 22 | exploratory/post hoc |
| cena aberta/horizonte | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 21 | exploratory/post hoc |
| corredor estreito | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 1 | exploratory/post hoc |
| corredor estreito | clear_to_nonclear | 7 | Ambientes internos com piso cerâmico, móveis e escadas | 2 | exploratory/post hoc |
| corredor estreito | clear_to_nonclear | 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 1 | exploratory/post hoc |
| desfoque | nonclear_to_clear | 4 | Piso de concreto liso em close, com juntas e manchas | 1 | exploratory/post hoc |
| faixa ou pintura | clear_to_nonclear | 0 | Asfalto e calçada em close com meio-fio, faixas e sombras | 2 | exploratory/post hoc |
| faixa ou pintura | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 1 | exploratory/post hoc |
| grama/vegetação | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 4 | exploratory/post hoc |
| grama/vegetação | clear_to_nonclear | 3 | Gramados e vegetação rasteira | 8 | exploratory/post hoc |
| grama/vegetação | clear_to_nonclear | 8 | Calçadas de bloquetes e pedra portuguesa | 1 | exploratory/post hoc |
| junta ou rachadura | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 5 | exploratory/post hoc |
| junta ou rachadura | clear_to_nonclear | 4 | Piso de concreto liso em close, com juntas e manchas | 3 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 0 | Asfalto e calçada em close com meio-fio, faixas e sombras | 1 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 14 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 3 | Gramados e vegetação rasteira | 1 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 8 | Calçadas de bloquetes e pedra portuguesa | 7 | exploratory/post hoc |
| mudança de piso/textura | clear_to_nonclear | 11 | Mobiliário e objetos diversos (cadeiras, placas, corrimãos, equipamentos) | 1 | exploratory/post hoc |
| obstáculo pequeno/distante | nonclear_to_clear | 7 | Ambientes internos com piso cerâmico, móveis e escadas | 1 | exploratory/post hoc |
| piso tátil | clear_to_nonclear | 8 | Calçadas de bloquetes e pedra portuguesa | 2 | exploratory/post hoc |
| piso tátil | clear_to_nonclear | 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 2 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 2 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 4 | Piso de concreto liso em close, com juntas e manchas | 1 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 5 | Área de piscina com piso estampado/sextavado e sombras | 1 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 8 | Calçadas de bloquetes e pedra portuguesa | 1 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 9 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 1 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 10 | Veículos estacionados próximos | 1 | exploratory/post hoc |
| rótulo questionável | clear_to_nonclear | 11 | Mobiliário e objetos diversos (cadeiras, placas, corrimãos, equipamentos) | 1 | exploratory/post hoc |
| rótulo questionável | nonclear_to_clear | 7 | Ambientes internos com piso cerâmico, móveis e escadas | 3 | exploratory/post hoc |
| sem causa aparente | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 16 | exploratory/post hoc |
| sombra | clear_to_nonclear | 2 | Ruas e calçadas largas com horizonte | 1 | exploratory/post hoc |
| sombra | clear_to_nonclear | 5 | Área de piscina com piso estampado/sextavado e sombras | 1 | exploratory/post hoc |

**Resumo dos clusters nomeados — exploratory/post hoc**

cluster_summary_named junta cluster_counts, os nomes e as estimativas já existentes em cluster_metrics.csv. As porcentagens de livres são frações × 100; acurácia e recall de obstruídos estão em frações. FT EfficientNetB3 usa a média das métricas das três seeds individuais, conforme a tabela de origem. Nenhuma métrica foi reestimada.

| cluster | external_images | via_images | external_clear_fraction | via_clear_fraction | name | external_clear_percent | via_clear_percent | fixed_b_accuracy | fixed_b_obstructed_recall | ft_efficientnetb3_accuracy | ft_efficientnetb3_obstructed_recall | ft_metric_aggregation | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 127 | 68 | 0.84252 | 0.647059 | Asfalto e calçada em close com meio-fio, faixas e sombras | 84.251969 | 64.705882 | 0.850394 | 0.9 | 0.897638 | 0.75 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 1 | 153 | 19 | 0.0 | 0.0 | Obstáculos verticais próximos (postes, pilares, lixeiras, bebedouros) | 0.0 | 0.0 | 1.0 | 1.0 | 0.947712 | 0.947712 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 2 | 188 | 23 | 0.925532 | 0.695652 | Ruas e calçadas largas com horizonte | 92.553191 | 69.565217 | 0.537234 | 0.928571 | 0.304965 | 0.97619 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 3 | 85 | 12 | 0.576471 | 0.166667 | Gramados e vegetação rasteira | 57.647059 | 16.666667 | 0.752941 | 0.972222 | 0.784314 | 0.833333 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 4 | 207 | 37 | 0.951691 | 0.810811 | Piso de concreto liso em close, com juntas e manchas | 95.169082 | 81.081081 | 0.913043 | 0.5 | 0.916264 | 0.566667 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 5 | 51 | 1 | 0.607843 | 0.0 | Área de piscina com piso estampado/sextavado e sombras | 60.784314 | 0.0 | 0.45098 | 0.65 | 0.712418 | 0.433333 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 6 | 179 | 50 | 0.072626 | 0.12 | Bordas com árvores, canteiros e meio-fio | 7.26257 | 12.0 | 0.865922 | 0.915663 | 0.826816 | 0.841365 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 7 | 115 | 43 | 0.426087 | 0.860465 | Ambientes internos com piso cerâmico, móveis e escadas | 42.608696 | 86.046512 | 0.869565 | 0.833333 | 0.811594 | 0.742424 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 8 | 167 | 50 | 0.826347 | 0.66 | Calçadas de bloquetes e pedra portuguesa | 82.634731 | 66.0 | 0.820359 | 0.758621 | 0.826347 | 0.83908 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 9 | 108 | 24 | 0.75 | 0.291667 | Elementos no piso: grelhas, bueiros, piso tátil, escadas e faixas | 75.0 | 29.166667 | 0.537037 | 0.777778 | 0.580247 | 0.666667 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 10 | 61 | 2 | 0.032787 | 0.0 | Veículos estacionados próximos | 3.278689 | 0.0 | 0.983607 | 1.0 | 0.808743 | 0.819209 | mean of 3 individual-seed metrics | exploratory/post hoc |
| 11 | 135 | 13 | 0.022222 | 0.0 | Mobiliário e objetos diversos (cadeiras, placas, corrimãos, equipamentos) | 2.222222 | 0.0 | 0.940741 | 0.962121 | 0.930864 | 0.94697 | mean of 3 individual-seed metrics | exploratory/post hoc |

**Leitura dos erros — exploratory/post hoc**

Bueiro/grelha (24), mudança de piso/textura (24) e cena aberta/horizonte (21) somam **69/137 (50,36%)** dos falsos “obstruído”. Há **16/137 (11,68%) sem causa aparente**. Essas categorias descrevem o conteúdo revisto pelo autor; não demonstram causas do erro. Borda de piscina é o padrão principal dos falsos “livre”: **4/11 (36,36%)**, seguido de rótulo questionável (3), baixa luz (2), desfoque (1) e obstáculo pequeno/distante (1).

A tabela por cluster localiza todos os 21 casos de cena aberta/horizonte e os 16 sem causa aparente no **cluster 2** (188 externas, 23 VIA); esse cluster também reúne 14/24 mudanças de piso/textura. O **cluster 9** (108 externas, 24 VIA), nomeado por seus elementos no piso, concentra 22/24 bueiros/grelhas. Os quatro falsos “livre” de borda de piscina estão no **cluster 5** (51 externas, somente 1 VIA). Essas concentrações são descritivas e post hoc; os nomes não foram usados para ajustar modelos, limiares ou clusters.

**5. Pacote cego — exploratory/post hoc**

| images | included_files | key_in_archive | responses_analysed | zip | zip_sha256 | key_kept_outside_package | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 102 | False | False | external-data/blind_relabel/blind_relabel_package.zip | 3882a5fe7c9db2e6440fbf74d628eaa383d06ce4a1b65ba39829bfb66816f27e | external-data/blind_relabel/key.csv | exploratory/post hoc |

Nota: o campo `responses_analysed=False` em `package_manifest` reflete o estado da preparação do pacote. As respostas foram analisadas em **2026-10-13**, conforme [status.json](../results/stage3/blind_relabel/status.json).

Zip local: `external-data/blind_relabel/blind_relabel_package.zip`, somente images/, instructions.md e responses_template.csv (100 itens embaralhados seed 1980). `key.csv` fica fora do zip. A análise oficial de concordância foi rodada com as respostas reais de Fabricio Breve. scripts/analyze_blind_relabel.py valida cobertura/rótulos/confiança e calcula concordância, Wilson, kappa com bootstrap por cena e discordâncias.

**Rotulagem às cegas — exploratory/post hoc**

Rotulador independente: Fabricio Breve; respostas recebidas em 2026-10-13, em [responses_breve.csv](../external-data/blind_relabel/responses_breve.csv). O script original scripts/analyze_blind_relabel.py foi executado sem alterações, com 10.000 iterações e seed 1980. Os três CSVs foram regenerados nesta máquina, junto com seus fragmentos LaTeX, e conferidos contra os valores esperados (tolerância 1e-6).

Concordância: **0.980000 (98/100; 98%)**. IC 95% de Wilson: **0.929988–0.994498**; IC 95% por bootstrap de cenas: **0.949495–1.000000**. κ de Cohen: **0.959612**; IC 95% por cenas: **0.897748–1.000000**.

| agreement | agreement_wilson_low | agreement_wilson_high | agreement_cluster_ci_low | agreement_cluster_ci_high | cohen_kappa | kappa_ci_low | kappa_ci_high | bootstrap_iterations | synthetic | analysis_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.98 | 0.929988209271456 | 0.9944980324498376 | 0.9494949494949496 | 1.0 | 0.9596122778675282 | 0.8977483990910969 | 1.0 | 10000 | False | exploratory/post hoc |

| Original (0=obstruído; 1=livre) | Cego 0 (obstruído) | Cego 1 (livre) | analysis_scope |
| --- | --- | --- | --- |
| 0 | 44 | 2 | exploratory/post hoc |
| 1 | 0 | 54 | exploratory/post hoc |

| item | filename | group_id | original_label | blind_label | analysis_scope |
| --- | --- | --- | --- | --- | --- |
| item_042.jpg | nonclear.0256.jpg | scene_0831 | 0 | 1 | exploratory/post hoc |
| item_056.jpg | nonclear.0258.jpg | scene_0833 | 0 | 1 | exploratory/post hoc |

As duas discordâncias são item_042.jpg (nonclear.0256.jpg, scene_0831; confiança 3) e item_056.jpg (nonclear.0258.jpg, scene_0833; confiança 2): ambas original 0 → cego 1. A leitura qualitativa indicada para estes casos é obstáculo lateral com faixa central transitável, a mesma ambiguidade de regra observada em “rótulo questionável” e “corredor estreito”. Essa interpretação não é uma justificativa escrita pelo rotulador: o campo notes das duas respostas está vazio. Nenhum rótulo foi corrigido.

SHA-256 das respostas: `b46b69264fcbbaf56f13a52653b8c541bfb3f26c2b5a0168274b28123ba09b38`. SHA-256 de agreement.csv: `961267a6436c383ca9806edb88668523f28b936edbeb86c7c5631d154725f2e8`. Os campos de autoria, recebimento e execução já existentes em status.json foram preservados.

**Regra de rotulagem a explicitar no artigo — exploratory/post hoc**

Casos-limite a explicitar na definição de “obstáculo no caminho à frente”, mantendo os rótulos atuais:

- Escadas e rampas com corrimão: delimitar a passagem avaliada e o papel do corrimão lateral.
- Bueiros/grelhas: explicitar como se distingue um elemento rente ao piso de uma obstrução.
- Piso tátil: explicitar o tratamento da sinalização de acessibilidade no caminho.
- Obstáculo lateral fora da faixa central: explicitar os limites da faixa transitável.
- Corredores estreitos: explicitar o critério aplicado quando há objetos próximos e centro livre.

Este relatório registra questões para a redação futura; não propõe troca de rótulos nem altera o manuscrito ou a carta de resposta.

**6. Verificações e entrega — exploratory/post hoc**

206 passed, 1 skipped, 46 warnings, 7 subtests passed in 167.69s (0:02:47)

O teste opcional de integração real em GPU usa --run-gpu e permanece ignorado na suíte padrão. Smoke completo em CPU, na ordem solicitada: `results/stage3/verification/smoke.json`.

Comandos: `.venv/bin/python scripts/analyze_stage3.py --dry-run`; `--smoke`; execução completa `--section all`; testes `python -m pytest -q`. O runner scripts/run_stage3.py registra logs, repete a conferência de hashes após testes e monta este relatório. Artefatos locais de pipelines/zip não entram no Git; tabelas, figuras, scripts, metadados e hashes são versionados. Fechamento concluído após revisão do autor e análise cega. Reproduzir com `.venv/bin/python scripts/build_stage3_report.py --mode final`. Nenhum rótulo, modelo, limiar ou predição foi alterado.
