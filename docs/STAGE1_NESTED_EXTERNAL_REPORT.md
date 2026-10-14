# Primeira etapa: nested CV e conjunto externo

Entrega de 06/10/2026, no repositório `cnn-obstacle-detection-reproducibility`, a partir do commit `4a5a1a5`. Foram lidos `PROMPT_NESTED_CV.md` e depois `PROMPT_NESTED_CV_EXTERNAL.md`; o complemento prevaleceu. A etapa terminou em `paused_for_author_review`, sem nested CV completa, treinamento completo dos modelos finais, treinamento com imagens externas ou classificação de imagens externas.

A primeira alteração foi incluir `external-data/via-dataset-extended/` no `.gitignore`. As 1.929 imagens que já estavam versionadas foram retiradas somente do índice: todas continuam no disco. O histórico Git anterior foi preservado. O protocolo `docs/EXTERNAL_TEST_PROTOCOL.md` e `configs/external_test.json` entram juntos no commit desta entrega, antes de qualquer classificação externa. O hash desse commit consta na resposta da entrega; também pode ser consultado com `git log -1 --format=%H -- docs/STAGE1_NESTED_EXTERNAL_REPORT.md`.

Foram criados os executores `src/nested_cv.py` e `src/external_test.py`; os scripts `scripts/clean_external_dataset.py`, `scripts/analyze_nested_cv.py`, `scripts/analyze_external_test.py` e `scripts/verify_stage1_nested_external.py`; as configurações `configs/nested_cv.json` e `configs/external_test.json`; os testes `tests/test_nested_cv.py` e `tests/test_external_test.py`; este relatório e o protocolo externo. Os módulos comuns novos são `src/selection.py`, `src/study_io.py`, `src/study_statistics.py`, `src/study_features.py` e `src/study_estimates.py`.

A seleção do artigo foi extraída para uma função pura comum. As seis funções originais têm AST idêntica à versão anterior. `scripts/build_manuscript_tables.py` passou a consumir essa seleção, e README/PROVENANCE receberam os comandos e as decisões registradas. Configurações anteriores, folds originais, resultados do artigo, resultados de fine-tuning, grupos VIA e código de treinamento existente foram preservados. O inventário verificou **4.103 arquivos protegidos sem alterações e 69 versões de pacotes de treinamento iguais às iniciais**.

Decisões de execução estão pré-especificadas no protocolo: tentativas PCA/UMAP inviáveis no treino interno são registradas e não selecionadas, sem truncar dimensões; outros erros interrompem o job. SVMs originais permanecem sem probabilidades; o adaptador externo separado usa somente OOF do VIA. Grupos mistos são reamostrados inteiros. A avaliação exige protocolo/configuração em commit, limpeza aprovada, OOF completo e modelos/limiares congelados por hashes. Artefatos de smoke não satisfazem essas condições.

**Verificações executadas**

| Verificação | Esperado | Obtido |
|---|---|---|
| Pytest completo, antigos e novos | Todos os testes comuns aprovados | **160 passed, 1 skipped, 7 subtests passed**, 46 warnings; 152,77 s |
| Regressão do finalista A | ResNet101V2 + logistic | `a_resnet101v2_logistic` |
| Regressão do finalista B | EfficientNetB3 + PCA300 + linear SVM | `b_efficientnetb3_pca_300_linear_svm` |
| Regressão do finalista C | MobileNet + ResNet50 + PCA300 + linear SVM | `c_mobilenet_resnet50_pca_300_linear_svm` |
| Regressão do finalista D | PCA300 por CNN, mesmo par + linear SVM | `d_mobilenet_resnet50_pca_300_linear_svm` |
| Regressão do vencedor global | B, conforme ranking arquivado | `b_efficientnetb3_pca_300_linear_svm` |
| Nested não agrupada, `--smoke --device cpu` | 1 externo, 2 internos, 20 configurações; decisões/métricas válidas | 40 tentativas válidas; 35 imagens externas VIA; 175 linhas de predição; 10 linhas de métricas recomputadas |
| Nested agrupada, `--smoke --device cpu` | Mesmo diagnóstico, com grupos disjuntos e folds verificados | 40 tentativas válidas; 34 imagens externas VIA; 170 linhas de predição; 10 linhas de métricas recomputadas |
| Limpeza, `--smoke --device cpu` | Inventário/hash completos; grafo de pequeno subconjunto; nenhuma classificação | Concluído em 3,66 s; 8 VIA + 8 novas no grafo; relatório não aprovado |
| Modelos finais, `--smoke --device cpu` | 4 fixos + 5 CNNs; ajuste diagnóstico, gravação/recarga | 9 painéis recarregados; CNNs com 1 época, seed 1980, treino 8/validação 4; `frozen=false` |
| Cinco `--dry-run` | Sem ajustar/predizer; tempo e disco por etapa | Cinco JSONs registrados; estimativas na tabela abaixo |
| Análises dos dois diagnósticos nested | Métricas recalculadas; CSV/LaTeX | Concluídas nos diretórios `nested_cpu` e `nested_grouped_cpu` |
| Avaliação/análise externa diagnóstica | Somente números sintéticos | 40 fixtures × 19 painéis = 760 linhas; 76 linhas de métricas; CSV/LaTeX e confiabilidade; zero imagens externas classificadas |
| Curva de aprendizado diagnóstica | Fixtures, cenas disjuntas, subconjuntos aninhados | Concluída; tamanhos 0/6/10/20; zero imagens externas treinadas/classificadas |
| Travas antes do commit | Recusar protocolo não registrado e limpeza não aprovada | Ambas recusas registradas em `gate_checks_before_commit.json` |
| Preservação do estudo existente | Zero arquivos/pacotes protegidos alterados | 4.103 arquivos e 69 versões preservados |
| Limite desta entrega | Nenhuma execução científica completa nem classificação externa real | Zero; limpeza permanece sem aprovação do autor |

O teste GPU é explicitamente opt-in em `tests/conftest.py` e corresponde ao único skip. Os diagnósticos desta entrega foram em CPU. Nos dois smokes nested, os classificadores foram ajustados em CPU sobre features canônicas já existentes, extraídas anteriormente em GPU; essa origem está nos metadados do cache. Esses diagnósticos não representam uma repetição CPU da extração canônica nem resultados científicos completos.

O log completo está em [`final_pytest.log`](../results/stage1_verification/final_pytest.log). A auditoria adicional registra 45 comparações técnicas esperado × obtido em [`expected_vs_obtained.csv`](../results/stage1_verification/expected_vs_obtained.csv), com detalhes, hashes e estimativas em [`verification_report.json`](../results/stage1_verification/verification_report.json). A regressão também é um teste da suíte: `test_article_selection_full_search_regression`.

**Estimativas dos comandos `--dry-run`**

| Etapa | Trabalho previsto | Tempo estimado | Disco adicional estimado | Registro |
|---|---|---:|---:|---|
| Nested não agrupada | 12.656 × 10 externos × 5 internos = 632.800 tentativas | **900,69–2.702,08 h** (37,53–112,59 dias) | 12,59–22,51 GiB | `dry_nested.json` |
| Nested agrupada | Mesma grade e número de tentativas | **900,69–2.702,08 h** (37,53–112,59 dias) | 12,59–22,51 GiB | `dry_nested_grouped.json` |
| Modelos finais | 4 fixos + 15 FT (5 CNNs × 3 seeds) | 0,84–2,32 h | 3–12 GiB | `dry_models.json` |
| Avaliação externa e análise | 19 painéis, principal provisório com 1.568 imagens | 0,18–0,49 h (10,54–29,18 min) | 0,02–0,15 GiB | `dry_external.json` |
| Curva de aprendizado | 57 novos ajustes; ponto 0% reutiliza modelos finais | 5,87–16,26 h | 8–35 GiB | `dry_learning.json` |

Os cinco registros estão em `results/stage1_verification/`. São intervalos de planejamento derivados dos custos existentes, com incerteza de early stopping, cache, CPU e folds agrupados; não são tempos medidos de novas execuções completas. As estimativas de CNNs/avaliação usam as medições anteriores em GPU RTX 4070 Ti; CPU será mais lenta. Cada nested tem referência serial de 1.536,23 h: 1.466,08 h de preparação e 70,14 h de classificadores. O cenário usa dois jobs de classificadores e oito workers Relief-F; rankings são reutilizados por treino/extrator e entre B/D. O cache não elimina o custo de recomputar rankings nos treinos internos diferentes. O número máximo previsto é 1.898.400 arquivos pequenos por nested, considerado na estimativa de disco. A grade não foi reduzida para melhorar a previsão. O cache de features pode ser compartilhado entre as versões, portanto somar os dois limites de disco é conservador.

A estimativa FT irrestrita da curva é 8,97 h, abaixo da regra de 48 h: as cinco arquiteturas permanecem previstas. Aprovação/rejeição das ligações de cena pode alterar a população externa e deve preceder uma nova estimativa para a etapa posterior. Downloads e ambientes auxiliares não estão no disco adicional da tabela.

**Limpeza real concluída até a pausa**

| Contagem | Esperado | Obtido |
|---|---|---|
| Bruto | 1.929 | 1.929: 1.026 clear, 903 nonclear |
| Correspondências exatas com VIA | 350, incluindo 4 cópias com classe trocada | 350, com os 4 conflitos registrados |
| Duplicatas exatas internas adicionais | 3 | 3 |
| Imagens novas após hash | **1.576: 844 clear, 732 nonclear** | **1.576: 844 clear, 732 nonclear** |
| Hashes repetidos nas novas ou presentes no VIA | 0 | 0 |
| Sobreposições automáticas diretas de cena | Quantificar, excluir provisoriamente e revisar | **8 imagens, 9 ligações**, todas pelo critério pHash; nenhuma ligação nova–VIA confirmada pelo ramo semântico + geometria |
| Principal depois da exclusão automática de cena | Novas menos as sinalizadas | **1.568 provisórias: 836 clear, 732 nonclear** |
| Agrupamento apenas das novas | Cobrir as 1.576, inclusive sinalizadas | **1.378 grupos**, cobrindo todas as novas |
| Auditoria de rótulos | 100, estratificação proporcional, seed 1980 | **100: 54 clear, 46 nonclear** |
| Mosaicos para revisão | Sobreposições sem nomes de classe; amostra de 100 | 2 de sobreposição + 5 de auditoria |
| Aprovação e predições externas | Parar antes da aprovação/predição | `approved_by=null`, `audit_completed=false`, zero classificações externas |

A união usa 1.918 imagens únicas (342 VIA + 1.576 novas); o pipeline avaliou 8.129 candidatos geométricos na união e 6.401 nas novas. Reutilizou as funções V2 e a configuração original. O encoder DINOv2 está fixado na revisão `ed25f3a31f01632728cabb09d1542f84ab7b0056`, em CPU, com hash de pesos no relatório. A limpeza final com cache levou 327,84 s (5 min 28 s); inventários, grafos/embeddings e mosaicos ocupam aproximadamente 21,4 MiB lógicos, sem contar as imagens brutas nem o ambiente auxiliar.

As oito sinalizações ainda **não são confirmação visual de sobreposição**. Inspeção assistida do primeiro mosaico encontrou pisos/cenas possivelmente diferentes. Os limiares foram mantidos; somente o autor decide quais ligações rejeitar. Para `clear.0442.jpg` há duas ligações: rejeitar uma só não remove a exclusão se a outra continuar aceita. As nove ligações e respectivas medidas estão em [`external_excluded_scene_overlap.csv`](../external-data/external_excluded_scene_overlap.csv).

Os arquivos de limpeza são `external_manifest_raw.csv`, `external_excluded_exact.csv`, `external_manifest_clean.csv`, `external_excluded_scene_overlap.csv`, `external_groups.csv`, `label_audit_sample.csv`, `label_corrections.csv`, `scene_overlap_rejections.csv`, `cleaning_report.json` e `identity.json`, todos em `external-data/`. Os diretórios `scene_union/` e `scene_new/` preservam pares, grupos e metadados; a união também preserva embeddings. Não houve classificação das externas: embeddings/pHash/RANSAC são somente detecção de correspondências de cena.

Mosaicos de sobreposição, com identificadores por hash e sem nomes de classe:

- [scene_overlap_001.png](../external-data/review/scene_overlap/scene_overlap_001.png)
- [scene_overlap_002.png](../external-data/review/scene_overlap/scene_overlap_002.png)

Mosaicos da auditoria de rótulos, cobrindo as 100 imagens:

- [label_audit_001.png](../external-data/review/label_audit/label_audit_001.png)
- [label_audit_002.png](../external-data/review/label_audit/label_audit_002.png)
- [label_audit_003.png](../external-data/review/label_audit/label_audit_003.png)
- [label_audit_004.png](../external-data/review/label_audit/label_audit_004.png)
- [label_audit_005.png](../external-data/review/label_audit/label_audit_005.png)

Para a revisão, registrar ligações rejeitadas em [`scene_overlap_rejections.csv`](../external-data/scene_overlap_rejections.csv) e correções de rótulo em [`label_corrections.csv`](../external-data/label_corrections.csv), com justificativa, revisor e data. A amostra fixa está em [`label_audit_sample.csv`](../external-data/label_audit_sample.csv). Discordância maior que 5% na auditoria bloqueia a sequência. A aprovação não foi preenchida automaticamente e nenhum resultado diagnóstico foi promovido a modelo final.

**Comandos usados para as verificações**

Executar no diretório do repositório. O ambiente auxiliar de imagens fica no cache ignorado; nenhuma dependência foi instalada no ambiente de treino `.venv`.

```bash
PYTHONPATH=cache/scene-groups-venv/lib/python3.12/site-packages .venv/bin/python -m pytest -q --disable-warnings

.venv/bin/python src/nested_cv.py --smoke --device cpu --output-dir results/stage1_verification/nested_cpu
.venv/bin/python src/nested_cv.py --smoke --device cpu --groups data/groups.csv --output-dir results/stage1_verification/nested_grouped_cpu
.venv/bin/python src/external_test.py --stage final-models --smoke --device cpu --output-dir results/stage1_verification/models_cpu
cache/scene-groups-venv/bin/python scripts/clean_external_dataset.py --smoke --device cpu

.venv/bin/python src/nested_cv.py --dry-run --estimate-output results/stage1_verification/dry_nested.json
.venv/bin/python src/nested_cv.py --groups data/groups.csv --dry-run --estimate-output results/stage1_verification/dry_nested_grouped.json
.venv/bin/python src/external_test.py --stage final-models --dry-run --estimate-output results/stage1_verification/dry_models.json
.venv/bin/python src/external_test.py --stage evaluate --dry-run --estimate-output results/stage1_verification/dry_external.json
.venv/bin/python src/external_test.py --stage learning-curve --dry-run --estimate-output results/stage1_verification/dry_learning.json

.venv/bin/python scripts/verify_stage1_nested_external.py
```

Os modelos/checkpoints binários dos diagnósticos ficam locais e ignorados; manifests, custos, histories, logs e métricas finais são versionados. Tentativas intermediárias foram preservadas em `cache/stage1_attempts/`, sem excluir arquivos. Os CSVs com escores externos em `results/external_evaluation_smoke/` são explicitamente fixtures numéricas sintéticas, não predições sobre o conjunto externo.
