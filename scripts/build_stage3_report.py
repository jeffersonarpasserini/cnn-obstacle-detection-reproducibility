#!/usr/bin/env python3
"""Assemble post-test outputs and verify author fields; no fitting/predictions."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.stage3_io import SCOPE,table
from src.study_io import write_json,sha256
from src.stage3_review import validate_author_review,validate_blind_analysis,build_review_tables
from scripts.build_stage2_report import markdown


def read(folder,name):
    return pd.read_csv(ROOT/'results/stage3'/folder/'manuscript_tables'/f'{name}.csv')


def author_review_lines(clusters,review_tables):
    addendum=(ROOT/'results/stage3/error_analysis/CATEGORY_ADDENDUM.md').read_text(encoding='utf-8')
    added=addendum.split('## Categorias acrescentadas\n',1)[1].split('\n## ',1)[0]
    rows=[line.strip().strip('|').split('|') for line in added.splitlines() if line.startswith('|')]
    reasons=pd.DataFrame([{'Categoria':cells[0].strip(),'Motivo':cells[2].strip()}
                          for cells in rows[2:]])
    return [
        '**Revisão do autor — exploratory/post hoc**','',
        'Jefferson Antonio Ribeiro Passerini revisou e preencheu os nomes dos 12 clusters e as categorias '
        'dos 148 erros em 2026-10-13. As planilhas foram lidas em UTF-8, com todos os campos de autoria/data '
        'preenchidos; filename/error_type preservam exatamente a ordem de consensus_errors.csv. '
        'Os nomes de clusters e as categorias de erro foram rascunhados com auxílio de IA e revisados, corrigidos e finalizados pelo autor em 2026-10-13; apenas os registros finais do autor (`cluster_names.csv`, `error_annotations.csv`) são distribuídos.','',

        markdown(clusters[['cluster','name','reviewer','date']]),'',
        '**Tabela 4.1 — Categorias por tipo de erro — exploratory/post hoc**','',
        'n e percentual dentro de cada tipo: 137 clear_to_nonclear (livre → obstruído) e '
        '11 nonclear_to_clear (obstruído → livre). Outro: zero em ambos.','',
        markdown(review_tables['error_category_counts'].round({'percent_within_type':6})), '',
        '**Categorias acrescentadas e motivo — exploratory/post hoc**','',
        'A lista válida é a união de error_categories em configs/stage3.json com a seção '
        '“Lista completa usada na anotação” do '
        '[CATEGORY_ADDENDUM.md](../results/stage3/error_analysis/CATEGORY_ADDENDUM.md). '
        'O adendo foi lido do disco; a configuração registrada permanece intacta. '
        'As seis categorias acrescentadas são exclusivas da anotação exploratória:','',
        markdown(reasons),'',
        '**Categorias por cluster — exploratory/post hoc**','',
        'Junção apenas com cluster_assignments.csv no domínio externo; cada erro de consenso '
        'contribui uma vez. Os nomes vêm da revisão do autor.','',
        markdown(review_tables['error_category_by_cluster']),'',
        '**Resumo dos clusters nomeados — exploratory/post hoc**','',
        'cluster_summary_named junta cluster_counts, os nomes e as estimativas já existentes em '
        'cluster_metrics.csv. As porcentagens de livres são frações × 100; acurácia e recall de '
        'obstruídos estão em frações. FT EfficientNetB3 usa a média das métricas das três seeds '
        'individuais, conforme a tabela de origem. Nenhuma métrica foi reestimada.','',
        markdown(review_tables['cluster_summary_named'].round(6)),'',
        '**Leitura dos erros — exploratory/post hoc**','',
        'Bueiro/grelha (24), mudança de piso/textura (24) e cena aberta/horizonte (21) somam '
        '**69/137 (50,36%)** dos falsos “obstruído”. Há **16/137 (11,68%) sem causa aparente**. '
        'Essas categorias descrevem o conteúdo revisto pelo autor; não demonstram causas do erro. '
        'Borda de piscina é o padrão principal dos falsos “livre”: **4/11 (36,36%)**, '
        'seguido de rótulo questionável (3), baixa luz (2), desfoque (1) e obstáculo pequeno/distante (1).','',
        'A tabela por cluster localiza todos os 21 casos de cena aberta/horizonte e os 16 sem causa '
        'aparente no **cluster 2** (188 externas, 23 VIA); esse cluster também reúne 14/24 mudanças '
        'de piso/textura. O **cluster 9** (108 externas, 24 VIA), nomeado por seus elementos no piso, '
        'concentra 22/24 bueiros/grelhas. Os quatro falsos “livre” de borda de piscina estão no '
        '**cluster 5** (51 externas, somente 1 VIA). Essas concentrações são descritivas e post hoc; '
        'os nomes não foram usados para ajustar modelos, limiares ou clusters.','',
    ]


def blind_analysis_lines(blind):
    agreement=read('blind_relabel','agreement').iloc[0]
    matrix=read('blind_relabel','confusion_matrix').rename(columns={
        'original_label':'Original (0=obstruído; 1=livre)',
        '0':'Cego 0 (obstruído)','1':'Cego 1 (livre)'})
    discordances=read('blind_relabel','discordances')[
        ['item','filename','group_id','original_label','blind_label','analysis_scope']]
    return [
        '**Rotulagem às cegas — exploratory/post hoc**','',
        f'Rotulador independente: {blind["rater"]}; respostas recebidas em '
        f'{blind["responses_received_date"]}, em '
        '[responses_breve.csv](../external-data/blind_relabel/responses_breve.csv). '
        'O script original scripts/analyze_blind_relabel.py foi executado sem alterações, '
        'com 10.000 iterações e seed 1980. Os três CSVs foram regenerados nesta máquina, '
        'junto com seus fragmentos LaTeX, e conferidos contra os valores esperados (tolerância 1e-6).','',
        f'Concordância: **{agreement.agreement:.6f} (98/100; 98%)**. '
        f'IC 95% de Wilson: **{agreement.agreement_wilson_low:.6f}–{agreement.agreement_wilson_high:.6f}**; '
        f'IC 95% por bootstrap de cenas: **{agreement.agreement_cluster_ci_low:.6f}–{agreement.agreement_cluster_ci_high:.6f}**. '
        f'κ de Cohen: **{agreement.cohen_kappa:.6f}**; '
        f'IC 95% por cenas: **{agreement.kappa_ci_low:.6f}–{agreement.kappa_ci_high:.6f}**.','',
        markdown(read('blind_relabel','agreement')),'',markdown(matrix),'',markdown(discordances),'',
        'As duas discordâncias são item_042.jpg (nonclear.0256.jpg, scene_0831; confiança 3) e '
        'item_056.jpg (nonclear.0258.jpg, scene_0833; confiança 2): ambas original 0 → cego 1. '
        'A leitura qualitativa indicada para estes casos é obstáculo lateral com faixa central '
        'transitável, a mesma ambiguidade de regra observada em “rótulo questionável” e “corredor estreito”. '
        'Essa interpretação não é uma justificativa escrita pelo rotulador: o campo notes das duas '
        'respostas está vazio. Nenhum rótulo foi corrigido.','',
        f'SHA-256 das respostas: `{blind["responses_sha256"]}`. '
        f'SHA-256 de agreement.csv: `{blind["agreement_sha256"]}`. '
        'Os campos de autoria, recebimento e execução já existentes em status.json foram preservados.','',
        '**Regra de rotulagem a explicitar no artigo — exploratory/post hoc**','',
        'Casos-limite a explicitar na definição de “obstáculo no caminho à frente”, mantendo os rótulos atuais:','',
        '- Escadas e rampas com corrimão: delimitar a passagem avaliada e o papel do corrimão lateral.',
        '- Bueiros/grelhas: explicitar como se distingue um elemento rente ao piso de uma obstrução.',
        '- Piso tátil: explicitar o tratamento da sinalização de acessibilidade no caminho.',
        '- Obstáculo lateral fora da faixa central: explicitar os limites da faixa transitável.',
        '- Corredores estreitos: explicitar o critério aplicado quando há objetos próximos e centro livre.','',
        'Este relatório registra questões para a redação futura; não propõe troca de rótulos '
        'nem altera o manuscrito ou a carta de resposta.','',
    ]


def build(mode='final'):
    output=ROOT/'results/stage3'; progress=json.loads((output/'progress.json').read_text())
    if any(not progress['stages'].get(str(s),{}).get('completed') or progress['stages'][str(s)].get('smoke') for s in [0,4,3,2,1,5]):
        raise ValueError('Ordered scientific post-test stages are incomplete')
    protection=json.loads((output/'protected_verification.json').read_text())
    if protection['changed'] or not protection['packages_unchanged']: raise ValueError('Protected inputs changed')
    registration=json.loads((output/'registration.json').read_text()); dry=json.loads((output/'dry_run.json').read_text())
    clusters,errors=validate_author_review(ROOT,mode)
    blind=validate_blind_analysis(ROOT,mode)
    if sha256(ROOT/'configs/stage3.json')!=registration['config_sha256']:
        raise ValueError('Registered stage-three configuration changed')
    if sha256(ROOT/'results/external_test/predictions.csv')!=registration['primary_predictions_sha256']:
        raise ValueError('Original external predictions changed')
    if sha256(ROOT/'results/external_test/final_models.json')!=registration['primary_freeze_sha256']:
        raise ValueError('Original model freeze changed')
    coverage={'Grouped refits':(40,len(list((output/'grouped_cv/jobs').glob('*/complete.json')))),
        'Cluster mosaics':(12,len(list((output/'error_analysis/mosaics').glob('cluster_*.jpg')))),
        'Paired AUC contrasts':(20,len(read('roc_analysis','paired_auc'))),
        'Paired specificity contrasts':(60,len(read('roc_analysis','paired_specificity'))),
        'Image attribute rows':(1918,len(read('error_analysis','image_attributes'))),
        'Logistic models/seeds':(19,len(read('error_analysis','logistic_diagnostics'))),
        'Blinded images':(100,int(read('blind_relabel','package_manifest').iloc[0]['images']))}
    if any(expected!=actual for expected,actual in coverage.values()):
        raise ValueError('Post-test output coverage differs from the requested design')
    review_tables=build_review_tables(ROOT,clusters,errors) if mode=='final' else {}
    lines=['# Etapa 3 — exploratory/post hoc','','Análises exploratórias e posteriores ao teste externo, continuando `b64221f`. '
        'Não substituem resultados principais. Ordem executada: 0 → 4 → 3 → 2 → 1 → 5. '
        'Nenhum modelo principal, limiar, predição ou rótulo protegido foi alterado. Nenhum FT agrupado executado. '+
        ('Revisão do autor concluída em 2026-10-13: 12 clusters nomeados e 148 erros categorizados; '
         'rotulagem independente às cegas analisada.' if mode=='final' else
         'Nomes de clusters e categorias de erro estão vazios, aguardando o autor.'),'',
        '**0. Registro e hashes protegidos — exploratory/post hoc**','',
        markdown(pd.DataFrame([{'Verificação':'Arquivos protegidos','Esperado':'todos intactos','Obtido':protection['files_checked']},
            {'Verificação':'Conteúdo protegido (GiB)','Esperado':'hash antes = depois','Obtido':round(protection['bytes_checked']/2**30,3)},
            {'Verificação':'Arquivos alterados','Esperado':0,'Obtido':len(protection['changed'])},
            {'Verificação':'Pacotes de treino','Esperado':'inalterados','Obtido':protection['packages_unchanged']},
            {'Verificação':'Nomes de clusters preenchidos','Esperado':12 if mode=='final' else 0,'Obtido':int((clusters.name!='').sum())},
            {'Verificação':'Categorias preenchidas','Esperado':148 if mode=='final' else 0,'Obtido':int((errors.category!='').sum())}])),'',
        'Inventários SHA-256 completos: `results/stage3/protected_before.json`, `protected_after.json` e '
        '`protected_hash_comparison.csv`. Incluem binários, imagens, resultados e configs; '
        'ambiente auxiliar OpenCV reutilizado, sem instalação no ambiente de treino. '
        f'Registro: `{registration["registered_utc"]}`. Baixa luz pré-fixada em Y médio < **{registration["low_light_threshold"]:.6f}**, '
        'percentil 10 das 342 imagens VIA, sem consultar erros/rótulos. Protocolo e PROVENANCE receberam o adendo pós-teste antes da execução.','',
        markdown(pd.DataFrame([{'Verificação':name,'Esperado':expected,'Obtido':actual,
            'analysis_scope':SCOPE} for name,(expected,actual) in coverage.items()])),'',
        f'Dry-run: {dry["stage3_time_hours_range"]} horas e {dry["stage3_additional_disk_gib_range"]} GiB adicionais. '
        'O espaço inclui os 40 pipelines completos de refits com PCA. Estimativas amplas para CPU, features em cache e hashing dos binários.','',
        '**4. Decisão nativa × adaptador — exploratory/post hoc**','',
        markdown(read('native_sensitivity','native_adapter_comparison')),'',
        'Todas as métricas/ICs: `native_adapter_metrics.csv/.tex`; proporções também com Wilson. '
        'Decisão nativa: margem > 0. Principal original: score_clear ≥ 0,5. '
        'AUC da sensibilidade nativa usa native_score; AUC do adaptador usa score_clear. '
        'Resultados de substituição nativa (Δ=3 p.p.), como sensibilidade:','',
        markdown(read('native_sensitivity','native_substitution')),'',
        '**3. Finalistas nos folds agrupados — exploratory/post hoc**','',
        markdown(read('grouped_cv','grouped_original_metrics')),'',
        'mean_fold e sd_fold descrevem os dez folds; estimate e seu IC usam as predições OOF agregadas. '
        'Bootstrap estratificado por imagem (10.000, seed 1980), pareado entre desenhos; Wilson para recall/FNR. '
        'AUC agrupada de margens OOF é descritiva: classificadores de folds diferentes têm escalas próprias. '
        'Nenhuma seleção nova; extração canônica de src/fixed_refit.py e fábrica de pipelines original reutilizadas.','',
        markdown(read('grouped_cv','grouped_minus_original')),'',markdown(read('grouped_cv','four_pairs')),'',
        'Os pares são colocados juntos no teste agrupado. opposite_original_train_test indica se cada imagem '
        'teve sua parceira no treino quando foi testada no desenho original. Mudança de predição não é causalmente '
        'atribuída ao agrupamento, pois outros membros dos folds também mudam.','',
        'Fine-tuning agrupado opcional: '+str(dry['grouped_fine_tuning_optional']['time_hours_range'])+
        ' horas, '+str(dry['grouped_fine_tuning_optional']['additional_disk_gib_range'])+' GiB, 150 jobs. Somente estimativa; não executado.','',
        '**2. Separabilidade e recall igualado — exploratory/post hoc**','']
    roc=read('roc_analysis','auc_specificity')
    main=roc[roc.seed.isna()] if 'seed' in roc else roc
    lines += [markdown(main),'',markdown(read('roc_analysis','paired_auc')),'',
        'Bootstrap pareado por cenas usa as mesmas 10.000 reamostragens para os 20 contrastes fixo − média '
        'das AUCs individuais FT. P bilateral do bootstrap centrado no nulo, correção Monte Carlo (1+contagem)/(B+1), Holm em 20 pares. '
        'DeLong multivariado: `delong_sensitivity.csv/.tex`, sensibilidade com independência entre imagens, '
        'contraste [1,−1/3,−1/3,−1/3]. [Referência](https://pubmed.ncbi.nlm.nih.gov/3203132/).','',
        markdown(read('roc_analysis','paired_specificity')),'',
        'Os limiares com recall 0,90/0,95/0,97 são oráculos externos não implantáveis, reestimados em cada bootstrap. '
        'Substituição com Δ=3 p.p. é apenas leitura descritiva pós-teste. Ensemble é sensibilidade separada '
        '(`ensemble_auc_sensitivity.csv/.tex`); as curvas principais FT são médias verticais de ROC por seed.','',
        markdown(read('roc_analysis','partial_auc').query('seed == "mean"')),'',
        'AUC parcial integra especificidade em sensibilidade de obstruídos [0,90;1,00], corrigida McClish com '
        'referência aleatória 0,005 e máximo 0,10, conforme [pROC](https://rdrr.io/cran/pROC/man/auc.html). '
        'Valores abaixo da diagonal são preservados na correção. ROC: `results/stage3/roc_analysis/roc.png/.pdf`, '
        'com zoom; coordenadas e curvas por seed em CSV/LaTeX.','',
        '**1. Limiar, atributos, clusters e erros — exploratory/post hoc**','']
    threshold=read('error_analysis','default_oracle_metrics')
    mean=threshold[threshold.seed.astype(str).eq('mean')&threshold.endpoint.eq('accuracy')]
    lines += [markdown(mean[['model','decision','estimate','ci_low','ci_high','analysis_scope']]),'',
        'Oráculo maximiza a acurácia nas próprias externas: estimativa de treino descritiva, não implantável. '
        'Seus ICs reescolhem o limiar em cada sorteio de cenas; métricas por seed e médias estão no CSV completo. '
        'Taxa prevista de obstruídos, distribuição/quantis por classe/domínio e KS: '
        '`predicted_obstructed_fraction.csv`, `score_distributions.csv`, `kolmogorov_smirnov.csv`. '
        'KS é descritivo e sua inferência usual não corrige correlação de cenas. '
        'Platt FT usa exclusivamente 342 OOF VIA por seed, sem ajustar em externas; '
        'adaptadores fixos existentes são reutilizados, `via_only_platt.csv/.tex`.','',
        markdown(read('error_analysis','via_only_platt').query('seed == "mean" and endpoint == "accuracy"')),'',
        'Atributos: Y médio/p10, saturação HSV, contraste RMS, densidade Canny, variância de Laplaciano, '
        'distâncias ao VIA em DINOv2 e quatro extratores. VIA exclui seu próprio vizinho. '
        'Quintis externos ficam fixos nos sorteios; a reamostragem por cenas é global antes de restringir subgrupos. '
        'Recalls de classes ausentes são não estimáveis (NaN), nunca zero. Em estratos pequenos, os ICs usam '
        'apenas sorteios com a classe necessária presente; bootstrap_valid/total registram essa cobertura. '
        'Ranking pós-teste: média da amplitude de acurácia entre quintis nos nove modelos; '
        'apenas descritivo, sem interpretação causal. Todos os atributos em `attribute_quintiles.csv/.tex`.','',
        'A classificação binária de baixa luz usa o limiar VIA pré-fixado; seus dois estratos estão em '
        '`low_light_metrics.csv/.tex` e não são divididos artificialmente em quintis.','',
        markdown(read('error_analysis','descriptive_attribute_ranking')),'',
        markdown(read('error_analysis','most_informative_quintiles')),'',
        'Regressões do acerto: por modelo/seed, atributos padronizados externamente e classe binária, '
        'máxima verossimilhança não penalizada, covariância sandwich com correção finita por cena. '
        'Coeficientes e SE/IC robustos: `cluster_robust_logistic.csv/.tex`; convergência/rank em `logistic_diagnostics.csv/.tex`. '
        'P-valores exploratórios não corrigidos e associações descritivas.','',
        markdown(read('error_analysis','cluster_robust_logistic')),'',markdown(read('error_analysis','logistic_diagnostics')),'',
        markdown(read('error_analysis','cluster_counts')),'',markdown(read('error_analysis','cluster_metrics')),'',
        'K-means k=12, n_init=20, seed 1980 sobre DINOv2 normalizado; VIA atribuído ao centroide mais próximo. '
        'O cluster 1 não contém imagens livres externas: seu recall de livres é não estimável. '
        'Mosaicos sem classes/nomes originais: `results/stage3/error_analysis/mosaics/cluster_00.jpg` até `cluster_11.jpg`, '
        '16 imagens aleatórias por cluster. Nomes do autor: `results/stage3/error_analysis/cluster_names.csv`'+
        (', preenchidos em 2026-10-13.' if mode=='final' else ', nome em branco.'),'',
        'Consenso: quatro fixos errados no padrão congelado e maioria das seeds EfficientNetB3 errada; '
        'mosaicos `consensus_clear_to_nonclear.jpg` e `consensus_nonclear_to_clear.jpg`, até 60 por tipo, '
        'ordenados pela confiança no erro, identificadores por hash e verdade/previsão visíveis. '
        'CSV do autor: `results/stage3/error_analysis/error_annotations.csv`'+
        (', categorias preenchidas em 2026-10-13. Lista original e adendo CATEGORY_ADDENDUM.md validados. '
         'Nenhum rótulo alterado.' if mode=='final' else ', categoria em branco. '
         'Lista de categorias em AUTHOR_REVIEW.md. Nenhum rótulo alterado; suspeitas ficam para anotação humana.'),'']
    if mode=='final': lines += author_review_lines(clusters,review_tables)
    lines += [
        '**5. Pacote cego — exploratory/post hoc**','',markdown(read('blind_relabel','package_manifest')),'']
    if mode=='final':
        lines += ['Nota: o campo `responses_analysed=False` em `package_manifest` reflete o estado da '
                  'preparação do pacote. As respostas foram analisadas em **2026-10-13**, conforme '
                  '[status.json](../results/stage3/blind_relabel/status.json).','']
    lines += [
        'Zip local: `external-data/blind_relabel/blind_relabel_package.zip`, somente images/, instructions.md '
        'e responses_template.csv (100 itens embaralhados seed 1980). `key.csv` fica fora do zip. '+
        ('A análise oficial de concordância foi rodada com as respostas reais de Fabricio Breve. ' if mode=='final' else
         'A análise de concordância NÃO foi rodada em respostas reais: aguardando Fabricio Breve. ')+
        'scripts/analyze_blind_relabel.py valida cobertura/rótulos/confiança e calcula concordância, '
        'Wilson, kappa com bootstrap por cena e discordâncias.'+
        ('' if mode=='final' else ' Testado só com respostas sintéticas.'),'']
    if mode=='final': lines += blind_analysis_lines(blind)
    log=output/'verification/pytest.log'; tests=log.read_text().splitlines()[-1] if log.exists() else 'Pytest completo pendente.'
    tests_completed=' passed' in tests and ' failed' not in tests
    smoke=json.loads((output/'verification/smoke.json').read_text())
    smoke_completed=all(smoke['stages'].get(str(s),{}).get('completed') and
        smoke['stages'][str(s)].get('smoke') for s in [0,4,3,2,1,5])
    if not smoke_completed: raise ValueError('Requested CPU smoke stages are incomplete')
    lines += ['**6. Verificações e entrega — exploratory/post hoc**','',tests,'',
        'O teste opcional de integração real em GPU usa --run-gpu e permanece ignorado na suíte padrão. '
        'Smoke completo em CPU, na ordem solicitada: `results/stage3/verification/smoke.json`.','',
        'Comandos: `.venv/bin/python scripts/analyze_stage3.py --dry-run`; '
        '`--smoke`; execução completa `--section all`; testes `python -m pytest -q`. '
        'O runner scripts/run_stage3.py registra logs, repete a conferência de hashes após testes e monta este relatório. '
        'Artefatos locais de pipelines/zip não entram no Git; tabelas, figuras, scripts, metadados e hashes são versionados. '+
        ('Fechamento concluído após revisão do autor e análise cega. Reproduzir com '
         '`.venv/bin/python scripts/build_stage3_report.py --mode final`. '
         'Nenhum rótulo, modelo, limiar ou predição foi alterado.' if mode=='final' else
         'Parada concluída para a anotação do autor; nenhum passo adicional de rotulagem/modelagem é autorizado nesta entrega.'),'']
    table([protection],output/'verification','protected_hash_summary')
    grouped_jobs=len(list((output/'grouped_cv/jobs').glob('*/complete.json')))
    cluster_mosaics=len(list((output/'error_analysis/mosaics').glob('cluster_*.jpg')))
    if grouped_jobs!=40 or cluster_mosaics!=12 or len(clusters)!=12:
        raise ValueError('Expected 40 grouped refits and 12 clusters/mosaics')
    write_json(output/'verification/completed.json',{'analysis_scope':SCOPE,'completed':tests_completed,
        'scientific_analyses_completed':True,'pytest_completed':tests_completed,'pytest_summary':tests,
        'cpu_smoke_completed':smoke_completed,'section_order':[0,4,3,2,1,5],
        'protected_files_unchanged':protection['files_checked'],'cluster_name_fields_filled':int((clusters.name!='').sum()),
        'error_categories_filled':int((errors.category!='').sum()),'report_mode':mode,
        'real_blind_responses_analysed':mode=='final','grouped_ft_executed':False,
        'primary_predictions_sha256':registration['primary_predictions_sha256'],
        'grouped_jobs':grouped_jobs,
        'cluster_mosaics':cluster_mosaics,
        'error_annotation_rows':len(errors)})
    destination=ROOT/'docs/STAGE3_REPORT.md'; destination.write_text('\n'.join(lines).rstrip()+'\n')
    print(f'Wrote {destination}; '+('stage three closed' if mode=='final' else 'stopped for author annotation'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['handoff','final'],default='final')
    build(mode=parser.parse_args().mode)
