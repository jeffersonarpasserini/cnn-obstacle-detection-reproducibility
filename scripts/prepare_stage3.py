#!/usr/bin/env python3
"""Section zero: register post-test scope, image threshold and protected hashes."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.stage3_io import SCOPE,configuration,snapshot,image_paths
from src.stage3_images import attributes,opencv
from src.study_io import write_json,sha256
import numpy as np
import pandas as pd


def prepare(output=None):
    config=configuration(); output=Path(output or ROOT/config['output']); output.mkdir(parents=True,exist_ok=True)
    registration=output/'registration.json'
    if registration.exists():
        recorded=json.loads(registration.read_text())
        if recorded['config_sha256']!=sha256(ROOT/'configs/stage3.json'):
            raise ValueError('Post-test numerical choices changed after registration')
        print('Section 0 already registered; preserving original baseline',flush=True); return recorded
    paths,_,_,_=image_paths()
    print('Computing VIA-only, label-free illumination threshold',flush=True)
    via=pd.DataFrame([attributes(path) for path in paths])
    threshold=float(np.quantile(via.luminance_mean,.1))
    via['analysis_scope']=SCOPE; via.to_csv(output/'via_image_attributes.csv',index=False)
    stamp=datetime.now(timezone.utc).isoformat()
    section=('\n\n## Análises pós-teste — 08/10/2026\n\n'
        'Etapa 3, continuando b64221f: todas as análises são `exploratory/post hoc`, posteriores ao teste externo; '
        'não substituem resultados principais nem alteram modelos, limiares ou predições congeladas. '
        'Ordem autorizada: 0 → 4 → 3 → 2 → 1 → 5. Novas saídas exclusivamente em results/stage3/ '
        'e pacote cego em external-data/blind_relabel/. Nomes de clusters e categorias de erro ficam vazios; '
        'nenhum rótulo será corrigido e nenhuma resposta humana será inventada.\n\n'
        f'Limiar de baixa luz, definido sem consultar erros: percentil 10 das 342 médias Y (YCbCr) do VIA = '
        f'**{threshold:.12f}**, escala 0–255. Baixa luz: média Y < limiar. Saturação HSV /255; '
        'contraste RMS = desvio-padrão de Y; Canny 100/200, apertureSize=3, L2gradient=True; '
        'nitidez = variância do Laplaciano CV_64F, ksize=1. Distâncias cosseno reutilizam DINOv2 e '
        'features ImageNet congeladas, sem nova inferência; para VIA, o próprio vizinho é excluído. '
        'Quintis definidos nas externas; k-means 12, seed 1980, n_init=20, embeddings normalizados.\n\n'
        'ICs: 10.000 sorteios de cenas inteiras nos estratos nonclear/clear/mistos, seed 1980; '
        'FT principal = média das métricas individuais das seeds, ensemble apenas sensibilidade. '
        'Oráculo externo não implantável: limiar de maior acurácia, desempate pelo menor limiar, '
        'livre iff score_clear >= limiar. Especificidade a recall fixo usa limites externos e é reestimada '
        'em cada bootstrap. ROC obstruído positivo usa a ordem exata de -score_clear, matematicamente '
        'equivalente a 1-score_clear, evitando cancelamento em probabilidades próximas de zero. '
        'AUC parcial em sensibilidade [0,90;1,00] integra especificidade nessa faixa, '
        'referência aleatória 0,005, máximo 0,10, correção McClish: 0,5×[1+(área−0,005)/0,095]. '
        '[Documentação e implementação pROC](https://rdrr.io/cran/pROC/man/auc.html). '
        'Contrastes de AUC fixo−média das seeds usam bootstrap pareado por cenas e Holm; '
        '[DeLong](https://pubmed.ncbi.nlm.nih.gov/3203132/) multivariado é sensibilidade, '
        'com contraste [1,−1/3,−1/3,−1/3] e sem ajustar a correlação entre cenas. '
        'Regressões logísticas do acerto são descritivas, por seed, máxima verossimilhança sem penalização '
        'e covariância sandwich por cena, correção finita G/(G−1)×(N−1)/(N−p). '
        'Platt FT: apenas OOF VIA por seed, logit com clip 1e−7, C=1, max_iter=500, seed 1980; '
        'não é ajuste no externo. Não há nova seleção de configuração. FT agrupado será só estimado, sem execução.\n')
    for name in ('docs/EXTERNAL_TEST_PROTOCOL.md','docs/PROVENANCE.md'):
        path=ROOT/name
        if '## Análises pós-teste — 08/10/2026' in path.read_text():
            raise ValueError('Post-test documentation already exists without its registration; inspect before continuing')
        path.write_text(path.read_text().rstrip()+section)
    recorded={'analysis_scope':SCOPE,'registered_utc':stamp,'baseline_commit':'b64221f',
        'config_sha256':sha256(ROOT/'configs/stage3.json'),'low_light_threshold':threshold,
        'illumination_source':'342 VIA images, no scores/errors/labels used', 'opencv_version':opencv().__version__,
        'primary_predictions_sha256':sha256(ROOT/'results/external_test/predictions.csv'),
        'primary_freeze_sha256':sha256(ROOT/'results/external_test/final_models.json'),
        'protocol_sha256':sha256(ROOT/'docs/EXTERNAL_TEST_PROTOCOL.md')}
    print('Hashing all protected files, including native model binaries',flush=True)
    before=snapshot(); write_json(output/'protected_before.json',before)
    write_json(registration,recorded)
    print(json.dumps({**recorded,'protected_files':len(before['files']),'protected_gib':before['total_bytes']/2**30},indent=2),flush=True)
    return recorded


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); prepare(args.output)
