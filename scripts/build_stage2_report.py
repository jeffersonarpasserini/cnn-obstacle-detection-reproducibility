#!/usr/bin/env python3
"""Build the author's stage-two report from frozen outputs; never fit or predict."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.study_io import sha256, write_json


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def markdown(frame):
    def value(item):
        if item is None or isinstance(item, float) and np.isnan(item): return '—'
        return str(item).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                      '| '+' | '.join('---' for _ in frame.columns)+' |',
                      *['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)]])


def percent(value): return '—' if pd.isna(value) else f'{100*float(value):.2f}%'
def label(model):
    if model.startswith('ft_'): return model.removeprefix('ft_')
    return model.split('_', 1)[0].upper()


def metric_table(summary, confidence, analysis=None):
    result = []
    for row in summary.itertuples():
        panel = confidence[confidence.model.eq(row.model) & confidence.operating_point.eq(row.operating_point)
                           & confidence.panel.eq('principal') & confidence.method.eq('whole-scene bootstrap')]
        panel = panel[panel.analysis.eq(analysis if analysis else row.analysis)]
        record = {'Modelo': label(row.model), 'Limiar': row.operating_point}
        for endpoint, title in [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'), ('mcc', 'MCC'),
                                ('roc_auc', 'AUC'), ('obstructed_recall', 'Recall nonclear'),
                                ('obstructed_false_negative_rate', 'FNR')]:
            interval = panel[panel.endpoint.eq(endpoint)].iloc[0]
            fmt = (lambda x: f'{float(x):.4f}') if endpoint in ('mcc', 'roc_auc') else percent
            record[title] = f'{fmt(getattr(row, endpoint))} [{fmt(interval.ci_low)}; {fmt(interval.ci_high)}]'
        result.append(record)
    return pd.DataFrame(result)


def build(*, prepare=False):
    directory = ROOT/'results/external_test'; tables = directory/'manuscript_tables'
    verification = ROOT/'results/stage2_verification'
    report = read_json(ROOT/'external-data/cleaning_report.json')
    review = pd.read_csv(ROOT/'external-data/external_manifest_clean.csv')
    groups = pd.read_csv(ROOT/'external-data/external_groups.csv')
    if report['status'] != 'approved' or not report['audit_completed']:
        raise ValueError('Author review is not approved')
    freeze = read_json(directory/'final_models.json')
    evaluation = read_json(directory/'first_evaluation.json')
    curve = read_json(ROOT/'results/learning_curve/run_metadata.json')
    if not prepare and (not freeze or not freeze.get('frozen') or not evaluation or evaluation['status'] != 'completed'
                        or not curve or not curve.get('completed')):
        raise ValueError('Stage-two scientific outputs are not all complete')
    lines = ['# Etapa 2 — validação externa geográfica e temporal', '',
        'Decisão do autor: 06/10/2026, `PROMPT_STAGE2_NESTED_EXTERNAL.md`, continuando a etapa 1 `fb1d7f1`. '+
        ('Execução científica ainda pendente; este registro preparatório não contém resultados externos.' if prepare else
         'Modelos finais, teste externo único e curva secundária concluídos. Nested CV completa não foi executada.'), '',
        'VIA: Fabricio Breve, 2020, Rio Claro/SP, 342 imagens. Extensão: Jefferson A. R. Passerini, coleta posterior, '+
        'Fernandópolis e Jales/SP, cerca de 400 km de Rio Claro, 1.576 novas. Período exato/aparelho: TODO(autor). '+
        'Definição de nonclear idêntica; auditoria de 100 rótulos aprovada. Hashes e revisão de cenas não identificaram '+
        'imagens/cenas compartilhadas. O histórico público `4a5a1a5`, contendo os JPG brutos, foi preservado por decisão do autor.', '',
        '**Aplicação da revisão**', '']
    checked = pd.DataFrame([
        ['Exclusões por cena', 0, int(review.excluded.sum())],
        ['Principal', 1576, int((~review.excluded).sum())], ['clear', 844, int(review.final_label.eq(1).sum())],
        ['nonclear', 732, int(review.final_label.eq(0).sum())], ['Grupos', 1378, int(groups.group_id.nunique())],
        ['Auditoria concluída', True, report['audit_completed']], ['Discordância', '0/100', f'{report["discordance_count"]}/100'],
        ['Aprovação', 'Jefferson Antonio Ribeiro Passerini (author)', report['approved_by']],
        ['Data', '2026-10-06', report['approved_date']]], columns=['Verificação', 'Esperado', 'Obtido'])
    if any(str(row.Esperado) != str(row.Obtido) for row in checked.itertuples()):
        raise ValueError('Author-review verification does not match the registered expectations')
    lines += [markdown(checked), '', f'Wilson 95% da discordância: [{percent(report["discordance_wilson"][0])}; '+
              f'{percent(report["discordance_wilson"][1])}]. CSV de nove rejeições e grafos automáticos preservados.', '',
        '**Planejamento informado antes do treino**', '']
    estimates = []
    for name in ('final_models', 'external', 'learning_curve'):
        estimate = read_json(verification/f'dry_{name}.json')
        estimates.append({'Etapa': name, 'Jobs': estimate['jobs'], 'Tempo (h)': '–'.join(f'{v:.2f}' for v in estimate['time_hours_range']),
                          'Disco (GiB)': '–'.join(f'{v:.2f}' for v in estimate['additional_disk_gib_range'])})
    lines += [markdown(pd.DataFrame(estimates)), '',
        'Estimativas derivadas dos timings existentes na RTX 4070 Ti; a curva mantém cinco CNNs, pois FT fica abaixo de 48 h. '+
        'Nested principal foi dispensada devido às 900–2.702 h previstas por versão; análise principal usa externo geográfico único. '+
        'O protocolo/configuração com o desvio são commitados antes de qualquer classificação externa.', '', '**Correção de viés sem treinamento**', '']
    supplementary = read_json(verification/'dry_nested_no_relieff.json')
    if supplementary:
        lines.insert(-2, f'Nested suplementar sem Relief-F: {supplementary["configurations_per_outer_fold"]} configurações; '+
                     '–'.join(f'{v:.2f}' for v in supplementary['time_hours_range'])+' h e '+
                     '–'.join(f'{v:.2f}' for v in supplementary['additional_disk_gib_range'])+' GiB. Somente dry-run, sem execução; '+
                     'não valida a seleção na grade integral do artigo.')
    bias = pd.read_csv(tables/'tibshirani_bias.csv')
    display = bias[['approach', 'accuracy_original', 'accuracy_bias', 'accuracy_corrected',
                    'obstructed_recall_original', 'obstructed_recall_bias', 'obstructed_recall_corrected']].copy()
    for column in display.columns[1:]: display[column] = display[column].map(percent)
    lines += [markdown(display), '',
        'Fórmula solicitada: média dos dez folds de máximo da abordagem menos finalista; corrigida = original − viés. '+
        'Vencedor usa toda a grade. Versão conservadora de Tibshirani & Tibshirani (2009), recall como extensão secundária; '+
        'não é uma validação independente nem garantia de estimativa sem viés. [Referência](https://doi.org/10.1214/08-AOAS224).', '']
    relief = pd.read_csv(tables/'relieff_vs_pca.csv').copy()
    lines += [markdown(relief), '', 'Diferença é PCA − melhor Relief-F por média da accuracy nos folds. '+
              'Diagnósticos completos: `tibshirani_fold_diagnostics.csv`; saídas também em LaTeX.', '']
    if not prepare:
        for filename, key in [('predictions.csv', 'predictions_sha256'), ('metrics.csv', 'metrics_sha256')]:
            if sha256(directory/filename) != evaluation[key]: raise ValueError('Frozen external outputs changed')
        for filename, checksum in curve['outputs'].items():
            if sha256(ROOT/'results/learning_curve'/filename) != checksum: raise ValueError('Learning outputs changed')
        for filename, checksum in freeze['artifacts'].items():
            if sha256(directory/filename) != checksum: raise ValueError(f'Frozen model artifact changed: {filename}')
        hashes = pd.DataFrame([{'artifact': name, 'sha256': checksum} for name,checksum in freeze['artifacts'].items()])
        hashes.to_csv(verification/'final_artifact_hashes.csv', index=False)
        lines += [f'Commit do protocolo/desvio: `{freeze["preregistration"]["protocol"]["commit"]}`. '+
                  f'Config SHA-256: `{freeze["preregistration"]["config"]["sha256"]}`.', '',
                  '**Modelos e limiares congelados**', '', markdown(hashes), '',
                  f'Freeze SHA-256: `{sha256(directory/"final_models.json")}`. '+
                  'Modelos/checkpoints binários são locais e ignorados; manifests, custos e hashes são versionados. '+
                  'Treino final só VIA; FT-D usa validação de aproximadamente 20% por grupos, idêntica entre seeds. '+
                  'Os 40 refits originais fornecem OOF canônico verificado sem novos ajustes. Adaptadores/limiares dos fixos '+
                  'herdam o otimismo da seleção; o teste externo verifica a manutenção do alvo de recall 0,95.', '',
                  '**Resultados externos principais, IC 95% por cenas**', '']
        confidence = pd.read_csv(tables/'cluster_bootstrap_wilson.csv')
        summary = pd.read_csv(tables/'seed_mean_metrics.csv')
        summary = summary[summary.panel.eq('principal')]
        lines += [markdown(metric_table(summary, confidence)), '',
                  'FT principal é a média das métricas das três seeds individuais, não a métrica das decisões médias. '+
                  'Bootstrap: 10.000 réplicas, seed 1980, cenas inteiras em estratos nonclear/clear/mistos. '+
                  'Wilson por seed para recall/FNR está no CSV de ICs.', '', '**Ensemble de três seeds, sensibilidade**', '']
        ensemble = pd.read_csv(tables/'ensemble_sensitivity.csv'); ensemble = ensemble[ensemble.panel.eq('principal')]
        lines += [markdown(metric_table(ensemble, confidence, '3-seed ensemble sensitivity')), '',
                  '**Substituição e McNemar**', '']
        paired = pd.read_csv(tables/'substitution_analysis.csv'); paired = paired[paired.panel.eq('principal')]
        paired_display = paired[['fixed', 'fine_tuning', 'operating_point', 'endpoint', 'difference', 'ci_low', 'ci_high', 'substitution']].copy()
        for name in ('fixed', 'fine_tuning'): paired_display[name] = paired_display[name].map(label)
        for name in ('difference', 'ci_low', 'ci_high'): paired_display[name] = paired_display[name].map(percent)
        lines += [markdown(paired_display), '',
                  'Diferença fixo − FT; margem 3 p.p. Supported: limite inferior ≥ −0,03; not_supported: '+
                  'limite superior < −0,03; demais inconclusive. FNR tem orientação invertida. Bootstrap pareado '+
                  'usa os mesmos sorteios de cenas e média das métricas por seed.', '']
        mcnemar = pd.read_csv(tables/'mcnemar_exact_holm.csv'); mcnemar = mcnemar[mcnemar.panel.eq('principal')]
        lines += [markdown(mcnemar[['fixed', 'fine_tuning', 'seed', 'operating_point', 'fixed_only_correct', 'ft_only_correct', 'p_value', 'p_holm']]), '',
                  'McNemar exato por seed; Holm aplicado à família de comparações de cada painel/ponto. '+
                  'Ensemble é sensibilidade e não substitui a análise principal.', '', '**VIA × domínio externo e calibração**', '',
                  markdown(pd.read_csv(tables/'domain_comparison.csv')), '',
                  'Mudança de domínio com ICs independentes entre VIA e externo: `domain_shift.csv`. '+
                  'Brier/confiabilidade por domínio: `domain_calibration.csv`; confiabilidade externa: '+
                  '`calibration_reliability.csv`, `reliability.png/.pdf`. O 10-fold VIA conserva decisões nativas; '+
                  'o padrão externo usa probabilidades congeladas. Calibração VIA dos fixos é otimista porque '+
                  'o adaptador é ajustado no mesmo OOF selecionado. Tibshirani é complemento descritivo, sem IC independente.', '',
                  '**Curva de aprendizado e custo de atualização**', '']
        learning_tables = ROOT/'results/learning_curve/manuscript_tables'
        learning = pd.read_csv(learning_tables/'learning_curve_metrics.csv')
        shown = learning[learning.endpoint.isin(['accuracy', 'obstructed_recall']) & learning.operating_point.eq('default')].copy()
        lines += [markdown(shown), '', markdown(pd.read_csv(learning_tables/'update_costs.csv')), '',
                  'Teste B e pool são disjuntos por cena. Frações 0/25/50/100% são subconjuntos aninhados de '+
                  'cenas inteiras; valores reais estão em `subsets.json` e `split.csv`. 0% reutiliza modelos finais; '+
                  'demais tamanhos reajustam sem busca nova. Custo de atualização é retreino fresco, não warm start. '+
                  'B nunca entra em seleção, early stopping ou limiares. Figura `learning_curve.png/.pdf`.', '']
        predictions = pd.read_csv(directory/'predictions.csv')
        learning_root = ROOT/'results/learning_curve'
        split = pd.read_csv(learning_root/'split.csv')
        test_b = split[split.split.eq('test_B')]
        pool = split[split.split.eq('pool')]
        if set(test_b.group_id) & set(pool.group_id) or set(split.filename) != set(review.filename):
            raise ValueError('Final learning split has scene leakage or population mismatch')
        curve_predictions = pd.read_csv(learning_root/'predictions.csv', float_precision='round_trip')
        if curve_predictions.duplicated(['pool_fraction', 'model', 'seed', 'filename']).any():
            raise ValueError('Duplicate secondary image/fraction/model/seed prediction')
        expected_panels = {(row['model'], row['seed']) for row in freeze['models']}
        fractions = {float(key): names for key,names in curve['identity']['subsets'].items()}
        if set(curve_predictions.pool_fraction) != set(fractions):
            raise ValueError('Secondary curve fractions do not match the frozen subsets')
        for fraction, rows in curve_predictions.groupby('pool_fraction'):
            if set(zip(rows.model, rows.seed)) != expected_panels:
                raise ValueError('Missing/extra secondary model/seed panel')
            for _, panel in rows.groupby(['model', 'seed']):
                aligned = test_b.set_index('filename').loc[panel.filename]
                if set(panel.filename) != set(test_b.filename) or not np.array_equal(panel.true_label, aligned.final_label):
                    raise ValueError('Secondary test-B coverage/labels mismatch')
                if not np.array_equal(panel.group_id, aligned.group_id):
                    raise ValueError('Secondary test-B scene identity mismatch')
                if set(panel.training_images) != {342+len(fractions[float(fraction)])}:
                    raise ValueError('Secondary training-size bookkeeping mismatch')
            if set(fractions[float(fraction)]) & set(test_b.filename):
                raise ValueError('A test-B image entered a secondary training subset')
        curve_costs = pd.read_csv(learning_root/'update_costs.csv')
        new_jobs = list(learning_root.glob('size_*/jobs/*/complete.json'))
        expected_jobs = len(expected_panels)*sum(fraction > 0 for fraction in fractions)
        if len(curve_costs) != expected_jobs or len(new_jobs) != expected_jobs:
            raise ValueError('Secondary completed-job/cost counts mismatch')
        lines += ['**Verificações finais da curva**', '', markdown(pd.DataFrame([
            ['Ajustes novos', expected_jobs, len(new_jobs)],
            ['Imagens no teste B', 788, len(test_b)], ['Imagens no pool', 788, len(pool)],
            ['Predições nos quatro pontos', len(expected_panels)*len(fractions)*len(test_b), len(curve_predictions)],
            ['Predições duplicadas', 0, 0], ['Cenas entre teste B e pool', 0, 0]],
            columns=['Verificação', 'Esperado', 'Obtido'])), '']
        write_json(verification/'completed_stage2_verification.json', {
            'completed': True, 'review': checked.to_dict('records'), 'model_panels': len(freeze['models']),
            'external_images': int(predictions.filename.nunique()), 'prediction_rows': len(predictions),
            'duplicate_image_model_seed': int(predictions.duplicated(['filename', 'model', 'seed']).sum()),
            'protocol_commit': freeze['preregistration']['protocol']['commit'],
            'freeze_sha256': sha256(directory/'final_models.json'), 'thresholds_sha256': sha256(directory/'thresholds.json'),
            'predictions_sha256': evaluation['predictions_sha256'], 'curve_completed': curve['completed'],
            'curve_new_jobs': len(new_jobs), 'curve_prediction_rows': len(curve_predictions),
            'curve_test_B_images': len(test_b), 'curve_pool_images': len(pool),
            'curve_scene_leakage': False, 'curve_duplicate_predictions': 0,
            'no_complete_nested_executed': not (ROOT/'results/nested_cv/outer_predictions.csv').exists()})
        history = read_json(verification/'execution_history.json') or []
        progress = read_json(verification/'progress.json') or {}
        history += [{'stage': name, **record} for name, record in progress.get('stages', {}).items()
                    if record.get('ended_utc')]
        unique = {(row['stage'], row['started_utc']): row for row in history}
        history_table = pd.DataFrame([{'Etapa': row['stage'], 'Estado': row['state'],
            'Início UTC': row['started_utc'], 'Fim UTC': row['ended_utc'],
            'Horas ativas': round((datetime.fromisoformat(row['ended_utc'])-
                                  datetime.fromisoformat(row['started_utc'])).total_seconds()/3600, 4)}
            for row in sorted(unique.values(), key=lambda x: x['started_utc'])])
        history_table.to_csv(verification/'measured_stage_times.csv', index=False)
        disk = []
        for folder in (directory, ROOT/'results/learning_curve'):
            files = [path for path in folder.rglob('*') if path.is_file()]
            disk.append({'Diretório': str(folder.relative_to(ROOT)), 'Arquivos locais': len(files),
                         'GiB lógicos': round(sum(path.stat().st_size for path in files)/2**30, 4),
                         'GiB alocados': round(sum(path.stat().st_blocks*512 for path in files)/2**30, 4)})
        pd.DataFrame(disk).to_csv(verification/'measured_disk.csv', index=False)
        lines += ['**Tempo e disco medidos**', '', markdown(history_table), '', markdown(pd.DataFrame(disk)), '',
                  'Horas ativas são duração de cada processo, incluindo recarga, I/O, inferência e verificações; '+
                  'intervalos entre retomadas não contam como computação. Retomadas verificam o freeze e reutilizam '+
                  'predições principais idênticas, sem nova inferência principal. Históricos completos de tentativas: '+
                  '`execution_history.json`; checkpoints e custos por job ficam nos respectivos manifests. '+
                  'Quando reconstruído a partir do registro de recuperação, o fim da tentativa é aproximado pelo evento imediatamente seguinte. '+
                  'Disco inclui modelos/checkpoints locais e saídas das etapas; exclui imagens e cache canônico já existentes. '+
                  'Interrupções e recuperação de memória estão documentadas em PROVENANCE.md.', '']
    log = verification/'pytest.log'
    publication = read_json(ROOT/'cache/stage2_publication.json')
    publication_note = ('Relatório e resultados medidos serão commitados e enviados após a auditoria final; '+
                        'o protocolo/desvio é enviado antes da avaliação.')
    if not prepare and publication and publication.get('state') == 'completed' and (
            publication.get('verification', {}).get('predictions_sha256') == evaluation['predictions_sha256']):
        publication_note = ('Resultados científicos publicados no commit `'+publication['commit']+'`; '+
                            'auditorias posteriores acrescentam verificações sem alterar as predições congeladas.')
    lines += ['**Testes e evidência de execução**', '',
              log.read_text().strip().splitlines()[-1] if log.exists() else 'Pytest completo ainda pendente.', '',
              'Log completo: `results/stage2_verification/pytest.log`. Progresso: `progress.json`; '+
              'logs por etapa em `logs/stage2/`. '+publication_note, '']
    target = ROOT/'docs/STAGE2_REPORT.md'
    temporary = target.with_suffix('.tmp.md'); temporary.write_text('\n'.join(lines).rstrip()+'\n'); temporary.replace(target)
    print(f'Wrote {target}; prepare={prepare}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); build(prepare=args.prepare)


if __name__ == '__main__': main()
