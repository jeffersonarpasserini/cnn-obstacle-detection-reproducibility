#!/usr/bin/env python3
"""Exact deduplication, image-only V2 scene checks and author-review pause."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.study_io import sha256, digest, write_json, atomic_csv, checked_identity, runtime, guard_output


def image_label(path):
    prefix = Path(path).name.split('.')[0]
    if prefix not in ('clear', 'nonclear'):
        raise ValueError(f'Unrecognized label: {path}')
    return int(prefix == 'clear')


def inventory(directory):
    rows = []
    for path in sorted(Path(directory).glob('*.jpg'), key=lambda p: p.name):
        with Image.open(path) as image:
            rows.append({'filename': path.name, 'original_label': image_label(path),
                         'bytes': path.stat().st_size, 'sha256': sha256(path),
                         'width': image.width, 'height': image.height, 'exif_entries': len(image.getexif())})
    if not rows:
        raise ValueError('Image directory is empty')
    return pd.DataFrame(rows)


def exact_clean(raw, via, *, expected=None):
    if raw.filename.duplicated().any() or via.filename.duplicated().any():
        raise ValueError('Duplicate inventory filenames')
    references = via.set_index('sha256').to_dict('index')
    seen, kept, excluded = {}, [], []
    # Numeric ascending within each class; image content, never labels, detects duplicates.
    for row in raw.sort_values('filename').to_dict('records'):
        checksum = row['sha256']
        if checksum in references:
            reference = references[checksum]
            excluded.append({**row, 'reason': 'via_exact_label_conflict' if row['original_label'] != reference['original_label']
                             else 'via_exact_duplicate', 'duplicate_of': reference['filename']})
        elif checksum in seen:
            excluded.append({**row, 'reason': 'external_exact_duplicate', 'duplicate_of': seen[checksum]})
        else:
            seen[checksum] = row['filename']
            kept.append(row)
    novel = pd.DataFrame(kept, columns=raw.columns)
    counts = {'total': len(novel), 'clear': int(novel.original_label.eq(1).sum()),
              'nonclear': int(novel.original_label.eq(0).sum())}
    if expected is not None and counts != expected:
        raise ValueError(f'STOP: exact-clean counts differ: expected {expected}, obtained {counts}')
    return novel, pd.DataFrame(excluded), counts


def embeddings(paths, config, output, device):
    from build_scene_groups import load_embedding_cache, semantic_embeddings
    identity = [{'filename': path.name, 'sha256': sha256(path)} for path in paths]
    data_hash = digest(identity)
    cached = load_embedding_cache(output, data_hash, config, len(paths))
    if cached is not None:
        return cached[0], cached[1]
    values, metadata = semantic_embeddings(paths, config, device, config['cpu_threads'], config['batch_size'])
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / 'embeddings.tmp.npz'
    np.savez_compressed(temporary, embeddings=values)
    temporary.replace(output / 'embeddings.npz')
    write_json(output / 'embedding_metadata.json', {
        'identity': {'dataset_sha256': data_hash, 'semantic_model': config['semantic_model'],
                     'fallback_model': config.get('fallback_model')},
        'model': metadata, 'cache_sha256': sha256(output / 'embeddings.npz')})
    return values, metadata


def scene_graph(paths, values, config, output):
    from build_scene_groups import perceptual_hashes, minimum_hamming, cosine_similarities, guard_components
    from scene_groups_geometry import structural_scores
    output.mkdir(parents=True, exist_ok=True)
    distances = {name: minimum_hamming(array) for name, array in perceptual_hashes(paths).items()}
    similarities = cosine_similarities(values)
    structural = structural_scores(paths, distances['phash'], similarities, config['structural'],
                                   config['seed'], config['cpu_threads'], config['ransac_threshold'])
    groups, edges, thresholds, history = guard_components(distances['phash'], similarities, config, structural['records'])
    # Version all measured candidates and accepted links. Unmeasured pairs cannot be geometric links.
    rows = []
    for first, second in zip(*np.triu_indices(len(paths), 1)):
        first, second = int(first), int(second)
        record = structural['records'].get((first, second))
        if record is None and not edges[first, second]:
            continue
        record = record or {}
        rows.append({'filename_a': paths[first].name, 'filename_b': paths[second].name,
                     'phash_distance': int(distances['phash'][first, second]),
                     'semantic_cosine': float(similarities[first, second]),
                     'confirmed_edge': bool(edges[first, second]),
                     'edge_reason': 'phash' if distances['phash'][first, second] <= config['phash_threshold']
                                    else 'semantic_geometry' if edges[first, second] else '',
                     **record})
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list, np.ndarray))).any():
            frame[column] = frame[column].map(lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
                                            if isinstance(value, (dict, list, np.ndarray)) else value)
    atomic_csv(frame, output / 'pairs.csv.gz')
    atomic_csv(pd.DataFrame({'filename': [p.name for p in paths], 'group_id': groups}), output / 'groups.csv')
    write_json(output / 'graph_metadata.json', {'images': len(paths), 'confirmed_links': int(np.triu(edges, 1).sum()),
               'candidates': len(structural['records']), 'thresholds': thresholds, 'component_alerts': history,
               'config_sha256': digest(config), 'groups_sha256': sha256(output / 'groups.csv')})
    return groups, edges, frame


def overlap_rows(paths, edges, via_names):
    rows = []
    for first, second in zip(*np.triu_indices(len(paths), 1)):
        a, b = paths[first].name, paths[second].name
        if edges[first, second] and ((a in via_names) != (b in via_names)):
            new, reference = (b, a) if a in via_names else (a, b)
            rows.append({'filename': new, 'via_filename': reference, 'reason': 'confirmed_scene_overlap'})
    return pd.DataFrame(rows, columns=['filename', 'via_filename', 'reason'])


def audit_sample(novel, size=100, seed=1980):
    if len(novel) < size:
        raise ValueError('Audit sample exceeds novel images')
    n_clear = int(round(size * novel.original_label.eq(1).mean()))
    rng = np.random.default_rng(seed)
    parts = []
    for label, count in ((0, size-n_clear), (1, n_clear)):
        candidates = novel[novel.original_label.eq(label)].sort_values('filename')
        parts.append(candidates.iloc[rng.choice(len(candidates), count, replace=False)])
    sample = pd.concat(parts).sort_values('filename').reset_index(drop=True)
    sample.insert(0, 'audit_id', [f'audit_{i+1:03d}' for i in range(len(sample))])
    return sample


def mosaics(items, output, stem, *, columns=4, rows=5):
    """Captions supplied by caller: overlap captions use ONLY content hashes."""
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset in range(0, len(items), columns*rows):
        page = items[offset:offset+columns*rows]
        canvas = Image.new('RGB', (columns*280, rows*410), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, (path, caption) in enumerate(page):
            x, y = j % columns*280, j//columns*410
            with Image.open(path) as source:
                image = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (270, 365))
            canvas.paste(image, (x+(280-image.width)//2, y))
            draw.text((x+8, y+370), caption, fill='black')
        target = output / f'{stem}_{offset//(columns*rows)+1:03d}.png'
        canvas.save(target)
        paths.append(str(target.relative_to(ROOT)))
    return paths


def dry_run(config):
    return {'stage': 'cleaning', 'images_union': 1918, 'images_novel': 1576,
            'time_hours_range': [0.15, 1.5], 'additional_disk_gib_range': [.05, 1.0],
            'basis': 'Pinned DINO CPU forwards + two V2 graphs; candidate geometry dominates. No classifiers.'}


def run(args):
    config = json.loads(args.config.read_text())
    if args.dry_run:
        print(json.dumps(dry_run(config), indent=2)); return
    output = guard_output(args.output_dir)
    prior_report = output/'cleaning_report.json'
    if prior_report.exists() and json.loads(prior_report.read_text()).get('approved_by'):
        raise ValueError('Approved cleaning is frozen; use a separate output for a new protocol')
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dry_run(config), indent=2), flush=True)
    start = perf_counter()
    raw, via = inventory(args.dataset), inventory(args.via)
    atomic_csv(raw, output / 'external_manifest_raw.csv')
    expected_raw = {'total': 1929, 'clear': 1026, 'nonclear': 903}
    actual_raw = {'total': len(raw), 'clear': int(raw.original_label.eq(1).sum()), 'nonclear': int(raw.original_label.eq(0).sum())}
    if not args.smoke and actual_raw != expected_raw:
        raise ValueError(f'STOP: raw counts differ: {actual_raw}')
    expected = None if args.smoke else {'total': 1576, 'clear': 844, 'nonclear': 732}
    novel, excluded, counts = exact_clean(raw, via, expected=expected)
    atomic_csv(excluded, output / 'external_excluded_exact.csv')
    if not args.smoke and ((raw.width != 750).any() or (raw.height != 1000).any() or raw.exif_entries.any()):
        raise ValueError('STOP: raw image dimensions/EXIF differ from the source protocol')
    checked_identity(output, {'raw_sha256': sha256(output / 'external_manifest_raw.csv'),
                     'via_inventory_sha256': digest(via.to_dict('records')), 'scene_config': sha256(args.scene_config),
                     'external_config': sha256(args.config), 'smoke': args.smoke})
    if args.smoke:
        via = via.groupby("original_label", group_keys=False).head(4)
        novel = novel.groupby("original_label", group_keys=False).head(4)
    paths = [args.via / name for name in via.filename] + [args.dataset / name for name in novel.filename]
    paths.sort(key=lambda p: p.name)
    scene_config = json.loads(args.scene_config.read_text())
    values, encoder = embeddings(paths, scene_config, output / 'scene_union', args.device)
    _, union_edges, union_pairs = scene_graph(paths, values, scene_config, output / 'scene_union')
    overlap = overlap_rows(paths, union_edges, set(via.filename))
    union_measures = union_pairs.rename(columns={'filename_a': 'a', 'filename_b': 'b'})
    if len(overlap):
        measures = {(r.a, r.b): r for r in union_measures.itertuples()}
        for i, row in overlap.iterrows():
            key = tuple(sorted((row.filename, row.via_filename)))
            measured = measures[key]
            overlap.loc[i, 'edge_reason'] = measured.edge_reason
            overlap.loc[i, 'phash_distance'] = measured.phash_distance
            overlap.loc[i, 'semantic_cosine'] = measured.semantic_cosine
    atomic_csv(overlap, output / 'external_excluded_scene_overlap.csv')
    lookup = {path.name: i for i, path in enumerate(paths)}
    new_paths = [args.dataset / name for name in novel.filename]
    group_ids, _, _ = scene_graph(new_paths, values[[lookup[p.name] for p in new_paths]], scene_config,
                                output / 'scene_new')
    groups = pd.DataFrame({'filename': novel.filename, 'group_id': group_ids})
    atomic_csv(groups, output / 'external_groups.csv')
    clean = novel.merge(groups, on='filename', validate='one_to_one')
    clean['final_label'] = clean.original_label
    clean['excluded'] = clean.filename.isin(overlap.filename)
    clean['exclusion_reason'] = np.where(clean.excluded, 'confirmed_scene_overlap', '')
    atomic_csv(clean, output / 'external_manifest_clean.csv')
    overlap_items = []
    by_name = {p.name: p for p in paths}
    for i, row in enumerate(overlap.itertuples()):
        for name in (row.filename, row.via_filename):
            path = by_name[name]
            overlap_items.append((path, f'pair_{i+1:04d} img_{sha256(path)[:12]}'))
    scene_mosaics = mosaics(overlap_items, output / 'review/scene_overlap', 'scene_overlap', columns=2, rows=5)
    sample = audit_sample(novel, min(100, len(novel)) if args.smoke else 100, config['seed'])
    atomic_csv(sample, output / 'label_audit_sample.csv')
    audit_items = [(args.dataset / r.filename, f'{r.audit_id}: {"clear" if r.original_label else "nonclear"}')
                   for r in sample.itertuples()]
    audit_mosaics = mosaics(audit_items, output / 'review/label_audit', 'label_audit')
    corrections = output / 'label_corrections.csv'
    if not corrections.exists():
        atomic_csv(pd.DataFrame(columns=['filename', 'original_label', 'corrected_label', 'reason', 'reviewer', 'date']), corrections)
    scene_rejections = output/'scene_overlap_rejections.csv'
    if not scene_rejections.exists():
        atomic_csv(pd.DataFrame(columns=['filename', 'via_filename', 'reason', 'reviewer', 'date']), scene_rejections)
    from src.study_statistics import wilson
    report = {'status': 'paused_for_author_review', 'approved_by': None, 'approved_date': None,
              'raw_counts': actual_raw, 'exact_clean_counts': counts, 'excluded_exact': len(excluded),
              'via_exact_matches': int(excluded.reason.str.startswith('via_').sum()),
              'via_label_conflicts': int(excluded.reason.eq('via_exact_label_conflict').sum()),
              'internal_duplicates': int(excluded.reason.eq('external_exact_duplicate').sum()),
              'scene_overlap_images': int(overlap.filename.nunique()), 'scene_overlap_links': len(overlap),
              'main_external_images_pending_approval': int((~clean.excluded).sum()), 'groups': int(groups.group_id.nunique()),
              'audit_images': len(sample), 'audit_completed': False, 'discordance_rate': None,
              'discordance_wilson': None, 'audit_reject_above': .05,
              'scene_mosaics': scene_mosaics, 'audit_mosaics': audit_mosaics,
              'external_config_sha256': sha256(args.config), 'manifest_sha256': sha256(output / 'external_manifest_clean.csv'),
              'groups_sha256': sha256(output / 'external_groups.csv'), 'raw_manifest_sha256': sha256(output / 'external_manifest_raw.csv'),
              'overlap_sha256': sha256(output/'external_excluded_scene_overlap.csv'),
              'audit_sample_sha256': sha256(output / 'label_audit_sample.csv'), 'encoder': encoder,
              'seconds': perf_counter()-start, 'smoke': args.smoke, 'runtime': runtime(),
              'external_classification_predictions': 0}
    write_json(output / 'cleaning_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('encoder', 'runtime')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/external_test.json'))
    parser.add_argument('--scene-config', type=Path, default=Path('configs/scene_groups.json'))
    parser.add_argument('--dataset', type=Path, default=Path('external-data/via-dataset-extended'))
    parser.add_argument('--via', type=Path, default=Path('via-dataset'))
    parser.add_argument('--output-dir', type=Path, default=Path('external-data'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and args.output_dir == Path('external-data'):
        args.output_dir = Path('results/external_cleaning_smoke')
    run(args)


if __name__ == '__main__':
    main()
