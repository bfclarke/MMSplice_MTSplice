import pandas as pd
import numpy as np
import pyranges
from kipoiseq.dataclasses import Variant
import kipoiseq.transforms.functional as F
from kipoiseq.extractors import MultiSampleVCF
from sklearn.externals import joblib
from pkg_resources import resource_filename


LINEAR_MODEL = joblib.load(resource_filename(
    'mmsplice', 'models/linear_model.pkl'))
LOGISTIC_MODEL = joblib.load(resource_filename(
    'mmsplice', 'models/Pathogenicity.pkl'))
EFFICIENCY_MODEL = joblib.load(resource_filename(
    'mmsplice', 'models/splicing_efficiency.pkl'))

ref_psi_annotation = {
    'grch37': resource_filename(
        'mmsplice', 'models/gtex_psi_map37_ranges.csv.gz'),
    'grch38': resource_filename(
        'mmsplice', 'models/gtex_psi_map_ranges.csv.gz')
}


def left_normalized(variant):
    """
    Left normalizated version of variant object.
    Example:
      CA:CAGG -> '':GC
    """
    pos = variant.pos

    for i in range(min(len(variant.ref), len(variant.alt))):
        if variant.ref[i] == variant.alt[i]:
            pos += 1
        else:
            break

    diff = pos - variant.pos
    ref = variant.ref[diff:]
    alt = variant.alt[diff:]

    return Variant(variant.chrom, pos, ref, alt)


def clip(x, clip_threshold=0.00001):
    return np.clip(x, clip_threshold, 1 - clip_threshold)


def logit(x):
    x = clip(x)
    return np.log(x) - np.log(1 - x)


def expit(x):
    return 1. / (1. + np.exp(-x))


def pyrange_remove_chr_from_chrom_annotation(pr):
    df = pr.df
    df['Chromosome'] = df['Chromosome'].str.replace('chr', '')
    return pyranges.PyRanges(df)


def max_varEff(df):
    """ Summarize largest absolute effect per variant across all affected exons.
    Args:
        df: result of `predict_all_table`
    """
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)

    df_max = df.groupby(['ID'], as_index=False).agg(
        {'delta_logit_psi': lambda x: max(x, key=abs)})

    df_max = df_max.merge(df, how='left', on=['ID', 'delta_logit_psi'])
    df_max = df_max.drop_duplicates(subset=['ID', 'delta_logit_psi'])
    return df_max


def _not_close0(arr):
    return ~np.isclose(arr, 0)


def _and_not_close0(x, y):
    return np.logical_and(_not_close0(x), _not_close0(y))


def transform(X, region_only=False):
    ''' Make interaction terms for the overlapping prediction region
    Args:
        X: modular prediction. Shape (, 5)
        region_only: only interaction terms with indicator function on overlapping
    '''
    exon_overlap = np.logical_or(
        _and_not_close0(X[:, 1], X[:, 2]),
        _and_not_close0(X[:, 2], X[:, 3])
    )
    acceptor_intron_overlap = _and_not_close0(X[:, 0], X[:, 1])
    donor_intron_overlap = _and_not_close0(X[:, 3], X[:, 4])

    if not region_only:
        exon_overlap = X[:, 2] * exon_overlap
        donor_intron_overlap = X[:, 4] * donor_intron_overlap
        acceptor_intron_overlap = X[:, 0] * acceptor_intron_overlap

    return np.hstack([
        X,
        exon_overlap.reshape(-1, 1),
        donor_intron_overlap.reshape(-1, 1),
        acceptor_intron_overlap.reshape(-1, 1)
    ])


def predict_deltaLogitPsi(X_ref, X_alt):
    return LINEAR_MODEL.predict(transform(X_alt - X_ref, region_only=False))


def predict_pathogenicity(X_ref, X_alt):
    X = transform(X_alt - X_ref, region_only=True)
    X = np.concatenate([X_ref, X_alt, X[:, -3:]], axis=-1)
    return LOGISTIC_MODEL.predict_proba(X)[:, 1]


def predict_splicing_efficiency(X_ref, X_alt):
    X = transform(X_alt - X_ref, region_only=False)
    X = X[:, [1, 2, 3, 5]]  # no intronic modules
    return EFFICIENCY_MODEL.predict(X)


def read_vep(vep_result_path,
             max_per_var=False):
    ''' Read MMSplice VEP plugin output. Only support vcf type output.

    Args:
        vep_result_path: file path to the returned result of VEP plugin.
        max_per_var: return maximum absolute effect size per variant.
    '''
    keys = [
        'alt_acceptor',
        'alt_acceptorIntron',
        'alt_donor',
        'alt_donorIntron',
        'alt_exon',
        'delta_logit_psi',
        'pathogenicity',
        'ref_acceptor',
        'ref_acceptorIntron',
        'ref_donor',
        'ref_donorIntron',
        'ref_exon'
    ]

    score_pred = []

    for v in MultiSampleVCF(vep_result_path):
        csq = v.source.INFO['CSQ'].split(',')
        predictions = map(lambda x: tuple(x.split('|')[-len(keys):]), csq)

        for pred in predictions:
            if pred != ('',) * len(keys):
                x = dict(
                    zip(keys, map(float, (i if i != '' else 0 for i in pred))))
                x['ID'] = str(v)
                score_pred.append(x)

    df_plugin = pd.DataFrame(score_pred)

    if max_per_var:
        df_plugin = max_varEff(df_plugin).set_index('ID')

    return df_plugin


def get_var_side(variant, exon):
    '''
    Get exon variant side.

    Args:
      variant: Variant class 1-based.
      exon: pybedtools.Interval 0-based.
    '''
    assert variant.chrom == exon.chrom

    variant = left_normalized(variant)
    var_end = variant.start + max(len(variant.ref), len(variant.alt))

    if exon.strand == '+':
        if variant.start < exon.start:
            return "left"
        elif var_end > exon.end:
            return "right"
        else:
            return "exon"
    else:
        if variant.start < exon.start:
            return "right"
        elif var_end > exon.end:
            return "left"
        else:
            return "exon"


bases = ['A', 'C', 'G', 'T']


def onehot(seq):
    X = np.zeros((len(seq), len(bases)))
    for i, char in enumerate(seq):
        if char == "N":
            pass
        else:
            X[i, bases.index(char.upper())] = 1
    return X


def encodeDNA(seq_vec):
    max_len = max(map(len, seq_vec))
    return np.array([
        F.one_hot(F.pad(seq, max_len, anchor="start"), neutral_value=0)
        for seq in seq_vec
    ])


ascot_to_gtex_tissue_mapping = {
    'Adrenal Gland': 'Adrenal Gland',
    'Amygdala - Brain': 'Brain - Amygdala',
    'Anterior cingulate - Brain': 'Brain - Anterior cingulate cortex (BA24)',
    'Aorta - Artery': 'Artery - Aorta',
    'Atrial Appendage - Heart': 'Heart - Atrial Appendage',
    'Bladder': 'Bladder',
    'Caudate nucleus - Brain': 'Brain - Caudate (basal ganglia)',
    'Cerebellar Hemisphere - Brain': 'Brain - Cerebellar Hemisphere',
    'Cerebellum - Brain': 'Brain - Cerebellum',
    'Coronary - Artery': 'Artery - Coronary',
    'Cortex - Brain': 'Brain - Cortex',
    'Cortex - Kidney': 'Kidney - Cortex',
    'EBV-xform lymphocytes - Cells': 'Cells - EBV-transformed lymphocytes',
    'Ectocervix - Cervix': 'Cervix - Ectocervix',
    'Endocervix - Cervix': 'Cervix - Endocervix',
    'Fallopian Tube': 'Fallopian Tube',
    'Frontal Cortex - Brain': 'Brain - Frontal Cortex (BA9)',
    'Gastroesoph. Junc. - Esophagus': 'Esophagus - Gastroesophageal Junction',
    'Hippocampus - Brain': 'Brain - Hippocampus',
    'Hypothalamus - Brain': 'Brain - Hypothalamus',
    'Ileum - Small Intestine': 'Small Intestine - Terminal Ileum',
    'Left Ventricle - Heart': 'Heart - Left Ventricle',
    'Leukemia (CML) - Cells': 'Cells - Leukemia cell line (CML)',
    'Liver': 'Liver',
    'Lung': 'Lung',
    'Mammary Tissue - Breast': 'Breast - Mammary Tissue',
    'Minor Salivary Gland': 'Minor Salivary Gland',
    'Mucosa - Esophagus': 'Esophagus - Mucosa',
    'Muscularis - Esophagus': 'Esophagus - Muscularis',
    'Not Sun Exposed - Skin': 'Skin - Not Sun Exposed (Suprapubic)',
    'Nucleus accumbens - Brain': 'Brain - Nucleus accumbens (basal ganglia)',
    'Ovary': 'Ovary',
    'Pancreas': 'Pancreas',
    'Pituitary': 'Pituitary',
    'Prostate': 'Prostate',
    'Putamen - Brain': 'Brain - Putamen (basal ganglia)',
    'Sigmoid - Colon': 'Colon - Sigmoid',
    'Skeletal - Muscle': 'Muscle - Skeletal',
    'Spinal cord (C1) - Brain': 'Brain - Spinal cord (cervical c-1)',
    'Spleen': 'Spleen',
    'Stomach': 'Stomach',
    'Subcutaneous - Adipose': 'Adipose - Subcutaneous',
    'Substantia nigra - Brain': 'Brain - Substantia nigra',
    'Sun Exposed (Lower leg) - Skin': 'Skin - Sun Exposed (Lower leg)',
    'Testis': 'Testis',
    'Thyroid': 'Thyroid',
    'Tibial - Artery': 'Artery - Tibial',
    'Tibial - Nerve': 'Nerve - Tibial',
    'Transverse - Colon': 'Colon - Transverse',
    'Uterus': 'Uterus',
    'Vagina': 'Vagina',
    'Visceral (Omentum) - Adipose': 'Adipose - Visceral (Omentum)',
    'Whole Blood': 'Whole Blood',
    'Xform. fibroblasts - Cells': 'Cells - Transformed fibroblasts'
}


def read_ref_psi_annotation(ref_psi_version, chroms=None):
    if ref_psi_version in ref_psi_annotation:
        df_ref = pd.read_csv(ref_psi_annotation[ref_psi_version])
    else:
        raise ValueError('ref_psi_version should be one of %s'
                         % str(list(ref_psi_annotation.keys())))

    df_ref = df_ref.rename(columns={
        v: k
        for k, v in ascot_to_gtex_tissue_mapping.items()
    })
    df_ref['Start'] -= 1  # 1-based to zero based

    chr_annotaion = any(chrom.startswith('chr') for chrom in chroms)
    if not chr_annotaion:
        df_ref['Chromosome'] = df_ref['Chromosome'].str.replace('chr', '')

    index_col = df_ref['Chromosome'].astype(str) + ':' + \
        df_ref['Start'].astype(str) + '-' + df_ref['End'].astype(str) \
        + ':' + df_ref['Strand'].astype(str)
    df_ref['exons'] = index_col

    return df_ref.set_index('exons')


def delta_logit_PSI_to_delta_PSI(delta_logit_psi, ref_psi,
                                 genotype=None, clip_threshold=0.001):
    ref_psi = clip(ref_psi, clip_threshold)
    pred_psi = expit(delta_logit_psi + logit(ref_psi))

    if genotype is not None:
        pred_psi = np.where(np.array(genotype) == 1,
                            (pred_psi + ref_psi) / 2,
                            pred_psi)

    return pred_psi - ref_psi
