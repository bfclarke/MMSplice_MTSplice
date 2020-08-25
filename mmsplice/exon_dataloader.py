import logging
import pandas as pd
import numpy as np
from kipoiseq.dataclasses import Interval, Variant
from kipoi.data import Dataset
from kipoiseq.extractors import VariantSeqExtractor
from mmsplice.utils import encodeDNA

from pyfaidx import Fasta, FastaVariant
from cyvcf2 import VCF
from allel import read_vcf, read_vcf_headers

logger = logging.getLogger('mmsplice')


class PyfaidxSeqExtractor:
    def __init__(self, fasta_file, vcf_file):
        # TODO: Get separate sequences for each phase
        self.fasta_variants = {
            sample: FastaVariant(
                fasta_file,
                vcf_file,
                sample=sample,
                het=True,
                hom=True
            )
            for sample in read_vcf_headers(vcf_file).samples
        }

        self.fasta = Fasta(fasta_file)

    def extract(self, interval, sample, anchor=0, fixed_len=True, strand='+'):
        # TODO: Take anchor and fixed_len into account
        extractor = self.fasta_variants.get(sample, self.fasta)
        # __import__("pdb").set_trace()
        # try:
        consensus = extractor[interval.chrom][interval.start:interval.end]
        # except ValueError:
        #     __import__("pdb").set_trace()
        if strand == '-':
            consensus = consensus.complement.reverse

        return consensus.seq


# TODO: Rewrite this with VariantSeqExtractor for extract(), cyvcf2 for VCF file
class MySeqExtractor():
    def __init__(self, fasta_file, vcf_file):
        # TODO: Get separate sequences for each phase
        self.fasta = Fasta(fasta_file)
        self.vcf = read_vcf(vcf_file, alt_number=1)
        # TODO: Is the next line correct?
        self.vcf['variants/POS'] -= 1  # Convert to 0-based
        self.samples = dict(zip(self.vcf['samples'],
                                range(len(self.vcf['samples']))))

    def extract(self, interval, sample, phase,
                anchor=0, fixed_len=True, strand='+'):
        # TODO: Take anchor and fixed_len into account
        consensus = self.fasta[interval.chrom][interval.start:interval.end]

        if sample is not None:
            # TODO: Precompute/cache variants relevant to each exon/interval?
            indices = np.where(np.all(np.stack([
                self.vcf['variants/POS'] >= interval.start,
                self.vcf['variants/POS'] < interval.end  # TODO: Should this be <= ?
            ]), axis=0))
            sample_index = self.samples[sample]
            phased_gt = [
                i for i in indices
                if self.vcf['calldata/GT'][i, sample, phase] == 1]
            # TODO: Convert consensus to list, replace where variants present, convert back to string
            consensus_list = list(consensus)
            for i in phased_gt:
                consensus_list[self.vcf['variants/POS'][i]] = self.vcf['variants/ALT']

        if strand == '-':
            # TODO: Implement this
            pass

        return consensus

    @staticmethod
    def reverse_complement(seq):
        # TODO: Implement
        pass


class SampleSeqExtractor(VariantSeqExtractor):
    def __init__(self, fasta_file, vcf_file):
        self.vcf = VCF(vcf_file)
        self.sample_indices = dict(zip(self.vcf.samples,
                                       range(len(self.vcf.samples))))

        super().__init__(fasta_file)

    def extract(self, interval, sample, phase, anchor,
                fixed_len=True, **kwargs):
        variants = []
        if sample is not None:
            if phase not in (0, 1):
                logger.error('phase argument must be in (0, 1)'
                             + 'if sample is not None')

            # Interval is  0-based, cyvcf2 positions are 1-based: need to add 1
            variants = self.get_sample_variants(
                self.vcf(
                    f'{interval.chrom}:'
                    + f'{interval.start + 1}-{interval.end + 1}'
                ),
                sample,
                phase
            )

        return super(SampleSeqExtractor, self).extract(
            interval, variants, anchor, fixed_len, **kwargs)

    def get_sample_variants(self, variants, sample, phase):
        sample_index = self.sample_indices[sample]
        return [
            Variant.from_cyvcf(v) for v in variants
            if v.genotypes[sample_index][phase]
        ]


class ExonVariantSeqExtrator:
    """
    Extracts sequence with the variant integrated. The lengths overhang
    are fixed irrelevant to variants, even if the variants are indels and
    is in introns, lengths overhang will adapt. If the variant is in the
    exon, the length of the alternative exon (with variant) might change
    for indels.
    """

    def __init__(self, fasta_file, vcf_file):
        self.variant_seq_extractor = SampleSeqExtractor(fasta_file, vcf_file)
        # self.fasta = self.variant_seq_extractor.fasta

    def extract(self, interval, sample_id=None, phase=None,
                overhang=(100, 100)):
        """
        Args:
          interval (pybedtools.Interval): zero-based interval of exon
            without overhang.
        """

        down_interval = Interval(
            interval.chrom, interval.start - overhang[0],
            interval.start, strand=interval.strand)
        up_interval = Interval(
            interval.chrom, interval.end,
            interval.end + overhang[1], strand=interval.strand)

        # DEBUG
        # print(f'Extracting down_seq for {down_interval}')
        down_seq = self.variant_seq_extractor.extract(
            down_interval, sample_id, phase, anchor=interval.start)
        # DEBUG
        # print(f'Extracting up_seq for {up_interval}')
        up_seq = self.variant_seq_extractor.extract(
            up_interval, sample_id, phase, anchor=interval.start)

        # DEBUG
        # print(f'Extracting exon_seq for {interval}')
        # DEBUG
        if interval.start >= interval.end:
            __import__("pdb").set_trace()
        exon_seq = self.variant_seq_extractor.extract(
            interval, sample_id, phase, anchor=0, fixed_len=False)

        if interval.strand == '-':
            down_seq, up_seq = up_seq, down_seq

        return down_seq + exon_seq + up_seq

# TODO: generalized seq_spliter


class SeqSpliter:
    """
    Splits given seq for each modules.

    length arguments of the __init__ function refer to the prefered
    sequence  length of the models

    Args:
      exon_cut_l: number of bp to cut out at the begining of an exon
      exon_cut_r: number of bp to cut out at the end of an exon
        (cut out the part that is considered as acceptor site or donor site)
      acceptor_intron_cut: number of bp to cut out at the end of
        acceptor intron that consider as acceptor site
      donor_intron_cut: number of bp to cut out at the end of donor intron
        that consider as donor site
      acceptor_intron_len: length in acceptor intron to consider
        for acceptor site model
      acceptor_exon_len: length in acceptor exon to consider
        for acceptor site model
      donor_intron_len: length in donor intron to consider for donor site model
      donor_exon_len: length in donor exon to consider for donor site model
    """

    def __init__(self, exon_cut_l=0, exon_cut_r=0,
                 acceptor_intron_cut=6, donor_intron_cut=6,
                 acceptor_intron_len=50, acceptor_exon_len=3,
                 donor_exon_len=5, donor_intron_len=13,
                 tissue_acceptor_intron=300, tissue_acceptor_exon=100,
                 tissue_donor_intron=300, tissue_donor_exon=100,
                 pattern_warning=False):
        self.exon_cut_l = exon_cut_l
        self.exon_cut_r = exon_cut_r
        self.acceptor_intron_cut = acceptor_intron_cut
        self.donor_intron_cut = donor_intron_cut
        self.acceptor_intron_len = acceptor_intron_len
        self.acceptor_exon_len = acceptor_exon_len
        self.donor_exon_len = donor_exon_len
        self.donor_intron_len = donor_intron_len
        self.tissue_acceptor_intron = tissue_acceptor_intron
        self.tissue_acceptor_exon = tissue_acceptor_exon
        self.tissue_donor_intron = tissue_donor_intron
        self.tissue_donor_exon = tissue_donor_exon
        self.pattern_warning = pattern_warning

    def split(self, seq, overhang, exon_row='', pattern_warning=True):
        """
        Split seqeunce for each module.

        Args:
          seq: seqeunce to split.
          overhang: (intron_length acceptor side, intron_length donor side) of
                    the input sequence
        """
        pattern_warning = self.pattern_warning and pattern_warning

        intronl_len, intronr_len = overhang
        assert intronl_len <= len(seq), "Input sequence acceptor intron" \
            " length cannot be longer than the input sequence"
        assert intronr_len <= len(seq), "Input sequence donor intron length" \
            " cannot be longer than the input sequence"

        # need to pad N if left seq not enough long
        lackl = self.acceptor_intron_len - intronl_len
        if lackl >= 0:
            seq = "N" * (lackl + 1) + seq
            intronl_len += lackl + 1
        lackr = self.donor_intron_len - intronr_len
        if lackr >= 0:
            seq = seq + "N" * (lackr + 1)
            intronr_len += lackr + 1

        acceptor_intron = seq[:intronl_len - self.acceptor_intron_cut]

        acceptor_start = intronl_len - self.acceptor_intron_len
        acceptor_end = intronl_len + self.acceptor_exon_len
        acceptor = seq[acceptor_start: acceptor_end]

        exon_start = intronl_len + self.exon_cut_l
        exon_end = -intronr_len - self.exon_cut_r
        exon = seq[exon_start: exon_end]

        donor_start = -intronr_len - self.donor_exon_len
        donor_end = -intronr_len + self.donor_intron_len
        donor = seq[donor_start: donor_end]

        donor_intron = seq[-intronr_len + self.donor_intron_cut:]

        if not exon:
            exon = 'N'

        if pattern_warning:
            if donor[self.donor_exon_len:self.donor_exon_len + 2] != "GT" \
               and overhang[1]:
                logger.warning('None GT donor: %s' % str(exon_row))

            if acceptor[self.acceptor_intron_len - 2:self.acceptor_intron_len] != "AG" \
               and overhang[0]:
                logger.warning('None AG acceptor: %s' % str(exon_row))

        splits = {
            "acceptor_intron": acceptor_intron,
            "acceptor": acceptor,
            "exon": exon,
            "donor": donor,
            "donor_intron": donor_intron
        }

        return splits

    def split_tissue_seq(self, seq, overhang):
        """
        Split seq for tissue specific predictions
        Args:
          seq: seqeunce to split
          overhang: (intron_length acceptor side, intron_length donor side) of
                    the input sequence
        """
        (acceptor_intron, donor_intron) = overhang

        assert acceptor_intron <= len(seq), "Input sequence acceptor intron" \
            " length cannot be longer than the input sequence"
        assert donor_intron <= len(seq), "Input sequence donor intron length" \
            " cannot be longer than the input sequence"

        # need to pad N if seq not enough long
        diff_acceptor = acceptor_intron - self.tissue_acceptor_intron
        if diff_acceptor < 0:
            seq = "N" * abs(diff_acceptor) + seq
        elif diff_acceptor > 0:
            seq = seq[diff_acceptor:]

        diff_donor = donor_intron - self.tissue_donor_intron
        if diff_donor < 0:
            seq = seq + "N" * abs(diff_donor)
        elif diff_donor > 0:
            seq = seq[:-diff_donor]

        return {
            'acceptor': seq[:self.tissue_acceptor_intron
                            + self.tissue_acceptor_exon],
            'donor': seq[-self.tissue_donor_exon
                         - self.tissue_donor_intron:]
        }


class ExonSplicingMixin:
    """
    Dataloader to run mmsplice on specific set of variant-exon pairs.
    This class will be inherited both by ExonDataset, which is the dataloader
    for variants provided in a csv file with match exons provided, and by
    SplicingVCFDataloader, which takes variants in vcf format.

    Args:
      fasta_file: fasta file to fetch exon sequences.
      split_seq: whether or not already split the sequence
        when loading the data.
      endcode: if split sequence, should it be one-hot-encoded.
      overhang: overhang of exon to fetch flanking sequence of exon.
      seq_spliter: SeqSpliter class instance specific how to split seqs.
      tissue_specific: provide sequences for tissue specific prediction
      tissue_overhang: overhang of exon to fetch flanking sequence of
        tissue specific model. The current model only accepts 300.
    """
    optional_metadata = ('exon_id', 'gene_id', 'gene_name',
                         'transcript_id', 'junction', 'side')

    def __init__(self, fasta_file, vcf_file, split_seq=True, encode=True,
                 overhang=(100, 100), seq_spliter=None,
                 tissue_specific=False, tissue_overhang=(300, 300)):
        self.fasta_file = fasta_file
        self.split_seq = split_seq
        self.encode = encode
        self.overhang = overhang
        self.spliter = seq_spliter or SeqSpliter()
        self.vseq_extractor = ExonVariantSeqExtrator(fasta_file, vcf_file)
        # self.fasta = self.vseq_extractor.fasta
        self.tissue_specific = tissue_specific
        self.tissue_overhang = tissue_overhang

    def _next(self, exon, sample, phase, overhang=None, mask_module=None):
        overhang = overhang or self.overhang

        inputs = {
            'seq': self.vseq_extractor.extract(
                Interval(
                    exon.chrom, exon.start - overhang[0],
                    exon.end + overhang[1], strand=exon.strand
                ),
            ).upper(),
            'mut_seq': self.vseq_extractor.extract(
                Interval(
                    exon.chrom, exon.start - overhang[0],
                    exon.end + overhang[1], strand=exon.strand
                ),
                sample_id=sample,
                phase=phase
            ).upper()
        }

        if self.tissue_specific:
            tissue_overhang = (
                0 if overhang[0] == 0 else self.tissue_overhang[0],
                0 if overhang[1] == 0 else self.tissue_overhang[1]
            )
            inputs['tissue_seq'] = self.vseq_extractor.extract(
                exon, sample, overhang=tissue_overhang,
                sample_id=sample, phase=phase).upper()

        if exon.strand == '-':
            overhang = (overhang[1], overhang[0])
            if self.tissue_specific:
                tissue_overhang = (tissue_overhang[1], tissue_overhang[0])

        if self.split_seq:
            inputs['seq'] = self.spliter.split(inputs['seq'], overhang, exon)
            inputs['mut_seq'] = self.spliter.split(inputs['mut_seq'], overhang,
                                                   exon, pattern_warning=False)
            if mask_module:
                for i in mask_module:
                    if i in inputs['seq']:
                        inputs['seq'][i] = 'N' * len(inputs['seq'][i])
                        inputs['mut_seq'][i] = 'N' * len(inputs['mut_seq'][i])
                    else:
                        raise ValueError('%s is not in mmsplice modules' % i)

            if self.tissue_specific:
                inputs['tissue_seq'] = self.spliter.split_tissue_seq(
                    inputs['tissue_seq'], tissue_overhang)
            if self.encode:
                inputs = {k: self._encode_seq(v) for k, v in inputs.items()}

        return {
            'inputs': inputs,
            'metadata': {
                'exon': self._exon_to_dict(exon, overhang),
                'sample': sample,
                'phase': phase
            }
        }

    def batch_iter(self, batch_size=32, **kwargs):
        encode = self.encode
        self.encode = False

        for batch in super().batch_iter(batch_size, **kwargs):
            if encode:
                batch['inputs']['seq'] = self._encode_batch_seq(
                    batch['inputs']['seq'])
                batch['inputs']['mut_seq'] = self._encode_batch_seq(
                    batch['inputs']['mut_seq'])
            if self.tissue_specific:
                batch['inputs']['tissue_seq'] = self._encode_batch_seq(
                    batch['inputs']['tissue_seq'])

            yield batch

        self.encode = encode

    def _encode_batch_seq(self, batch):
        return {k: encodeDNA(v.tolist()) for k, v in batch.items()}

    def _encode_seq(self, seq):
        return {k: encodeDNA([v]) for k, v in seq.items()}

    def _variant_to_dict(self, variant):
        return {
            'chrom': variant.chrom,
            'pos': variant.pos,
            'ref': variant.ref,
            'alt': variant.alt,
            'annotation': str(variant)
        }

    def _exon_to_dict(self, exon, overhang):
        # return {
        #     'chrom': exon.chrom,
        #     'start': exon.start,
        #     'end': exon.end,
        #     'strand': exon.strand,
        #     'gene_id': None,
        #     'exon_id': None,
        #     'transcript_id': None,
        #     'left_overhang': overhang[0],
        #     'right_overhang': overhang[1],
        #     'annotation': str(exon),
        #     **exon.attrs
        # }
        return dict(exon)


class ExonDataset(ExonSplicingMixin, Dataset):
    """
    Dataloader to run mmsplice on specific set of variant-exon pairs
    provided by csv file.

    Args:
        exon_file: csv file specify exon-variant pairs with required
        columns of ('chrom', 'start', 'end', 'strand', 'pos', 'ref', 'alt')
        and optional columns of
        ('exon_id', 'gene_id', 'gene_name', 'transcript_id').
        fasta_file: fasta file to fetch exon sequences.
        split_seq: whether or not already split the sequence
        when loading the data. Otherwise it can be done in the model class.
        endcode: if split sequence, should it be one-hot-encoded.
        overhang: overhang of exon to fetch flanking sequence of exon.
        seq_spliter: SeqSpliter class instance specific how to split seqs.
    """

    exon_cols_mapping = {
        "hg19_variant_position": "pos",
        "variant_position": "pos",
        "POS": "pos",
        "reference": "ref",
        "variant": "alt",
        "REF": "ref",
        "ALT": "alt",
        "exon_start": "Exon_Start",
        "exon_end": "Exon_End",
        "start": "Exon_Start",
        "end": "Exon_End",
        "Stop": "Exon_End",
        "chr": "Chromosome",
        "chrom": "Chromosome",
        "seqnames": "Chromosome",
        "chromosome": "Chromosome",
        "CHR": "Chromosome",
        "strand": "Strand"
    }
    required_cols = ('Chromosome', 'Exon_Start', 'Exon_End',
                     'Strand', 'pos', 'ref', 'alt')

    def __init__(self, exon_file, fasta_file, split_seq=True, encode=True,
                 overhang=(100, 100), seq_spliter=None,
                 tissue_specific=False, tissue_overhang=(300, 300),
                 **kwargs):
        super().__init__(fasta_file, split_seq, encode, overhang, seq_spliter,
                         tissue_specific, tissue_overhang)
        self.exon_file = exon_file
        self.exons = self.read_exon_file(exon_file, **kwargs)
        self._check_chrom_annotation()

    @staticmethod
    def read_exon_file(exon_file, **kwargs):
        df = pd.read_csv(exon_file, **kwargs) \
               .rename(columns=ExonDataset.exon_cols_mapping)
        df['Chromosome'] = df['Chromosome'].astype('str')
        return df

    def _check_chrom_annotation(self):
        fasta_chroms = set(self.fasta.fasta.keys())
        exon_chroms = set(self.exons['Chromosome'])

        if not fasta_chroms.intersection(exon_chroms):
            raise ValueError(
                'Fasta chrom names do not match with vcf chrom names')

        for c in self.required_cols:
            if c not in self.exons.columns:
                raise ValueError('Required column "%s" are missings' % c)

    def __getitem__(self, idx):
        row = self.exons.iloc[idx]
        exon_attrs = {k: row[k] for k in self.optional_metadata if k in row}
        exon = Interval(row['Chromosome'], row['Exon_Start'] - 1,
                        row['Exon_End'], strand=row['Strand'],
                        attrs=exon_attrs)
        variant = Variant(row['Chromosome'], row['pos'],
                          row['ref'], row['alt'])
        return self._next(exon, variant)

    def __len__(self):
        return len(self.exons)
