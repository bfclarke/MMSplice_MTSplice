import logging
from pkg_resources import resource_filename
import pandas as pd
import pyranges
from kipoi.data import SampleIterator
from kipoiseq.extractors import MultiSampleVCF, SingleVariantMatcher
from mmsplice.utils import pyrange_remove_chr_from_chrom_annotation
from mmsplice.exon_dataloader import ExonSplicingMixin

from allel import read_vcf_headers
from pyfaidx import Fasta

logger = logging.getLogger('mmsplice')

prebuild_annotation = {
    'grch37': resource_filename('mmsplice', 'models/grch37_exons.csv.gz'),
    'grch38': resource_filename('mmsplice', 'models/grch38_exons.csv.gz')
}


def read_exon_pyranges(gtf_file, overhang=(100, 100), first_last=True):
    '''
    Read exon as pyranges from gtf_file

    Args:
      gtf_file: gtf file from ensembl/gencode.
      overhang: padding of exon to match variants.
      first_last: set overhang of first and last exon of the gene to zero
        so seq intergenic region will not be processed.
    '''
    df_gtf = pyranges.read_gtf(gtf_file).df
    df_exons = df_gtf[df_gtf['Feature'] == 'exon']
    df_exons = df_exons[['Chromosome', 'Start', 'End', 'Strand',
                         'exon_id', 'gene_id', 'gene_name', 'transcript_id']]

    if first_last:
        df_genes = df_gtf[df_gtf['Feature'] == 'transcript']
        df_genes.set_index('transcript_id', inplace=True)
        df_genes = df_genes.loc[df_exons['transcript_id']]
        df_genes.set_index(df_exons.index, inplace=True)

        starting = df_exons['Start'] == df_genes['Start']
        ending = df_exons['End'] == df_genes['End']

        df_exons.loc[:, 'left_overhang'] = ~starting * overhang[0]
        df_exons.loc[:, 'right_overhang'] = ~ending * overhang[1]

        df_exons.loc[:, 'Start'] -= df_exons['left_overhang']
        df_exons.loc[:, 'End'] += df_exons['right_overhang']

    return pyranges.PyRanges(df_exons)


class SplicingVCFMixin(ExonSplicingMixin):
    def __init__(self, pr_exons, annotation, fasta_file, variant_file,
                 split_seq=True, encode=True,
                 overhang=(100, 100), seq_spliter=None,
                 tissue_specific=False, tissue_overhang=(300, 300),
                 interval_attrs=tuple()):
        super().__init__(fasta_file, variant_file, split_seq, encode,
                         overhang, seq_spliter,
                         tissue_specific, tissue_overhang)
        self.pr_exons = pr_exons
        self.annotation = annotation
        self.variant_file = variant_file
        self.fasta_file = fasta_file

        # TODO: This is quite the hack!
        try:
            self.variants
        except NameError:
            self.variants = MultiSampleVCF(variant_file)

        # TODO: This is quite the hack!
        try:
            self.variants_chroms
        except NameError:
            self.variants_chroms = self.variants.seqnames

        self._check_chrom_annotation()
        # TODO: change to MultiVariantsMatcher?
        self.matcher = SingleVariantMatcher(
            variant_file, pranges=self.pr_exons,
            interval_attrs=interval_attrs
        )
        self._generator = iter(self.matcher)

    def _check_chrom_annotation(self):
        fasta_chroms = set(Fasta(self.fasta_file).keys())
        variants_chroms = set(self.variants_chroms)

        if not fasta_chroms.intersection(variants_chroms):
            raise ValueError(
                'Fasta chrom names do not match with vcf chrom names')

        if self.annotation == 'grch37' or self.annotation == 'grch38':
            chr_annotaion = any(chrom.startswith('chr')
                                for chrom in variants_chroms)
            if not chr_annotaion:
                self.pr_exons = pyrange_remove_chr_from_chrom_annotation(
                    self.pr_exons)

        gtf_chroms = set(self.pr_exons.Chromosome)
        if not gtf_chroms.intersection(variants_chroms):
            raise ValueError(
                'GTF chrom names do not match with vcf chrom names')


class SplicingVCFDataloader(SplicingVCFMixin, SampleIterator):
    """
    Load genome annotation (gtf) file along with a vcf file,
      return reference sequence and alternative sequence.

    Args:
      gtf: gtf file. Can be dowloaded from ensembl/gencode.
        Filter for protein coding genes.
      fasta_file: file path; Genome sequence
      vcf_file: vcf file, each line should contain one
        and only one variant, left-normalized
      split_seq: whether or not already split the sequence
        when loading the data. Otherwise it can be done in the model class.
      endcode: if split sequence, should it be one-hot-encoded.
      overhang: overhang of exon to fetch flanking sequence of exon.
      seq_spliter: SeqSpliter class instance specific how to split seqs.
         if None, use the default arguments of SeqSpliter
      tissue_specific: tissue specific predicts
      tissue_overhang: overhang of exon to fetch flanking sequence of
        tissue specific model.
    """

    def __init__(self, gtf, fasta_file, variant_file,
                 split_seq=True, encode=True,
                 overhang=(100, 100), seq_spliter=None,
                 tissue_specific=False, tissue_overhang=(300, 300)):
        pr_exons = self._read_exons(gtf, overhang)

        variant_file_type = self._check_variant_file_type(variant_file)
        if variant_file_type == 'unknown':
            raise ValueError(
                'variant_file must be of type ybgen, vcf, or vcf.gz')

        if variant_file_type == 'vcf':
            self.variants = MultiSampleVCF(variant_file)
            self.variants_chroms = set(self.variants.seqnames)
            self.samples = read_vcf_headers(variant_file).samples
        else:
            self.variants = MultiSampleBGEN(variant_file)
            self.variants_chroms = self.variants.chroms()
            self.samples = self.variants.samples()

        super().__init__(pr_exons, gtf, fasta_file, variant_file,
                         split_seq, encode, overhang, seq_spliter,
                         tissue_specific, tissue_overhang,
                         interval_attrs=('left_overhang', 'right_overhang',
                                         'exon_id', 'gene_id',
                                         'gene_name', 'transcript_id'))

        df_exons = pd.concat(
            [pr_exons[chrom].as_df() for chrom in self.variants_chroms]
        )
        self.df_exons = df_exons.rename(columns={
            'Chromosome': 'chrom',
            'Start': 'start',
            'End': 'end',
            'Strand': 'strand'
        })
        self._generator = (
            (exon, sample, phase) for _, exon in self.df_exons.iterrows()
            for sample in self.samples
            for phase in (0, 1)
        )

    def _read_exons(self, gtf, overhang=(100, 100)):
        if gtf in prebuild_annotation:
            if overhang != (100, 100):
                logger.warning('Overhang argument will be ignored'
                               ' for prebuild annotation.')
            df = pd.read_csv(prebuild_annotation[gtf])
            df['Start'] -= 1  # convert prebuild annotation to 0-based
            return pyranges.PyRanges(df)
        else:
            return read_exon_pyranges(gtf, overhang=overhang)

    @staticmethod
    def _check_variant_file_type(variant_file):
        split_name = variant_file.split('.')
        ext1 = split_name[-1] if len(split_name) >= 2 else None
        ext2 = split_name[-2] if len(split_name) >= 3 else None
        if ext1 == 'bgen':
            return 'bgen'
        elif ext1 == 'vcf' or (ext1 == 'gz' and ext2 == 'vcf'):
            return 'vcf'
        else:
            return 'unknown'

    def __next__(self):
        exon, sample, phase = next(self._generator)
        overhang = (exon.left_overhang, exon.right_overhang)
        exon = exon.copy()
        exon.start += overhang[0]
        exon.end -= overhang[1]
        return self._next(exon, sample, phase, overhang)

    def __iter__(self):
        return self
