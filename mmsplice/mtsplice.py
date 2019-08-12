from pkg_resources import resource_filename
from keras.models import load_model
from concise.preprocessing import encodeDNA
from mmsplice.exon_dataloader import SeqSpliter


MTSPLICE = resource_filename('mmsplice', 'models/mtsplice.h5')

tissue_names = [
    'Retina - Eye', 'RPE/Choroid/Sclera - Eye', 'Subcutaneous - Adipose',
    'Visceral (Omentum) - Adipose', 'Adrenal Gland', 'Aorta - Artery',
    'Coronary - Artery', 'Tibial - Artery', 'Bladder', 'Amygdala - Brain',
    'Anterior cingulate - Brain', 'Caudate nucleus - Brain',
    'Cerebellar Hemisphere - Brain', 'Cerebellum - Brain', 'Cortex - Brain',
    'Frontal Cortex - Brain', 'Hippocampus - Brain', 'Hypothalamus - Brain',
    'Nucleus accumbens - Brain', 'Putamen - Brain',
    'Spinal cord (C1) - Brain', 'Substantia nigra - Brain',
    'Mammary Tissue - Breast', 'EBV-xform lymphocytes - Cells',
    'Leukemia (CML) - Cells', 'Xform. fibroblasts - Cells',
    'Ectocervix - Cervix', 'Endocervix - Cervix', 'Sigmoid - Colon',
    'Transverse - Colon', 'Gastroesoph. Junc. - Esophagus',
    'Mucosa - Esophagus', 'Muscularis - Esophagus', 'Fallopian Tube',
    'Atrial Appendage - Heart', 'Left Ventricle - Heart', 'Cortex - Kidney',
    'Liver', 'Lung', 'Minor Salivary Gland', 'Skeletal - Muscle',
    'Tibial - Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
    'Not Sun Exposed - Skin', 'Sun Exposed (Lower leg) - Skin',
    'Ileum - Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
    'Uterus', 'Vagina', 'Whole Blood'
]


class MTSplice:
    """
    Load modules of mtsplice model, perform prediction on batch of dataloader.

    Args:
      acceptor_intronM: acceptor intron model,
        score acceptor intron sequence.
      acceptorM: accetpor splice site model. Score acceptor sequence
        with 50bp from intron, 3bp from exon.
      exonM: exon model, score exon sequence.
      donorM: donor splice site model, score donor sequence
        with 13bp in the intron, 5bp in the exon.
      donor_intronM: donor intron model, score donor intron sequence.
    """

    def __init__(self, mtspliceM=MTSPLICE, seq_spliter=None):
        self.mtspliceM = load_model(MTSPLICE)
        self.spliter = seq_spliter or SeqSpliter()

    def predict_on_batch(self, batch):
        '''
        Performe prediction on batch of dataloader.

        Args:
          batch: batch of dataloader.

        Returns:
          np.matrix of tissue predictions as [[tissues]]
        '''
        return self.mtspliceM.predict([batch['acceptor'], batch['donor']])

    def predict(self, seq, overhang=(300, 300)):
        """
        Performe prediction of overhanged exon sequence string.

        Args:
          seq (str):  sequence of overhanged exon.
          overhang (Tuple[int, int]): overhang of seqeunce.

        Returns:
          np.array of modular predictions
          as [[tissues]]
        """
        batch = self.spliter.split_tissue_seq(seq, overhang)
        batch = {k: encodeDNA([v]) for k, v in batch.items()}
        return self.predict_on_batch(batch)[0]
