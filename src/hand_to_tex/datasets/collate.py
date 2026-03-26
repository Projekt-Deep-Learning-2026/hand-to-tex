from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from hand_to_tex.utils import LatexVocab


class HMECollateFunction:
    def __init__(self, vocab: LatexVocab):
        """Creates collate function suited for HMEDataset class

        Parameters
        ----------
        vocab : LatexVocab
            Vocabulary that will be used for padding
        """
        self.pad_idx = vocab.PAD

    def __call__(
        self, batch: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, list[int], Tensor, list[int]]:
        """Collate a list of variable-length samples into padded batch tensors.

        Parameters
        ----------
        batch : list[tuple[Tensor, Tensor]]
            List of `(features, tokens)` pairs where:
            - `features` has shape `(T_feat, F)`
            - `tokens` has shape `(T_tok,)`

        Returns
        -------
        tuple[Tensor, list[int], Tensor, list[int]]
            A 4-element tuple:
            - `padded_ft`: tensor `(B, max_T_feat, F)` padded with `0.0`
            - `ft_lengths`: original feature lengths for each sample
            - `padded_ts`: tensor `(B, max_T_tok)` padded with `self.pad_idx`
            - `ts_lengths`: original token lengths for each sample
        """

        features = [ft for ft, _ts in batch]
        tokens = [ts for _ft, ts in batch]

        ft_lengths = [f.size(0) for f in features]
        ts_lengths = [t.size(0) for t in tokens]

        padded_ft = pad_sequence(features, True, 0.0)
        padded_ts = pad_sequence(tokens, True, self.pad_idx)

        return padded_ft, ft_lengths, padded_ts, ts_lengths
