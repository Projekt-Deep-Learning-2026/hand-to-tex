from pathlib import Path
import torch
from torch.utils.data.dataset import Dataset
from typing import Optional, Callable
import numpy as np

from .ink_data import InkData, Trace
from ..utils import LatexVocab


class HMEDataset(Dataset):
    """Handwritten Mathematical Expressions dataset for processing
    `.inkml` files
    """
    EPS: float = 1e-8

    def __init__(
                self,
                root: Path | str,
                vocab: Optional[LatexVocab] = None,
                transform: Optional[Callable] = None,
            ):
        """Creates the dataset

        Parameters
        ----------

        root : Path | str
            Directory of all dataset files
        vocab : LatexVocab | None
            Vocabulary for tokenization, default `LatexVocab.default()`
        """

        self.root = Path(root)
        self.filenames = sorted(list(self.root.rglob('*.inkml')))

        self.vocab = LatexVocab.default() if vocab is None else vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        # p = self.filenames[idx]
        return InkData.load(self.filenames[idx])
        # raise NotImplementedError

    @staticmethod
    def extract_features(ink: InkData) -> torch.Tensor:
        """Extracts features of an InkData object

        Returns
        -------

        torch.Tensor containing following features:
            - x_norm, y_norm : float 
                Normalised x,y coordinates of a TracePoint
            - dx, dy, dt : float
                V

        """
        data = HMEDataset._flatten_traces(ink.traces)
        data_norm = HMEDataset._normalise_data(data)
        # X, Y, T, TraceID = [], [], [], []
        # for t_id, trace in enumerate(ink.traces):
        #     for x, y, t in trace:
        #         X.append(x)
        #         Y.append(y)
        #         T.append(t)
        #         TraceID.append(t_id)
        # X = np.array((x for x,_,_ in trace) for trace in ink.traces)
        pass

    @staticmethod
    def _flatten_traces(traces: list[Trace]) -> torch.Tensor:
        """Returns a tensor of elements (x, y, t, trace_id) for InkData traces

        Parameters
        ----------

        traces : Traces
            List of Trace elements obtained from `InkData.traces`
        """
        result = []

        for t_id, t in enumerate(traces):
            points = torch.tensor(t, dtype=torch.float32)
            ids = torch.full((len(t), 1), t_id, dtype=torch.float32)

            result.append(torch.cat([points, ids], dim=1))

        return torch.cat(result)

    @staticmethod
    def _normalise_data(data: torch.Tensor) -> torch.Tensor:
        """Perform normalization on data tensor
        """
        xy = data[:, :2]
        xy_min = xy.min(dim=0, keepdim=True).values
        xy_max = xy.max(dim=0, keepdim=True).values
        t0 = data[:, 2].min(dim=0, keepdim=True).values

        data[:, :2] = (xy - xy_min) / (xy_max - xy_min + HMEDataset.EPS)
        data[:, 2] -= t0

        return data



if __name__ == "__main__":
    d = HMEDataset('data/train')
    ink = d[4]
    HMEDataset.extract_features(ink)
