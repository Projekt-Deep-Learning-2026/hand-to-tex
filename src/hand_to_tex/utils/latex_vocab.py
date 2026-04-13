from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class LatexVocab:
    """Vocabulary for tokenization and encoding of LaTeX expressions.

    Provides mapping between LaTeX tokens and integer IDs,
    along with tokenization of LaTeX expression strings.

    Fields
    ------
    - tokens : dict[str, list[str]]
        Dictionary of tokens grouped by category (e.g., 'greek', 'operators').
        Primarily for display purposes.
    - _encode : dict[str, int]
        Internal mapping from token string to integer ID.
    - _decode : dict[int, str]
        Internal mapping from integer ID to token string.
    - EOS : int
        ID of end-of-sequence token `<EOS>`.
    - SOS : int
        ID of start-of-sequence token `<SOS>`.
    - PAD : int
        ID of padding token `<PAD>`.
    - UNK : int
        ID returned for tokens not found in vocabulary (`<UNK>`).

    Examples
    --------
    >>> vocab = LatexVocab.load("vocab.json")
    >>> vocab.encode("\\alpha")
    42
    >>> vocab.decode(42)
    '\\alpha'
    >>> vocab.encode_expr("x + y")
    [10, 50, 11]
    """

    tokens: dict[str, list[str]]
    _encode: dict[str, int]
    _decode: dict[int, str]

    # IDs of required special tokens for sequence processing.
    EOS: int
    SOS: int
    PAD: int
    UNK: int
    # Required special-token strings that must exist in every vocab file.
    required_keys = frozenset(["<EOS>", "<SOS>", "<PAD>", "<UNK>"])

    # Regex for LaTeX command tokenization.
    # Source: https://arxiv.org/pdf/2404.10690 (CROHME dataset paper)
    _COMMAND_RE: re.Pattern[str] = re.compile(
        r"\\("
        + "|".join(
            [
                r"mathbb\{[a-zA-Z]\}",
                r"begin\{[a-z]+\}",
                r"end\{[a-z]+\}",
                r"[a-zA-Z]+",
                r".",
            ]
        )
        + r")"
    )

    @staticmethod
    def load(path: Path | str) -> LatexVocab:
        """Create a vocabulary with given `.json` file path that contains
        categorised tokens (category_name -> list of tokens).
        For example `'digits' -> [0, 1, 2 ... 8, 9]`

        The loaded vocabulary must contain all required special tokens:
        `<EOS>`, `<SOS>`, `<PAD>`, `<UNK>`.

        Parameters
        ----------
        path : Path | str
            Path to the `.json` file

        Returns
        -------
        LatexVocab object for tokenization-detokenization of latex expressions

        Raises
        ------
        ValueError if any of the required keys are missing
        """
        with open(path) as f:
            data = json.load(f)

        token_list = []
        tokens = {}
        for cat, ts in data.items():
            if isinstance(ts, str):
                ts = [ts]
            tokens[cat] = ts
            token_list += ts

        _encode = {token: idx for idx, token in enumerate(token_list)}
        _decode = dict(enumerate(token_list))

        if not LatexVocab.required_keys.issubset(token_list):
            raise ValueError(
                f"Provided vocab at: {path} doesn't contain \
                            required tokens {str(LatexVocab.required_keys)}"
            )
        else:
            EOS, SOS = _encode["<EOS>"], _encode["<SOS>"]
            PAD, UNK = _encode["<PAD>"], _encode["<UNK>"]

        return LatexVocab(tokens, _encode, _decode, EOS, SOS, PAD, UNK)

    @staticmethod
    def default() -> LatexVocab:
        """Initialise LatexVocab using default vocab.json."""
        return LatexVocab.load(Path("data/assets/vocab.json"))

    def encode(self, token: str) -> int:
        """Convert a single token to its integer ID.

        Parameters
        ----------
        token : str
            Token string to encode.

        Returns
        -------
        int
            Integer ID of the token, or `UNK` ID if not in vocabulary.
        """
        return self._encode.get(token, self.UNK)

    def decode(self, token_id: int) -> str:
        """Convert an integer ID to its token string.

        Parameters
        ----------
        token_id : int
            Integer ID to decode.

        Returns
        -------
        str
            Token string, or `<UNK>` if ID not in vocabulary.
        """
        return self._decode.get(token_id, "<UNK>")

    def encode_expr(self, expr: str) -> list[int]:
        """Tokenize and encode a LaTeX expression to a list of token IDs.

        Uses regex-based tokenization to split the expression into tokens,
        then encodes each token to its integer ID.

        Encoded sequence always starts with additional `SOS` and ends with `EOS`

        Parameters
        ----------
        expr : str
            LaTeX expression string (e.g., "x + \\frac{a}{b}").

        Returns
        -------
        list[int]
            List of token IDs.

        Raises
        ------
        ValueError
            If expression contains unparseable content after backslash.
        """
        tokens = []
        while expr:
            if expr[0] == "\\":
                if (match := self._COMMAND_RE.match(expr)) is None:
                    raise ValueError(f"LatexVocab couldn't parse expr: {expr}")
                t = match.group()
            else:
                t = expr[0]
            tokens.append(t)
            expr = expr[len(t) :]

        return [self.SOS] + [self.encode(t) for t in tokens] + [self.EOS]

    def decode_sequence(self, token_ids: list[int]) -> list[str]:
        """Decode a list of token IDs to token strings.

        Parameters
        ----------
        token_ids : list[int]
            List of integer token IDs.

        Returns
        -------
        list[str]
            List of token strings. Unknown IDs are decoded as `<UNK>`.
        """
        return [self.decode(t_id) for t_id in token_ids]

    def __len__(self) -> int:
        return len(self._encode)
