from dataclasses import dataclass
from typing import Hashable, List, Tuple, Optional, Union

import pandas as pd

from fips.indices import sanitize_index


@dataclass
class Block:
    name: str | Hashable
    data: pd.Series

    def __post_init__(self):
        if not isinstance(self.data, pd.Series):
            raise ValueError(f"Block '{self.name}' data must be a pandas Series.")
        self.data.name = self.name  # force series name to block name


class Vector:
    def __init__(self, blocks: List[Block]):
        self.blocks = {b.name: b for b in blocks}
        self.block_order = [b.name for b in blocks]
        self._assemble()
        self.size = len(self.data)

    def _assemble(self):
        block_list = []
        for name in self.block_order:
            s = self.blocks[name].data.copy()
            if isinstance(s.index, pd.MultiIndex):
                # Prepend block level to existing MultiIndex
                original_names = s.index.names
                new_names = ['block'] + list(original_names)
                new_levels = [[name]] + list(s.index.levels)
                new_codes = [pd.array([0] * len(s), dtype='int8')] + [s.index.codes[i] for i in range(s.index.nlevels)]
                s.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=new_names)
            else:
                # Simple index case
                original_names = [s.index.name or 'index']
                new_names = ['block'] + list(original_names)
                s.index = pd.MultiIndex.from_product([[name], s.index], names=new_names)
            block_list.append(s)
        self.data = pd.concat(block_list)

    def get_block_slice(self, block_name: str) -> slice:
        start = 0
        for name in self.block_order:
            length = len(self.blocks[name].data)
            if name == block_name:
                return slice(start, start + length)
            start += length
        raise KeyError(f"Block {block_name} not found.")

    @property
    def names(self) -> list[str | Hashable]:
        return self.block_order

    def __getitem__(self, key: Hashable) -> pd.Series:
        return self.blocks[key].data
    
    def __iter__(self):
        return iter(self.blocks)


def prepare_vector(vector: Union[Vector, pd.Series], default_name: str, float_precision: Optional[int]
                   ) -> Tuple[Vector, bool]:
    """Helper to normalize input vectors into Vector objects and sanitize indices."""
    promote = False
    if isinstance(vector, pd.Series):
        v_clean = vector.copy()
        v_clean.index = sanitize_index(v_clean.index, float_precision)
        name = v_clean.name if v_clean.name else default_name
        vector_obj = Vector([Block(name, v_clean)])
        promote = True
    else:
        vector_obj = vector
        vector_obj.data.index = sanitize_index(vector_obj.data.index, float_precision)
    return vector_obj, promote