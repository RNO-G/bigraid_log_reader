import struct
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from pathlib import Path


class LogReader:

    @dataclass
    class Tag:
        name: str
        index: int
        type: int
        dtype: int

    def __init__(self, tagfile: Path | str):
        self._tagfile = Path(tagfile)
        if not self._tagfile.exists():
            raise FileNotFoundError(self._tagfile)

        self._floatfile = self._tagfile.parent / self._tagfile.name.replace("(Tagname)", "(Float)")
        if not self._floatfile.exists():
            raise FileNotFoundError(self._floatfile)

        self._tags = {}
        for name, index, type, dtype in self._iter_file(self._tagfile, "<256s4s2s1s"):
            tag = LogReader.Tag(
                name=name.decode().strip(),
                index=int(index.decode()),
                type=int(type.decode()),
                dtype=int(dtype.decode())
            )
            self._tags[tag.index] = tag

    @staticmethod
    def _iter_file(file_path: Path, fmt: str):
        with file_path.open('rb') as f:
            # Skip header
            while f.read(1) != b'\r':
                pass

            # Read individual records
            sz = struct.calcsize(fmt)
            while (k := f.read(1)) not in [b'\x1a', b'']:
                r = f.read(sz)
                try:
                    yield struct.unpack(fmt, r)
                except struct.error as e:
                    print(f"Error decoding record: err='{e}', raw='{r}', sep='{k}'")
                    return

    def __iter__(self):
        for time_str, msec_str, tag_index_str, value, status, marker, internal in self._iter_file(self._floatfile, '<16s3s5sdcci'):
            time = datetime.strptime(time_str.decode(), "%Y%m%d%H:%M:%S")
            time = time.replace(microsecond= int(msec_str.decode()) * 1000)

            tag_index = int(tag_index_str.decode())

            tag = self._tags.get(tag_index, "UNKNOWN")
            yield time, tag, value, status, marker, internal

    def as_df(self):
        data = [(time, tag.name, value) for time, tag, value, _, _, _ in self]
        df = pd.DataFrame(data, columns=['time', 'tag', 'value'])
        return df.pivot(index='time', columns='tag', values='value')

