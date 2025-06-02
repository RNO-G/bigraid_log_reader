from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Self

import pandas as pd

from log_reader import LogReader


class MultiLogReader:

    def __init__(
            self,
            log_readers: List[LogReader],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None):
        self._readers = log_readers
        self._start_date = start_date
        self._end_date = end_date

    @classmethod
    def find_files(
            cls,
            folder: Union[Path | str],
            start_date: Union[datetime, str],
            end_date: Optional[Union[datetime, str]] = None
    ) -> Self:
        folder = Path(folder)
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)

        def _expand_date(date):
            return datetime(year=date.year,
                            month=date.month,
                            day=date.day,
                            hour=23,
                            minute=59,
                            second=59)
        if end_date is None:
            end_date = _expand_date(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
                end_date = _expand_date(end_date)

        readers = []
        for file in folder.glob("*(Tagname).DAT"):
            file_date = datetime.strptime(file.name[:10], "%Y %m %d")
            if start_date <= file_date <= end_date:
                readers.append(LogReader(file))

        return cls(readers, start_date, end_date)

    def as_df(self):
        frames = [lr.as_df() for lr in self._readers]
        df = pd.concat(frames).sort_index()
        if self._start_date is not None:
            df = df[df.index >= self._start_date]
        if self._end_date is not None:
            df = df[df.index <= self._end_date]
        return df
