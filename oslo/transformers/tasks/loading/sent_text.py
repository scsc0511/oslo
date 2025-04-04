import itertools
from dataclasses import dataclass
from typing import Optional

import datasets
import pyarrow as pa

logger = datasets.util.logging.get_logger(__name__)


@dataclass
class SentTextConfig(datasets.BuilderConfig):
    """BuilderConfig for text files."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    chunksize: int = 10 << 20  # 10MB
    keep_linebreakds: bool = False


class SentText(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = SentTextConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """The 'data_files' kwarg in load_dataset() can be a str, List[str], Dict[str, str], or Dict[str, List[str]].
        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the 'datasets.Split' enum.
        """
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"files": files}
                )
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )
        return splits

    def _generate_tables(self, files):
        schema = pa.schema(
            self.config.features.type
            if self.config.features is not None
            else {"text": pa.string()}
        )
        for file_idx, file in enumerate(files):
            with open(file, "r", encoding=self.config.encoding) as f:
                while True:
                    batch = f.read(self.config.chunksize)
                    if not batch:
                        break
                    batch += "".join(itertools.takewhile(lambda x: x != "\n", f))
                    list_of_docs = [
                        doc.rstrip().replace("\n", " ")
                        for doc in batch.split("\n\n")
                        if doc
                    ]
                    pa_table = pa.Table.from_arrays(
                        [pa.array(list_of_docs)], schema=schema
                    )
                    # Uncomment for debugging (will print the Arrow table size and elements)
                    # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                    # logger.warning("\n".join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                    yield file_idx, pa_table
