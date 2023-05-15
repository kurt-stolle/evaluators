from typing import Any, Iterable

from detectron2.data import MetadataCatalog

CATALOG = MetadataCatalog


def read_metadata(names: str | Iterable[str]) -> dict[str, Any]:
    if isinstance(names, str):
        name = names
    else:
        name = next(iter(names))
    meta = CATALOG.get(name)

    return meta.as_dict()
