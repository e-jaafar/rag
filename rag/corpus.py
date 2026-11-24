import hashlib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    heading: str
    start_line: int
    end_line: int
    excerpt: str

    @property
    def search_text(self) -> str:
        return f"{self.heading}\n{self.excerpt}"


def load_corpus(root: Path) -> list[Passage]:
    passages = []
    for path in sorted(root.rglob("*.md")):
        source = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        heading = path.stem
        paragraph = []
        start = 0
        for line_number, line in enumerate([*lines, ""], 1):
            header = re.fullmatch(r"#{1,6}\s+(.+)", line.strip())
            if line.strip() and header is None:
                if not paragraph:
                    start = line_number
                paragraph.append(line.strip())
                continue
            if paragraph:
                for index, excerpt in enumerate(textwrap.wrap(" ".join(paragraph), 520)):
                    location = f"{source}:{start}:{index}\n{heading}\n{excerpt}"
                    identifier = hashlib.sha256(location.encode()).hexdigest()[:24]
                    passages.append(Passage(identifier, source, heading, start, line_number - 1, excerpt))
                paragraph.clear()
            if header:
                heading = header[1]
    if not passages:
        raise ValueError(f"No Markdown passages found in {root}")
    return passages
