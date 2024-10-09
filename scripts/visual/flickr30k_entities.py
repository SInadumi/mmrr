# Copyright (c) 2024, Nobuhiro Ueda
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional, Union

from cl_mmref.utils.util import CamelCaseDataClassJsonMixin


@dataclass(frozen=True)
class BndBox(CamelCaseDataClassJsonMixin):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Obj(CamelCaseDataClassJsonMixin):
    name: str
    bndbox: Optional[BndBox]


@dataclass(frozen=True)
class Size(CamelCaseDataClassJsonMixin):
    width: int
    height: int
    depth: int


@dataclass(frozen=True)
class Annotation(CamelCaseDataClassJsonMixin):
    filename: str
    size: Size
    objects: list[Obj]

    @classmethod
    def from_xml(cls, xml: ET.Element) -> "Annotation":
        filename = xml.find("filename").text  # type: ignore
        assert filename is not None
        size = xml.find("size")
        assert size is not None
        width = int(size.find("width").text)  # type: ignore
        height = int(size.find("height").text)  # type: ignore
        depth = int(size.find("depth").text)  # type: ignore
        objects = []
        for obj in xml.findall("object"):
            name: str = obj.find("name").text  # type: ignore
            if (bndbox := obj.find("bndbox")) is not None:
                xmin = int(bndbox.find("xmin").text)  # type: ignore
                ymin = int(bndbox.find("ymin").text)  # type: ignore
                xmax = int(bndbox.find("xmax").text)  # type: ignore
                ymax = int(bndbox.find("ymax").text)  # type: ignore
                objects.append(
                    Obj(
                        name=name,
                        bndbox=BndBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    )
                )  # type: ignore
            else:
                assert obj.find("nobndbox") is not None
                objects.append(Obj(name=name, bndbox=None))
        return cls(
            filename=filename,
            size=Size(width=width, height=height, depth=depth),
            objects=objects,
        )


@dataclass(frozen=True)
class Phrase:
    tag_idx: int
    text: str
    span: tuple[int, int]
    phrase_id: int
    phrase_type: str

    def to_string(self) -> str:
        return f"[/EN#{self.phrase_id}/{self.phrase_type} {self.text}]"


@dataclass(frozen=True)
class Sentence:
    text: str
    phrases: list[Phrase]

    def to_string(self) -> str:
        # 3:[/EN#549/people ほかの男]が立って4:[/EN#551/other ロープ]を握っている間に、1:[/EN#547/people ７人のクライマー]が2:[/EN#548/bodyparts 岩壁]を登っている。
        output_string = ""
        cursor = 0
        for phrase in self.phrases:
            output_string += self.text[cursor : phrase.span[0]]
            output_string += f"{phrase.tag_idx}:" + phrase.to_string()
            cursor = phrase.span[1]
        output_string += self.text[cursor:]
        return output_string

    @classmethod
    def from_string(cls, sentence_string: str) -> "Sentence":
        # 3:[/EN#549/people ほかの男]が立って4:[/EN#551/other ロープ]を握っている間に、1:[/EN#547/people ７人のクライマー]が2:[/EN#548/bodyparts 岩壁]を登っている。
        tag_pat = re.compile(
            r"(?P<idx>[0-9]+):\[/EN#(?P<id>[0-9]+)(/(?P<type>[A-Za-z_\-()/]+))+ (?P<words>[^]]+)]"
        )
        chunks: list[Union[str, dict[str, Any]]] = []
        sidx = 0
        matches: list[re.Match] = list(re.finditer(tag_pat, sentence_string))
        for match in matches:
            # chunk 前を追加
            if sidx < match.start():
                text = sentence_string[sidx : match.start()]
                chunks.append(text)
            # match の中身を追加
            chunks.append(
                {
                    "tag_idx": match.group("idx"),
                    "phrase": match.group("words"),
                    "phrase_id": match.group("id"),
                    "phrase_type": match.group("type"),
                }
            )
            sidx = match.end()
        # chunk 後を追加
        if sidx < len(sentence_string):
            chunks.append(sentence_string[sidx:])
        sentence = ""
        phrases = []
        char_idx = 0
        for chunk in chunks:
            if isinstance(chunk, str):
                sentence += chunk
                char_idx += len(chunk)
            else:
                chunk["first_char_index"] = char_idx
                sentence += chunk["phrase"]
                char_idx += len(chunk["phrase"])
                phrases.append(
                    Phrase(
                        tag_idx=int(chunk["tag_idx"]),
                        text=chunk["phrase"],
                        span=(
                            chunk["first_char_index"],
                            chunk["first_char_index"] + len(chunk["phrase"]),
                        ),
                        phrase_id=int(chunk["phrase_id"]),
                        phrase_type=chunk["phrase_type"],
                    )
                )
        assert "EN" not in sentence
        return cls(sentence.strip(), phrases)


@dataclass(frozen=True)
class Document:
    flickr_image_id: int
    sentences: list[Sentence]

    def to_string(self) -> list[str]:
        output_sentences = []
        for sentence in self.sentences:
            output_sentences.append(sentence.to_string())
        return output_sentences

    @classmethod
    def from_string(cls, image_id: int, flickr_sentences: list[str]) -> "Document":
        _sentences = []
        for flickr_sentence in flickr_sentences:
            sentence = Sentence.from_string(flickr_sentence)
            if sentence.text == "":
                continue
            if len(sentence.phrases) == 0:
                continue
            _sentences.append(sentence)
        return cls(image_id, _sentences)
