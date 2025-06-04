"""Pipeline to convert LEED PDFs into a Neo4j knowledge graph.

This script demonstrates an end-to-end approach for extracting triples from
LEED documentation and loading them into Neo4j.
"""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List
import backoff
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

TRIPLE_SCHEMA = {
    "title": "LEEDTriple",
    "type": "object",
    "properties": {
        "sub": {"type": "string"},
        "sub_type": {"type": "string"},
        "pred": {"type": "string"},
        "obj": {"type": "string"},
        "obj_type": {"type": "string"},
    },
    "required": ["sub", "sub_type", "pred", "obj", "obj_type"],
}

SYSTEM_PROMPT = f"""
You are a LEED domain expert extracting knowledge triples.
Return ONLY a JSON array of objects matching this schema:

{json.dumps(TRIPLE_SCHEMA, indent=2)}

Entity type definitions:
- Category: one of ['Sustainable Sites','Water Efficiency','Energy & Atmosphere',
  'Materials & Resources','Indoor Environmental Quality','Location & Transportation',
  'Innovation','Regional Priority']
- Credit: a specific LEED credit title (e.g., 'Rainwater Management')
- Prerequisite: a LEED prerequisite title (e.g., 'Construction Activity Pollution Prevention')
- Metric: a quantitative requirement or threshold (e.g., '35% water use reduction')
Use canonical LEED names where possible.
"""


def pdf_to_text(path: Path) -> str:
    """Convert a PDF file to raw text."""
    doc = fitz.open(path)
    texts: List[str] = []
    for page in doc:
        page.clean_contents()
        text = page.get_text()
        text = re.sub(r"
?Page \d+.*", "", text)
        texts.append(text)
    return "
".join(texts)


def chunk_text(text: str, size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["

", "
", ".", " "],
    )
    return splitter.split_text(text)


client = OpenAI()


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def extract_triples(chunk: str) -> List[Dict[str, str]]:
    """Call the LLM to extract triples from a text chunk."""
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": chunk}],
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return ast.literal_eval(content)


model = SentenceTransformer("all-MiniLM-L6-v2")
ENT_THRESHOLD = 0.75
PRED_THRESHOLD = 0.6


def _cluster_texts(texts: Iterable[str], threshold: float) -> Dict[str, str]:
    """Cluster strings using sentence embeddings and return label mapping."""
    uniq = list(dict.fromkeys(texts))
    if not uniq:
        return {}
    vectors = model.encode(uniq)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        affinity="cosine",
        linkage="average",
    ).fit(vectors)
    return {text: f"ENT_{label}" for text, label in zip(uniq, clustering.labels_)}


def dedup_triples(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Assign cluster IDs to entities and predicates."""
    entities = [t["sub"] for t in triples] + [t["obj"] for t in triples]
    ent_labels = _cluster_texts(entities, ENT_THRESHOLD)
    preds = [t["pred"] for t in triples]
    pred_labels = _cluster_texts(preds, PRED_THRESHOLD)

    for t in triples:
        t["sub_id"] = ent_labels.get(t["sub"], t["sub"])
        t["obj_id"] = ent_labels.get(t["obj"], t["obj"])
        t["pred_id"] = pred_labels.get(t["pred"], t["pred"].upper())
    return triples


driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pwd"))
MERGE_QUERY = """
MERGE (s:{sub_type} {uid:$sub_id, name:$sub})
MERGE (o:{obj_type} {uid:$obj_id, name:$obj})
MERGE (s)-[r:{rel}]->(o)
"""


def load_to_neo4j(triples: List[Dict[str, str]]) -> None:
    """Write triples to Neo4j using MERGE semantics."""
    with driver.session() as session:
        for t in triples:
            session.run(
                MERGE_QUERY.format(
                    sub_type=t["sub_type"], obj_type=t["obj_type"], rel=t["pred"].upper()
                ),
                sub_id=t["sub_id"],
                sub=t["sub"],
                obj_id=t["obj_id"],
                obj=t["obj"],
            )


def process_pdf(path: Path) -> None:
    text = pdf_to_text(path)
    chunks = chunk_text(text)
    triples: List[Dict[str, str]] = []
    for chunk in chunks:
        try:
            triples.extend(extract_triples(chunk))
        except Exception as exc:  # pragma: no cover - depends on API
            print(f"Extraction failed: {exc}")
    triples = dedup_triples(triples)
    load_to_neo4j(triples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract LEED triples into Neo4j")
    parser.add_argument("pdf", type=Path, help="Path to LEED PDF file")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="pwd")
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    process_pdf(args.pdf)
