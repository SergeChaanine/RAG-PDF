"""Prepare the USEK crawl once; the presentation loads the finished package directly."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from rag_pdf.config import ChunkConfig, Settings  # noqa: E402
from rag_pdf.embeddings import LocalEmbeddingModel  # noqa: E402
from rag_pdf.presentation import load_prepared_website, prepare_website  # noqa: E402
from rag_pdf.reranking import LocalReranker  # noqa: E402


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crawl-dir", type=Path, default=settings.data_dir / "usek" / "full")
    parser.add_argument(
        "--output", type=Path, default=settings.data_dir / "presentations" / "usek-full"
    )
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="Usable sources; 0 = full crawl.")
    parser.add_argument("--backend", choices=["Chroma", "Qdrant", "LanceDB"], default="Chroma")
    parser.add_argument("--no-reextract", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify a completed package without loading a model or embedding.",
    )
    options = parser.parse_args(argv)
    if options.check:
        prepared = load_prepared_website(options.output)
        print(
            json.dumps(
                {
                    "ready": True,
                    "path": str(prepared.path),
                    "sources": prepared.index.document_count,
                    "chunks": prepared.index.chunk_count,
                },
                indent=2,
            )
        )
        if hasattr(prepared.store, "close"):
            prepared.store.close()
        return 0
    print(f"Loading {options.embedding_model} on {options.device}...", flush=True)
    model = LocalEmbeddingModel(options.embedding_model, options.device, options.batch_size)
    print(f"Embedding device: {model.device_label}", flush=True)
    print("Preparing the compact passage reranker...", flush=True)
    reranker = LocalReranker(device=options.device)
    reranker.rank("How do I register?", [])
    prepared = prepare_website(
        options.crawl_dir,
        options.output,
        model,
        chunk_config=ChunkConfig(options.chunk_size, options.overlap),
        backend=options.backend,
        limit=options.limit,
        reextract_html=not options.no_reextract,
        data_dir=settings.data_dir,
        progress=lambda value: print(json.dumps(value), flush=True),
    )
    print(
        json.dumps(
            {
                "ready": True,
                "path": str(prepared.path),
                "sources": prepared.index.document_count,
                "chunks": prepared.index.chunk_count,
            },
            indent=2,
        ),
        flush=True,
    )
    if hasattr(prepared.store, "close"):
        prepared.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
