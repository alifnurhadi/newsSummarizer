.PHONY: help install ollama ingest run clean

# Default variables for ingestion (You can override these in the terminal)
FILE ?= data/raw_laws/Mock_law.pdf
SOURCE ?= "Mock Law"
KEYWORD ?= "corporate_governance_and_infrastructure"

help:
	@echo "======================================================================"
	@echo "                 AI Wealth Management Agent - Commands                "
	@echo "======================================================================"
	@echo "  make install    - Install/Sync all project dependencies using uv"
	@echo "  make ollama     - Start the Ollama AI engine (llama3:latest)"
	@echo "  make ingest     - Ingest a document into the Vector DB."
	@echo "                    (Default: ingests Mock_law.pdf)"
	@echo "                    Override example:"
	@echo "                    make ingest FILE=data/raw_laws/mock_law.txt SOURCE=\"New Law\" KEYWORD=\"finance\""
	@echo "  make run        - Run the main RAG AI pipeline to generate a report"
	@echo "  make clean      - Delete the ChromaDB vector database to start fresh"
	@echo "======================================================================"

install:
	uv sync

ollama:
	ollama run llama3:latest

ingest:
	uv run src/ingestion.py --file "$(FILE)" --source "$(SOURCE)" --keyword "$(KEYWORD)"

run:
	uv run business_logic/rag_logic.py

clean:
	rm -rf local_vectordb/
	@echo "✅ Cleaned up local vector database (local_vectordb/)."
