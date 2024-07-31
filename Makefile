.PHONY: run-pdf-query run-pdf-query-with-reranker run-text-query run-text-query-with-reranker run-pdf-query-retrieval-compare run-text-query-retrieval-compare download-pdf

MAKEFILE_DIR = $(dir $(firstword $(MAKEFILE_LIST)))

run-pdf-query-retrieval-compare:
	python chain.py --pdf $(MAKEFILE_DIR)/data/rav4-hybrid-owners-manual-jan-24-current.pdf --query "Are there any limitation on the radar on this vehicle?"

run-text-query-retrieval-compare:
	python chain.py  --text $(MAKEFILE_DIR)/data/the-odyssey.txt --query "Summarise Telemachus's journey"

download-pdf:
	wget https://toyotamanuals.com.au/docs/rav4-hybrid-owners-manual-jan-24-current/ -O $(MAKEFILE_DIR)/data/rav4-hybrid-owners-manual-jan-24-current.pdf

download-text:
	wget https://www.gutenberg.org/cache/epub/1727/pg1727.txt -O $(MAKEFILE_DIR)/data/the-odyssey.txt

clean-empty-model-directory:
	# do this if you see "failed:Load model /tmp/ms-marco-MiniLM-L-12-v2/flashrank-MiniLM-L-12-v2_Q.onnx failed. File doesn't exist"
	rmdir /tmp/ms-marco-MiniLM-L-12-v2