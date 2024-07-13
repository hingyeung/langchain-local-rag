.PHONY: run

run-pdf-query:
	python start.py --pdf ~/Downloads/toyota_rav4_user_manual_Oct22-Oct23.pdf --query 'Describe how one would replace a tyre of this car?'

run-pdf-query-with-reranker:
	python start.py --pdf ~/Downloads/toyota_rav4_user_manual_Oct22-Oct23.pdf --query 'Describe how one would replace a tyre of this car?' --use-reranker

run-text-query:
	python start.py --text ~/Downloads/odyssey.txt --query "Who is Penelope?"

run-text-query-with-reranker:
	python start.py --text ~/Downloads/odyssey.txt --query "Who is Penelope?" --use-reranker