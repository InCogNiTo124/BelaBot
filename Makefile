.PHONY: test
.PHONY: type

check: test type

test:
	pytest -v tests/

type:
	mypy .

