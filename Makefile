.PHONY: test
.PHONY: type

check: test type

test:
	pytest tests/

type:
	mypy .

