.PHONY: test
.PHONY: type

check: test type

test:
	pytest -v tests/

type:
	mypy --pretty .

train_loss:
	cat ${FILE} | grep 'Train|' | cut -d '|' -f 10 | ./analyse.py

test_loss:
	cat ${FILE} | grep 'Test|' | cut -d '|' -f 10 | ./analyse.py

point_diff:
	cat ${FILE} | grep 'Test|' | cut -d '|' -f 6,7 | sed -E 's/\|/-/' | bc -l | ./analyse.py
