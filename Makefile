# Convenience wrappers so nothing depends on remembering `.venv/bin/python`.
# Every target runs the project's own interpreter, not whatever `python` happens
# to be on PATH -- which on this machine is a conda base without torch.

PY := .venv/bin/python
PORT ?= 8000

.PHONY: help venv test watch demo train train-m2 evaluate divergence clean-runs

help:
	@echo "make venv        create .venv and install dependencies"
	@echo "make watch       serve the viewer at http://localhost:$(PORT)"
	@echo "make demo        generate a replay with no training required"
	@echo "make test        run the test suite"
	@echo "make train       train the shared policy (Milestone 1)"
	@echo "make train-m2    fork it into per-agent brains (Milestone 2)"
	@echo "make evaluate    score the M1 checkpoint against both baselines"
	@echo "make divergence  measure per-agent behavioural divergence"
	@echo ""
	@echo "Override the port with: make watch PORT=8123"

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q torch numpy pyyaml pytest
	@echo "ready — try 'make demo' then 'make watch'"

test:
	$(PY) -m pytest

watch:
	@echo "serving the viewer on http://localhost:$(PORT)  (ctrl-c to stop)"
	$(PY) -m http.server $(PORT) --directory viewer

demo:
	$(PY) -m sim.make_fake_replay

train:
	$(PY) -m sim.train --run-name m1

train-m2:
	$(PY) -m sim.train --run-name m2 --policy-mode individual \
		--init-from checkpoints/m1/latest.pt

evaluate:
	$(PY) -m sim.evaluate --checkpoint checkpoints/m1/latest.pt --baselines

divergence:
	$(PY) -m sim.divergence --checkpoint checkpoints/m2/latest.pt

clean-runs:
	rm -rf runs checkpoints viewer/replays/*.json viewer/reports/*.json
