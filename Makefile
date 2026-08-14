# Convenience wrappers so nothing depends on remembering `.venv/bin/python`.
# Every target runs the project's own interpreter, not whatever `python` happens
# to be on PATH -- which on this machine is a conda base without torch.

PY := .venv/bin/python
PORT ?= 8000

.PHONY: help venv test watch demo train train-m2 train-m3 train-m4 train-m5 \
        evaluate divergence exchange clean-runs

help:
	@echo "make venv        create .venv and install dependencies"
	@echo "make watch       serve the viewer at http://localhost:$(PORT)"
	@echo "make demo        generate a replay with no training required"
	@echo "make test        run the test suite"
	@echo "make train       train the shared policy (Milestone 1)"
	@echo "make train-m2    fork it into per-agent brains (Milestone 2)"
	@echo "make train-m3    the competition milestone (Milestone 3)"
	@echo "make train-m4    construction, shaped + unshaped control (Milestone 4)"
	@echo "make train-m5    exchange, unpaid + shaped ablation (Milestone 5)"
	@echo "make evaluate    score the M1 checkpoint against both baselines"
	@echo "make divergence  measure per-agent behavioural divergence"
	@echo "make exchange    write the transfer ledger and its report"
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

train-m3:
	$(PY) -m sim.train --config config/m3_masked.yaml --run-name m3-masked \
		--policy-mode individual --init-from checkpoints/m2/latest.pt

train-m4:
	$(PY) -m sim.train --config config/m4.yaml --run-name m4 --updates 400 \
		--policy-mode individual --init-from checkpoints/m3-masked/latest.pt
	$(PY) -m sim.train --config config/m4_unshaped.yaml --run-name m4-unshaped \
		--updates 400 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt

train-m5:
	$(PY) -m sim.train --config config/m5.yaml --run-name m5 --updates 200 \
		--policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
	$(PY) -m sim.train --config config/m5_shaped.yaml --run-name m5-shaped \
		--updates 200 --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt

evaluate:
	$(PY) -m sim.evaluate --checkpoint checkpoints/m1/latest.pt --baselines

divergence:
	$(PY) -m sim.divergence --checkpoint checkpoints/m2/latest.pt

exchange:
	$(PY) -m sim.exchange --checkpoint checkpoints/m5/latest.pt

clean-runs:
	rm -rf runs checkpoints viewer/replays/*.json viewer/reports/*.json
