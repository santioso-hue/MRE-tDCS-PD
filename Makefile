# Makefile — pipeline sync checker for MRE_tDCS_PD
#
# The canonical script versions live in pipeline/. Per-subject scripts in
# FullPD5_segmentation/scripts/ should be kept in sync whenever the pipeline
# evolves. Run `make sync-check` to see which files have diverged.
#
# Usage:
#   make sync-check        — diff pipeline/ vs FullPD5_segmentation/scripts/
#   make sync-push         — overwrite subject scripts with pipeline versions
#   make sync-pull         — overwrite pipeline versions with subject scripts
#   make help              — show this message

PIPELINE_DIR := pipeline
SUBJECT_DIR  := FullPD5_segmentation/scripts

# Scripts that are shared between pipeline/ and subject-specific scripts/.
# 03_run_mddmri_only.py is included; 00_charm.sh and 04_* are pipeline-only.
SHARED_SCRIPTS := \
	00_dwi2cond.sh \
	01_register_dMRI_to_T1.sh \
	01b_save_v1_nifti.py \
	01c_save_dps_niftis.py \
	02_build_conductivity_tensor.py \
	03_run_simulations.py \
	03_run_mddmri_only.py

.PHONY: sync-check sync-push sync-pull help

help:
	@echo ""
	@echo "  make sync-check   — show diffs between pipeline/ and FullPD5_segmentation/scripts/"
	@echo "  make sync-push    — copy pipeline/ → FullPD5_segmentation/scripts/ (overwrite)"
	@echo "  make sync-pull    — copy FullPD5_segmentation/scripts/ → pipeline/ (overwrite)"
	@echo ""

sync-check:
	@echo "======================================================"
	@echo "Sync check: pipeline/ vs $(SUBJECT_DIR)/"
	@echo "======================================================"
	@DIFFS=0; \
	for f in $(SHARED_SCRIPTS); do \
		P="$(PIPELINE_DIR)/$$f"; \
		S="$(SUBJECT_DIR)/$$f"; \
		if [ ! -f "$$P" ]; then \
			echo "  MISSING in pipeline/: $$f"; \
			DIFFS=$$((DIFFS+1)); \
		elif [ ! -f "$$S" ]; then \
			echo "  MISSING in $(SUBJECT_DIR)/: $$f"; \
			DIFFS=$$((DIFFS+1)); \
		elif ! diff -q "$$P" "$$S" > /dev/null 2>&1; then \
			echo "  DIFFERS: $$f"; \
			diff --unified=3 "$$P" "$$S" | head -40; \
			echo "  ---"; \
			DIFFS=$$((DIFFS+1)); \
		else \
			echo "  OK:      $$f"; \
		fi; \
	done; \
	echo "======================================================"; \
	if [ $$DIFFS -eq 0 ]; then \
		echo "All scripts in sync."; \
	else \
		echo "$$DIFFS file(s) differ. Run 'make sync-push' or 'make sync-pull'."; \
		exit 1; \
	fi

sync-push:
	@echo "Copying pipeline/ → $(SUBJECT_DIR)/ ..."
	@for f in $(SHARED_SCRIPTS); do \
		P="$(PIPELINE_DIR)/$$f"; \
		S="$(SUBJECT_DIR)/$$f"; \
		if [ -f "$$P" ]; then \
			cp "$$P" "$$S"; \
			echo "  copied: $$f"; \
		else \
			echo "  SKIP (not in pipeline/): $$f"; \
		fi; \
	done
	@echo "Done."

sync-pull:
	@echo "Copying $(SUBJECT_DIR)/ → pipeline/ ..."
	@for f in $(SHARED_SCRIPTS); do \
		P="$(PIPELINE_DIR)/$$f"; \
		S="$(SUBJECT_DIR)/$$f"; \
		if [ -f "$$S" ]; then \
			cp "$$S" "$$P"; \
			echo "  copied: $$f"; \
		else \
			echo "  SKIP (not in $(SUBJECT_DIR)/): $$f"; \
		fi; \
	done
	@echo "Done."
