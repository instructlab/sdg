SDG_IMPORT_REF:=d9e7bf2f59819fcd42d9648b0ebbb81b6d2bf893

.PHONY:check
check:
	@(git remote | grep -q "^instructlab_repo") || git remote add instructlab_repo https://github.com/instructlab/instructlab
	@git fetch instructlab_repo
	@echo "==="
	@echo "=== CHANGES SINCE LAST IMPORT FROM instructlab/instructlab repo:"
	@echo "==="
	@git diff $(SDG_IMPORT_REF)..origin/main -- src/instructlab/generator/ | cat
