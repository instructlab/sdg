SDG_IMPORT_REF:=d9e7bf2f59819fcd42d9648b0ebbb81b6d2bf893

#
# If you want to see the full commands, run:
#   NOISY_BUILD=y make
#
ifeq ($(NOISY_BUILD),)
    ECHO_PREFIX=@
    CMD_PREFIX=@
    PIPE_DEV_NULL=> /dev/null 2> /dev/null
else
    ECHO_PREFIX=@\#
    CMD_PREFIX=
    PIPE_DEV_NULL=
endif

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY:check
check: ## check git diff between this repo and the CLI generator directory
	@(git remote | grep -q "^instructlab_repo") || git remote add instructlab_repo https://github.com/instructlab/instructlab
	@git fetch instructlab_repo
	@echo "==="
	@echo "=== CHANGES SINCE LAST IMPORT FROM instructlab/instructlab repo:"
	@echo "==="
	@git diff $(SDG_IMPORT_REF)..origin/main -- src/instructlab/generator/ | cat
