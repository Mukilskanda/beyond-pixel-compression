# Thin wrapper around run_all.sh (see that file for details).
.PHONY: install tables figures audit test all
install tables figures audit test all:
	./run_all.sh $@
