# Thin wrapper around run_all.sh (see that file for details).
.PHONY: install tables figures audit test paper all
install tables figures audit test paper all:
	./run_all.sh $@
