all: unformatted-data breast-cancer-wisconsin wdbc wpbc
.PHONY: all

unformatted-data:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/unformatted-data

breast-cancer-wisconsin: breast-cancer-wisconsin.data breast-cancer-wisconsin.names
.PHONY: breast-cancer-wisconsin

breast-cancer-wisconsin.data:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

breast-cancer-wisconsin.names:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names

wdbc: wdbc.data wdbc.names
.PHONY: wdbc

wdbc.data:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

wdbc.names:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

wpbc: wpbc.data wpbc.names
.PHONY: wpbc

wpbc.data:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data

wpbc.names:
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.names

ultraclean:
	rm *data* *names*
.PHONY: ultraclean
