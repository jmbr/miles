#############################################################################
# miles development make file.
#############################################################################

SOURCE_DIR := miles
TEST_DIR := tests
SCRIPTS_DIR := scripts

SOURCES := $(filter-out $(SOURCE_DIR)/__init__.py, $(wildcard $(SOURCE_DIR)/*.py))
TESTS := $(wildcard $(TEST_DIR)/*.py)

AUTO_GENERATED_SOURCES := $(SOURCE_DIR)/__init__.py $(SOURCE_DIR)/version.py conda-recipe/meta.yaml

all: $(AUTO_GENERATED_SOURCES)

$(SOURCE_DIR)/__init__.py: $(SOURCES)
	@(cat $(SCRIPTS_DIR)/init-header.txt; \
	 $(SCRIPTS_DIR)/get-dependencies $(SOURCE_DIR)) > $@

$(SOURCE_DIR)/version.py:
	@scripts/make-version > $@

conda-recipe/meta.yaml: conda-recipe/meta.yaml.in $(SOURCE_DIR)/version.py
	@VERSION=`grep v_short miles/version.py | cut -f 2 -d "'"`; \
	sed "s/VERSION/$$VERSION/g" conda-recipe/meta.yaml.in > $@

doc: $(SOURCES)
	rm -f doc/miles.rst doc/modules.rst
	sphinx-apidoc -o doc miles
	$(MAKE) -C doc html

	pyreverse --only-classnames --output=svg --project=miles miles
	mv packages_miles.svg classes_miles.svg doc

.PHONY: all $(SOURCE_DIR)/version.py doc
