src = $(shell find -name '*.md')
out = $(src:%.md=html/%.html)

PANDOC_FLAGS = --highlight-style=pygments --katex # kate/zenburn/tango

all: $(out)

html/%.html: %.md Makefile
	@ mkdir -p $(dir $@)
	pandoc -f markdown -t html $< -s $(PANDOC_FLAGS) -o $@
