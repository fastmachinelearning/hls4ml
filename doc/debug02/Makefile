TARGET=dense

SRC=dense.c

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)
.PHONY: run

$(TARGET): $(SRC)
	gcc $(C_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)
.PHONY: clean
