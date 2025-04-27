NVCC := nvcc
PROGRAM_NAME := kernel
SRC := kernel_1.cu
TARGET := $(PROGRAM_NAME)

# CUDA flags with Ampere architecture (RTX 3060)
NVCC_FLAGS := -O3 -arch=sm_86

# Optional: Enable fast math for better performance (may reduce precision)
# NVCC_FLAGS += --use_fast_math

# Default target
all: $(TARGET)

# Compile the CUDA source file
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run