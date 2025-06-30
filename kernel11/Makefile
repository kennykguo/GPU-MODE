NVCC = nvcc
PROGRAM_NAME = kernel
SRC = kernel_11.cu
TARGET = $(PROGRAM_NAME)

# CUDA flags with Ampere architecture (RTX 3060)
NVCC_FLAGS = -O3 -arch=sm_86

# Default target
all: $(TARGET)

# Compile the CUDA source file
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Generate PTX code
ptx: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -ptx -o $(PROGRAM_NAME).ptx $(SRC)

# Generate SASS code
sass: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -keep -o $(PROGRAM_NAME)_sass $(SRC)
	cuobjdump --dump-sass $(PROGRAM_NAME)_sass > $(PROGRAM_NAME).sass

# Clean target
clean:
	rm -f $(TARGET) $(PROGRAM_NAME).ptx $(PROGRAM_NAME).sass $(PROGRAM_NAME)_sass

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run ptx sass