# Makefile for rummage - GPU Nostr Key Search

SRCDIR = src

SRC = $(SRCDIR)/rummage.cpp \
      $(SRCDIR)/CPU/Point.cpp \
      $(SRCDIR)/CPU/Int.cpp \
      $(SRCDIR)/CPU/IntMod.cpp \
      $(SRCDIR)/CPU/SECP256K1.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
		GPU/GPURummage.o \
		GPU/CudaPowMiner.o \
		GPU/rummage_ffi.o \
		CPU/Point.o \
		CPU/Int.o \
		CPU/IntMod.o \
		CPU/SECP256K1.o \
        rummage.o \
)

CCAP      = 120
CUDA      = /usr/local/cuda-13.2
CXX       = g++
CXXCUDA   = /usr/bin/g++

# GPU Performance Configuration
# Adjust these based on your GPU (see docs/PERFORMANCE.md)
NOSTR_BLOCKS_PER_GRID   = 3072
NOSTR_THREADS_PER_BLOCK = 256
KEYS_PER_THREAD_BATCH   = 256

# Extra defines for CudaPowMiner.cu (used by benchmark.sh)
# Example: make POW_DEFINES="-DNONCES_PER_THREAD=128 -DNUM_STREAMS=4"
POW_DEFINES =

CXXFLAGS  = -DWITHGPU -march=native -Wno-write-strings -O2 -I$(SRCDIR) -I$(CUDA)/include \
            -DNOSTR_BLOCKS_PER_GRID=$(NOSTR_BLOCKS_PER_GRID) \
            -DNOSTR_THREADS_PER_BLOCK=$(NOSTR_THREADS_PER_BLOCK) \
            -DKEYS_PER_THREAD_BATCH=$(KEYS_PER_THREAD_BATCH)
LFLAGS    = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart -lcurand -lssl -lcrypto
NVCC      = $(CUDA)/bin/nvcc

#--------------------------------------------------------------------

all: rummage

$(OBJDIR)/GPU/GPURummage.o: $(SRCDIR)/GPU/GPURummage.cu
	$(NVCC) -allow-unsupported-compiler --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -O2 -I$(SRCDIR) -I$(CUDA)/include \
	-DNOSTR_BLOCKS_PER_GRID=$(NOSTR_BLOCKS_PER_GRID) \
	-DNOSTR_THREADS_PER_BLOCK=$(NOSTR_THREADS_PER_BLOCK) \
	-DKEYS_PER_THREAD_BATCH=$(KEYS_PER_THREAD_BATCH) \
	-gencode=arch=compute_$(CCAP),code=sm_$(CCAP) -o $(OBJDIR)/GPU/GPURummage.o -c $(SRCDIR)/GPU/GPURummage.cu

$(OBJDIR)/GPU/CudaPowMiner.o: $(SRCDIR)/GPU/CudaPowMiner.cu
	$(NVCC) -allow-unsupported-compiler --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -O2 -I$(SRCDIR) -I$(CUDA)/include \
	$(POW_DEFINES) \
	-gencode=arch=compute_$(CCAP),code=sm_$(CCAP) -o $(OBJDIR)/GPU/CudaPowMiner.o -c $(SRCDIR)/GPU/CudaPowMiner.cu

$(OBJDIR)/GPU/rummage_ffi.o: $(SRCDIR)/GPU/rummage_ffi.cu
	$(NVCC) -allow-unsupported-compiler --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -O2 -I$(SRCDIR) -I$(CUDA)/include \
	-DNOSTR_BLOCKS_PER_GRID=$(NOSTR_BLOCKS_PER_GRID) \
	-DNOSTR_THREADS_PER_BLOCK=$(NOSTR_THREADS_PER_BLOCK) \
	-DKEYS_PER_THREAD_BATCH=$(KEYS_PER_THREAD_BATCH) \
	-gencode=arch=compute_$(CCAP),code=sm_$(CCAP) -o $(OBJDIR)/GPU/rummage_ffi.o -c $(SRCDIR)/GPU/rummage_ffi.cu

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/CPU/%.o : $(SRCDIR)/CPU/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

rummage: $(OBJET)
	@echo Making rummage...
	$(CXX) $(OBJET) $(LFLAGS) -o rummage

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/CPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/CPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CPU

clean:
	@echo Cleaning...
	@rm -rf obj || true
	@rm -f rummage || true

.PHONY: all clean
