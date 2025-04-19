# Compiler and flags
CC = g++
CFLAGS = -std=c++11 -Wall -O2
#CFLAGS = -std=c++11 -Wall -O2 -g -pg

# Source, include, build and bin directories
SRCDIR := src
INCDIR := include
BUIDIR := build
BINDIR := ../BIN

# Source files
SRCS := $(wildcard $(SRCDIR)/*.cc)

# Object files
OBJS := $(patsubst $(SRCDIR)/%.cc, $(BUIDIR)/%.o, $(SRCS))

# Target executable
TARGET := $(BINDIR)/main

# Default target
all: $(TARGET)

# Compile source files into object files
$(BUIDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(BUIDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Link object files into the target executable
$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@
	
# Clean up object files and the target executable
clean:
#	rm -rf $(BUIDIR) $(TARGET) $(OBJS) $(BINDIR)/results/*
	rm -rf $(BUIDIR) $(TARGET) $(OBJS)
