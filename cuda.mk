# This library may require a newer version of the CUDA Toolkit than what is
# installed. The latest version can be found at the below site:
#   https://developer.nvidia.com/cuda-toolkit
# May need to add more sm_XX levels for later releases.
# If unsure of which architecture your GPU uses, this chart may be helpful:
#   https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# SUPPORTED RELEASES
# Minimum CUDA Compute Capability is 3.5.
MIN = compute_35

# Kepler:   3.5
#           3.7
KEPLER = sm_35, sm_37

# Maxwell:  5.0
#           5.2
#           5.3
MAXWELL = sm_50, sm_52, sm_53

# Pascal:   6.0
#           6.1
#          *6.2
PASCAL = sm_60, sm_61 #, sm_62

# Volta:    *7.0
#           *7.1
# VOLTA = sm_70, sm_71

# (*) = Not yet implemented.

# TODO: Need a way to auto-detect the proper compute capability version so the
#       user never needs to modify this file.
################################################################################
# Your GPU architecture should be the last one listed below. If not, remove    #
# the architecture names following yours.                                      #
# EXAMPLE: For a Maxwell GPU, remove the PASCAL variable so it looks like      #
# this: --gpu-code=$(MIN), $(KEPLER), $(MAXWELL)                               #
################################################################################
GPU_FLAGS = --gpu-architecture\=$(MIN) \
            --gpu-code=$(MIN), $(KEPLER), $(MAXWELL), $(PASCAL)

.cu.o:
	$(NVCC) $(GPU_FLAGS) -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) $(GPU_FLAGS) --compiler-options=\" \
    $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
    $(CPPFLAGS) \" -c $<

