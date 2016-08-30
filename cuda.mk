# Minimum CUDA Compute Capability is 3.5.
# May need to add more sm_XX levels for later releases.
GPU_FLAGS = \
	--gpu-architecture\=compute_30 \
	--gpu-code=compute_30,sm_30,sm_35,sm_37,sm_50,sm_52,sm_60,sm_61

.cu.o:
	$(NVCC) $(GPU_FLAGS) -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) $(GPU_FLAGS) --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

