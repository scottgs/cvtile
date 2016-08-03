.cu.o:
	$(NVCC) -gencode=arch=compute_61,code=sm_61 -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -gencode=arch=compute_61,code=sm_61 --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

