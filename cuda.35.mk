.cu.o:
	$(NVCC) -gencode=arch=compute_35,code=sm_35 -o $@ -c $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -gencode=arch=compute_35,code=sm_35 --compiler-options=\" $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) \" -c $<

