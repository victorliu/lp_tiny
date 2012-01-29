LAPACK_LIBS = /d/dev/libs/output/LAPACK/lapack_mingw32.a \
              /d/dev/libs/output/BLAS/blas_mingw32.a \
              -lgfortran

#LAPACK_LIBS = /d/dev/libs/output/numerical_mingw32/libgoto2_barcelonap-r1.13.lib \
#              /d/dev/libs/output/numerical_mingw32/libgoto2_barcelona-r1.13.lib \
#              -lgfortran

DEBUG_LIBS = -L/d/dev/libs/output/duma -lduma

test:
	gcc -O0 -ggdb lp_tiny.c test_lp_tiny.c $(LAPACK_LIBS) $(DEBUG_LIBS)
eval:
	gcc -O2 lp_tiny.c eval_lp_tiny.c $(LAPACK_LIBS)
