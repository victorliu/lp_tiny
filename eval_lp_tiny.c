#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lp_tiny.h"

// Note, this test code will produce errors (it will print "Error: ...")
// when the problems are severely underconstrained. I am not sure at the
// moment if this is because of the test generation code or if it is an
// actual bug in the solver. I am guessing the former, since the generator
// is pretty ad-hoc and takes random stabs at producing a reasonable problem.

// Uncomment this to produce lots of test files to compare against CVX
//#define DO_OUTPUT

int main(int argc, char *argv){
	int n, m, probno = 0;
	srand(time(0));
	for(n = 2; n <= 1024; n *= 2){
		for(m = 1; m < n; m *= 2){
			//int feas = rand()%2;
			int feas = 1;

			lp_tiny lp;
			lp_tiny_generate(&lp, n, m, feas, NULL);
			
#ifdef DO_OUTPUT
			char filename[32];
			sprintf(filename, "eval%02d.m", probno); probno++;
			FILE *fp_instr = fopen(filename, "wt");
			
			lp_tiny_print(&lp, fp_instr);
#endif
			int info;
			double *lambda = (double*)malloc(n*sizeof(double));
			double *nu = (double*)malloc(m*sizeof(double));
			double *x_opt = (double*)malloc(n*sizeof(double));
			lp_tiny_status status;
			int steps = 0;
			
			clock_t tstart = clock();
			info = lp_tiny_solve(&lp, x_opt, lambda, nu, &status, &steps);
			clock_t tend = clock();
			
			printf("n = %d, m = %d, Time: %g\n", n, m, (double)(tend-tstart) / (double)CLOCKS_PER_SEC);
			fflush(stdout);
			
			if(0 == info){
#ifdef DO_OUTPUT
				if(LP_TINY_SOLVED == status){
					fprintf(fp_instr, "strcmp(cvx_status,'Solved')\n");
					fprintf(fp_instr, "x_opt = ");
						lp_tiny_print_vector(n, x_opt, fp_instr);
						fprintf(fp_instr, ";\n");
					fprintf(fp_instr, "norm(x_opt-x) < n*1e-6\n");
				}else{
					fprintf(fp_instr, "strcmp(cvx_status,'Infeasible')\n");
				}
#endif
			}else{
				printf("Error: info = %d\n", info);
			}
#ifdef DO_OUTPUT
			fclose(fp_instr);
#endif
			free(lambda);
			free(x_opt);
			free(nu);
			
			lp_tiny_destroy(&lp);
		}
	}
	return 0;
}
