/*
Copyright (c) 2012, Victor Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "lp_tiny.h"
#include "lapack_decl.h"
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>

void lp_tiny_print_vector(int n, double *v, FILE *fp){
	fprintf(fp, "[");
	register int i;
	for(i = 0; i < n; ++i){
		fprintf(fp, "%f  ", v[i]);
	}
	fprintf(fp, "]'");
}

void lp_tiny_print_matrix(int rows, int cols, double *m, int ldm, FILE *fp){
	fprintf(fp, "[");
	register int i, j;
	for(i = 0; i < rows; ++i){
		for(j = 0; j < cols; ++j){
			fprintf(fp, "%f  ", m[i+j*ldm]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "]");
}


// Here we allocate one extra column in A so that the solver does not
// need to reallocate an entire matrix just to perform an augmentation.
int lp_tiny_init(lp_tiny *lp, int n, int m){
	if(NULL == lp){ return -1; }
	if(0 == n){ return -2; }
	if(0 == m){ return -3; }
	if(m >= n){ return -2; }
	
	lp->n = n;
	lp->m = m;
	int alloc_size = (n+1)*m + m + n; // note the n+1, see above
	lp->A = (double*)malloc(alloc_size*sizeof(double));
	if(NULL == lp->A){ return 1; }
	lp->b = lp->A + (n+1)*m; // size m
	lp->c = lp->b + m;       // size n
	
	memset(lp->A, 0, alloc_size*sizeof(double));
	return 0;
}

void lp_tiny_destroy(lp_tiny *lp){
	if(NULL != lp){
		free(lp->A);
	}
}

double lp_tiny_eval(const lp_tiny *lp, double *x){
	const int ione = 1;
	int n = lp->n;
	return FCALL(ddot)(&n, lp->c, &ione, x, &ione);
}

int lp_tiny_in_domain(const lp_tiny *lp, double *x){
	register int i;
	for(i = 0; i < lp->n; ++i){
		if(x[i] <= 0){ return 0; }
	}
	return 1;
}


////
// Notes:
//   Gradient of objective: c - 1./x
//   Hessian of objective:  1./x.^2   (diagonal)
int lp_tiny_centering_newton(const lp_tiny *lp,
                        const double *x0,
                        const double *barrier_t,
                        int iter_limit,
                        double *x_opt, double *nu_opt,
                        int *n_steps)
{
	// Parameters
	const double stopping_newton_decrement = 1e-6;
	const double beta = 0.2; // in (0, 1)
	const double alpha = 0.1; // in (0, 0.5)
	const int max_iterations = iter_limit; // to prevent out of control loops
	
	if(NULL == lp       ){ return -1; }
	if(NULL == x0       ){ return -2; }
	if(NULL == barrier_t){ return -3; }
	if(NULL == x_opt    ){ return -4; }
	if(NULL == nu_opt   ){ return -5; }
	
	const int m = lp->m, n = lp->n;
	const double kappa = 1.0/(*barrier_t);
	int info, ret = 0;
	register int i, j;
	double f;
	int iter;
	
	const int ione = 1;
	const double done = 1.0;
	const double dzero = 0.;
	const double dnone = -1.;
	const double dnhalf = -0.5;
	
	double lam2; // lambda^2 (Newton decrement)
	double *work; // the overall work array that contains all of the following
	double *g;    // gradient
	double *Hi;   // Hessian inverse (diagonal)
	double *AH;   // AH = A*H^{-1}
	double *S;    // S = -A*H^{-1}*A' = -A*AH'
	double *dx;   // Newton step
	
	// Set up our temporaries
	work = (double*)malloc((
		 m*(m+n) // for AH,S
		+3*n     // for g,Hi,dx
		) * sizeof(double));
	if(NULL == work){
		ret = 1;
		goto error_work;
	}
	int *iwork; // for pivots
	iwork = (int*)malloc(n*sizeof(int));
	if(NULL == iwork){
		ret = 1;
		goto error_iwork;
	}
	AH = work;    // size m x n
	S = AH + n*m; // size m x m
	g = S + m*m;  // size n
	Hi = g + n;   // size n
	dx = Hi + n;  // size m
	
	FCALL(dcopy)(&n, x0, &ione, x_opt, &ione); // x_opt = x0;
	
	f = lp_tiny_eval(lp, x_opt); // function value at previous point
	for(i = 0; i < n; ++i){
		f -= kappa*log(x_opt[i]);
	} // f = c'*x0 - kappa*sum(log(x0))
	for(iter = 0; iter < max_iterations; ++iter){
		//// compute gradient and Hessian
		for(i = 0; i < n; ++i){
			g[i] = lp->c[i] - kappa/x_opt[i];
			Hi[i] = (x_opt[i]*x_opt[i])/kappa; // diagonal of Hessian inverse
		}

		// Compute Newton step by block elimination of KKT system:
		// [ H  A' ] [ dx ] == [ -g ]   (g is gradient, H is Hessian)
		// [ A  0  ] [ nu ]    [  0 ]
		// H*dx + A'*nu == -g  -->  dx == -H\(g + A'*nu)
		// A*dx == 0
		// Together giving A*H^{-1}*g == -A'*H^{-1}*A*nu == S*nu
		// Therefore nu can be found first from the above symmetric system
		// Then dx can be back solved.
		for(j = 0; j < n; ++j){
			FCALL(dcopy)(&m, lp->A+m*j, &ione, AH+m*j, &ione);
			FCALL(dscal)(&m, &Hi[j], AH+m*j, &ione);
		} // AH = A*H^{-1}
		//FCALL(dgemm)("N","T", &m, &m, &n, &done, lp->A, &m,
		//              AH, &m, &dzero, S, &m); // S = -A*AH'
		FCALL(dsyr2k)("L", "N", &m, &n, &dnhalf, lp->A, &m,
		              AH, &m, &dzero, S, &m); // S = -A*AH'
		FCALL(dgemv)("N", &m, &n, &done, AH, &m, g, &ione,
		             &dzero, nu_opt, &ione); // AHg = AH*g
		
		// Solves S*nu = AHg
		FCALL(dsysv)("L", &m, &ione, S, &m, iwork, nu_opt, &m,
		             AH /*use as scratch*/, &n,
					 &info);
		//FCALL(dgesv)(&m, &ione, S, &m, iwork, nu_opt, &m, &info);
		if(0 != info){ ret = 3; goto error_solve; }
		// Solves H*v = -A'*nu - g
		FCALL(dgemv)("T", &m, &n, &dnone, lp->A, &m, nu_opt, &ione,
		             &dzero, dx, &ione); // dx = -A'*nu
		for(i = 0; i < n; ++i){
			dx[i] = Hi[i]*(dx[i] - g[i]);
		}

		lam2 = -FCALL(ddot)(&n, dx, &ione, g, &ione); // lam2 = dx'*g
		
		// At this point, g is preserved, Hi no longer needed;
		double *xp = Hi; // xp = x + t*dx (reusing storage)
		
		//// stopping criterion
		if(lam2 <= 2*stopping_newton_decrement){ break; }
		//// Line search
		double t = 1.0;
		int found = 0;
		while(0 == found){
			if(0 == t){ ret = 4; break; }
			FCALL(dcopy)(&n, x_opt, &ione, xp, &ione);
			FCALL(daxpy)(&n, &t, dx, &ione, xp, &ione); // xp = x + t*dx;
			if(0 == lp_tiny_in_domain(lp, xp)){ t *= beta; continue; }
			double fp = lp_tiny_eval(lp, xp);
			for(i = 0; i < n; ++i){
				fp -= kappa*log(xp[i]);
			} // fp = c'*xp - kappa*sum(log(xp))
			if(fp >= f - alpha*t*lam2){ t *= beta; continue; }
			else{
				found = 1;			
				//// Update
				FCALL(dcopy)(&n, xp, &ione, x_opt, &ione); // x = xp;
				f = fp; // update the objective value
				break;
			}
		}
	}
	if(NULL != n_steps){ *n_steps = iter; }

	if(iter == max_iterations){
		ret = 2;
	}
//	printf("x = "); lp_tiny_print_vector(n, x_opt); printf("\n");
//	printf("nu = "); lp_tiny_print_vector(m, nu_opt); printf("\n");
//	printf("f = %f\n", f);

error_solve:
	free(iwork);
error_iwork:
	free(work);
error_work:
	return ret;
}


int lp_tiny_solve_with_feasible_starting_point(const lp_tiny *lp,
                                          const double *x0,
                                          int iter_limit,
                                          double *x_opt,
                                          double *lambda_opt, double *nu_opt,
                                          int *n_steps)
{
	// Parameters
	const double mu = 16;
	const double tolerance = 1e-10;
	const double initial_objective_weight = 1.0;
	const int max_iterations = 1000; // to prevent out of control loops
	
	const int ione = 1;
	
	if(NULL == lp){ return -1; }
	if(NULL == x0){ return -2; }
	if(NULL == x_opt){ return -3; }
	if(NULL != n_steps){ *n_steps = 0; }
	
	int info, ret = 0;
	const int m = lp->m, n = lp->n;
	int i;
	double t = initial_objective_weight;
	
	double *work;
	double *xp, *nu = NULL;
	int lwork = n;
	if(NULL == nu_opt){ lwork += m; } // If nu_opt was NULL, use allocated nu
	else{ nu = nu_opt; }              // otherwise use the provided buffer
	work = (double*)malloc(lwork * sizeof(double));
	if(NULL == work){ return 1; }
	xp = work; // size n
	if(NULL == nu_opt){ nu = xp + n; }
	
	FCALL(dcopy)(&n, x0, &ione, xp, &ione);
	
	int iter;
	for(iter = 0; iter < max_iterations; ++iter){
		//// Centering step
		int ns = 0;
		info = lp_tiny_centering_newton(lp, xp, &t, iter_limit, x_opt, nu, &ns);
		if(0 != info){ ret = info; break; }
		if(NULL != n_steps){ *n_steps += ns; }
		
		//// Update
		FCALL(dcopy)(&n, x_opt, &ione, xp, &ione);
		
		//// Stopping criterion
		if((double)n/t < tolerance){ break; }
		t *= mu;
	}
	if(0 == ret && iter == max_iterations){
		ret = 2;
	}
	
	// Compute the dual variables if requested
	t = 1.0/t;
	if(NULL != lambda_opt){
		for(i = 0; i < n; ++i){
			lambda_opt[i] = -t/x_opt[i];
		}
	}
	if(NULL != nu_opt){
		for(i = 0; i < m; ++i){
			nu_opt[i] = t*nu[i];
		}
	}
	
	free(work);
	return ret;
}

int lp_tiny_solve(const lp_tiny *lp,
             double *x,
             double *lambda, double *nu,
             lp_tiny_status* status,
             int *n_steps)
{
	if(NULL == lp){ return -1; }
	if(NULL == x){ return -2; }
	if(NULL == status){ return -5; }
	if(NULL != n_steps){ *n_steps = 0; }
	
	int info, ns, ret = 0;
	const int m = lp->m, n = lp->n;
	const int mn = m*n;
	register int i, j;
	
	const int ione = 1;
	const double done = 1.;
	const double dnone = -1.;
	
	double *work;
	double *b_new, *c_new, *Acopy, *zt, *zt_opt;
	int lwork = m + (n+1) + mn + 2*(n+1);
	work = (double*)malloc(lwork * sizeof(double));
	if(NULL == work){ return 1; }
	b_new = work;          // size m
	c_new = b_new + m;     // size n+1
	Acopy = c_new + (n+1); // size mn
	zt = Acopy + mn;       // size n+1
	zt_opt = zt + (n+1);   // size n+1
	
	// Phase I:
	//   Solve:
	//     minimize   t
	//     subject to A*x == b
	//                x >= (1-t)*ones(n,1), t >= 0
	//   If t < 1, x is strictly feasible
	//   Change variables:
	//     z = x + (t-1)*ones
	//     A*x == b  becomes  A*z == b+(t-1)*A*ones
	//   Modified problem:
	//     minimize   t
	//     subject to [A, -A*ones] * [z; t] == b - A*ones(n,1)
	//                [z; t] >= 0
	//   Any x0 for which A*x0 == b is feasible.
	//   Then if there is an x_i < 0, choose t0 = 2-min_i(x_i)
	//   Otherwise choose t0 = 1
	// We will rely on lp_init to allocate an augmented A matrix
	// so that we don't have to do it here.
	lp_tiny lp1;
	lp1.n = n+1;
	lp1.m = m;
	lp1.A = lp->A;
	lp1.b = b_new;
	lp1.c = c_new;
	// Set the new b vector
	FCALL(dcopy)(&n, lp->b, &ione, lp1.b, &ione);
	for(i = 0; i <= n; ++i){ // set all of c to be 1
		lp1.c[i] = 1.0;
	}
	FCALL(dgemv)("N", &m, &n, &dnone, lp->A, &m, lp1.c, &ione,
	             &done, lp1.b, &ione); // b <- -A*ones + b
	memset(lp1.c, 0, n*sizeof(double)); // set the first n elements to zero
	for(i = 0; i < m; ++i){ // set the last column of the augmented A
		lp1.A[i+mn] = 0;
		for(j = 0; j < n; ++j){
			lp1.A[i+mn] -= lp1.A[i+m*j];
		}
	}
	// Find an x0 (place it in x for now)
	double *dgels_work;
	int dgels_lwork = -1;
	FCALL(dgels)("N", &m, &n, &ione, Acopy, &m, zt, &n,
	             Acopy, &dgels_lwork, &info); // workspace query
	dgels_lwork = (int)Acopy[0];
	dgels_work = (double*)malloc(dgels_lwork*sizeof(double));
	if(NULL == dgels_work){ ret = 1; goto error_dgels_work; }

	FCALL(dcopy)(&mn, lp->A, &ione, Acopy, &ione);
	FCALL(dcopy)(&m, lp->b, &ione, zt, &ione);
	// Solves A*xt == b, least norm solution
	FCALL(dgels)("N", &m, &n, &ione, Acopy, &m, zt, &n,
	             dgels_work, &dgels_lwork,
				 &info);
	if(0 != info){ ret = 5; goto error_rank; }
	
	zt[n] = 1.0;
	for(i = 0; i < n; ++i){ // determine t
		if(zt[i] < 0){
			double temp = 2.0-zt[i];
			if(temp > zt[n]){ zt[n] = temp; }
		}
	}
	for(i = 0; i < n; ++i){ // transform x into z
		zt[i] += (zt[n]-1.0);
	}
	// Solve the feasibility test LP
	ns = 0;
	*status = LP_TINY_INFEASIBLE;
	info = lp_tiny_solve_with_feasible_starting_point(&lp1, zt, 300, zt_opt,
	                                             NULL, NULL, &ns);
	if(0 == info){
		if(NULL != n_steps){ *n_steps += ns; }
		if(zt_opt[n] < 1.0){ // strictly feasible
			// transform optimal z into x, x is the strictly feasible x0
			for(i = 0; i < n; ++i){
				zt_opt[i] -= (zt_opt[n]-1.0);
			}
			// Solve the actual problem
			ns = 0;
			info = lp_tiny_solve_with_feasible_starting_point(lp, zt_opt, 300, x,
			                                             lambda, nu, &ns);
			if(0 != info){
				ret = info;
			}else{
				if(NULL != n_steps){ *n_steps += ns; }
				*status = LP_TINY_SOLVED;
			}
		}
	}else{
		*status = LP_TINY_UNBOUNDED;
		ret = info;
	}
	
error_rank:
	free(dgels_work);
error_dgels_work:
	free(work);
	return ret;
}

int lp_tiny_check_KKT(const lp_tiny *lp, double *x, double *nu){
	const int m = lp->m, n = lp->n;
	double nrm, *temp;
	register int i;
	const double tol = 8*(double)n*DBL_EPSILON;
	const int ione = 1;
	const double done = 1.;
	const double dnone = -1.;
	
	temp = (double*)malloc(n*sizeof(double));
	if(NULL == temp){ return 1; }
	
	// The KKT condition is:
	//   Ax = b
	//   A'nu + c - 1./x = 0
	// First check the condition A*x == b:
	FCALL(dcopy)(&m, lp->b, &ione, temp, &ione); // temp = b
	// temp = b - A*x
	FCALL(dgemv)("N", &m, &n, &done, lp->A, &m, x, &ione, &dnone, temp, &ione);
	// temp should be small
	nrm = FCALL(dnrm2)(&m, temp, &ione);
	if(nrm > tol){
		return -1;
	}
	
	// Now check the second condition
	// The Hessian is 1./(x.*x), so H*x = 1./x
	for(i = 0; i < n; ++i){
		temp[i] = lp->c[i] - 1.0/x[i];
	}
	// temp = A'*nu + c - H*x
	FCALL(dgemv)("T", &m, &n, &done, lp->A, &m, nu, &ione, &done, temp, &ione);
	// temp should be small
	nrm = FCALL(dnrm2)(&n, temp, &ione);
	if(nrm > tol){
		return -1;
	}
	
	free(temp);
	return 0;
}

void lp_tiny_print(lp_tiny *lp, FILE *fp){
	if(NULL == lp){ return; }
	fprintf(fp, "n = %u; m = %u;\n", lp->n, lp->m);
	fprintf(fp, "A = ");
		lp_tiny_print_matrix(lp->m, lp->n, lp->A, lp->m, fp);
		fprintf(fp, ";\n");
	fprintf(fp, "b = ");
		lp_tiny_print_vector(lp->m, lp->b, fp);
		fprintf(fp, ";\n");
	fprintf(fp, "c = ");
		lp_tiny_print_vector(lp->n, lp->c, fp);
		fprintf(fp, ";\n");
	fprintf(fp, "cvx_begin\n");
	fprintf(fp, "  variable x(n);\n");
	fprintf(fp, "  dual variables lambda nu;\n");
	fprintf(fp, "  minimize (c'*x);\n");
	fprintf(fp, "  nu: A*x == b;\n");
	fprintf(fp, "  lambda: x >= 0;\n");
	fprintf(fp, "cvx_end\n");
}

const char *lp_tiny_status_string(lp_tiny_status status){
	static const char *strs[3] = {
		"Solved",
		"Infeasible",
		"Unbounded"
	};
	static const char *inv = "Invalid";
	int i = (int)status;
	if(0 <= i && i < 3){
		return strs[i];
	}else{
		return inv;
	}
}

static double drand(){
	return (double)rand() / (double)RAND_MAX;
}

void lp_tiny_generate(lp_tiny *lp, int n, int m, int feasible, double *x0){
	if(NULL == lp){ return; }
	int ret = lp_tiny_init(lp, n, m);
	if(0 != ret){ return; }
	
	int estimate_rank = 1;
	double *work;
	double *Acopy, *sigma;
	int lwork = 3*m+n;
	if(lwork < 5*m){ lwork = 5*m; } // lwork for SVD rank computation
	work = (double*)malloc((lwork + m*n + m) * sizeof(double));
	if(NULL == work){ estimate_rank = 0; }
	Acopy = work+lwork;  // size m*n
	sigma = Acopy + m*n; // size m
	int full_rank;
	const int ione = 1;
	const double done = 1;
	const double dzero = 0;
	register int i, j;
	
	for(j = 0; j < n; ++j){ // Generate a random c vector
		lp->c[j] = 2*drand()-1.0;
	}
	// Generate a full rank A (keep trying until it is full rank)
	do{
		for(j = 0; j < n; ++j){
			for(i = 0; i < m-1; ++i){
				lp->A[i+m*j] = 2*drand()-1.0;
			}
			// make last row strictly positive (for bounded sublevel sets)
			lp->A[i+m*j] = drand()+DBL_EPSILON;
		}
		full_rank = 1; // Assume it is full rank unless proven otherwise
		
		if(0 != estimate_rank){
			int info;
			
			int nm = n*m;
			FCALL(dcopy)(&nm, lp->A, &ione, Acopy, &ione); // dgesvd is destructive
			FCALL(dgesvd)("N", "N",
			              &m, &n, Acopy, &m,
			              sigma, NULL, &ione, NULL, &ione,
			              work, &lwork,
						  &info);
			if(0 == info){
				// This is the same criterion used by Matlab
				double tol = (double)n * DBL_EPSILON * sigma[0];
				if(sigma[m-1] < tol){
					full_rank = 0;
				}
			}
		}
	}while(0 == full_rank);
	
	if(NULL != x0){ // Wanted a random feasible initial point
		for(j = 0; j < n; ++j){
			x0[j] = drand()+DBL_EPSILON;
		}
		FCALL(dgemv)("N", &m, &n, &done, lp->A, &m, x0, &ione,
		             &dzero, lp->b, &ione); // Generate a valid b using x0
	}else{
		// use Acopy to hold a temporary x0
		double *x0temp = Acopy;
		if(0 != feasible){ // requested a feasible problem
			for(j = 0; j < n; ++j){
				x0temp[j] = drand()+DBL_EPSILON;
			}
		}else{ // requested an infeasible problem
			do{
				for(j = 0; j < n; ++j){
					x0temp[j] = 2*drand()-1.0;
				}
			}while(0 != lp_tiny_in_domain(lp, x0temp));
		}
		FCALL(dgemv)("N", &m, &n, &done, lp->A, &m, x0temp, &ione,
		             &dzero, lp->b, &ione); // Generate a b using x0
	}
	
	if(0 != estimate_rank){
		free(work);
	}
}
