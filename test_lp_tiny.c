#include <stdio.h>
#include "lp_tiny.h"

int main(int argc, char *argv){
	{ printf("Testing n=2 infeasible\n");
		// This problem geometrically is equivalent to finding a point in
		// a null region of the plane; the region is the positive quadrant,
		// but our one constraint says that it must lie below the line
		// x + y == -1, which is impossible. Our objective function is
		// just to minimize x.
		lp_tiny lp;
		lp_tiny_status status;
		double x[2];
		lp_tiny_init(&lp, 2, 1);
		
		lp.c[0] = 1;
		lp.A[0] = 1;
		lp.A[1] = 1;
		lp.b[0] = -1;
		
		lp_tiny_solve(&lp, x, NULL, NULL, &status, NULL);
		printf(" Status: %s\n", lp_tiny_status_string(status));
		
		lp_tiny_destroy(&lp);
	}
	{ printf("Testing n=2 unbounded sublevel set\n");
		// This problem geometrically is equivalent to finding a point 
		// along y == x in the first quadrant that minimizes x. Even
		// though the origin is a solution, the sublevel sets are
		// unbounded, so this should be an unbounded problem.
		lp_tiny lp;
		lp_tiny_status status;
		double x[2];
		lp_tiny_init(&lp, 2, 1);
		
		lp.c[0] = 1;
		lp.A[0] = 1;
		lp.A[1] = -1;
		lp.b[0] = 0;
		
		lp_tiny_solve(&lp, x, NULL, NULL, &status, NULL);
		printf(" Status: %s\n", lp_tiny_status_string(status));
		
		lp_tiny_destroy(&lp);
	}
	{ printf("Testing n=2 unbounded\n");
		// This problem geometrically is equivalent to finding a point 
		// along y == x in the first quadrant that maximizes x. Not
		// only are the sublevel sets unbounded, the problem is also
		// unbounded.
		lp_tiny lp;
		lp_tiny_status status;
		double x[2];
		lp_tiny_init(&lp, 2, 1);
		
		lp.c[0] = -1;
		lp.A[0] = 1;
		lp.A[1] = -1;
		lp.b[0] = 0;
		
		lp_tiny_solve(&lp, x, NULL, NULL, &status, NULL);
		printf(" Status: %s\n", lp_tiny_status_string(status));
		
		lp_tiny_destroy(&lp);
	}
	{ printf("Testing n=2 feasibility\n");
		// This problem geometrically is equivalent to finding a point on
		// the line x+y == 1. We are only interested in feasibility,
		// so the object is just zero.
		lp_tiny lp;
		lp_tiny_status status;
		double x[2];
		lp_tiny_init(&lp, 2, 1);
		
		lp.c[0] = 0;
		lp.A[0] = 1;
		lp.A[1] = 1;
		lp.b[0] = 1;
		
		lp_tiny_solve(&lp, x, NULL, NULL, &status, NULL);
		printf(" Status: %s\n", lp_tiny_status_string(status));
		printf(" Solution: %g, %g\n", x[0], x[1]);
		
		lp_tiny_destroy(&lp);
	}
	{ printf("Testing n=2 feasibility\n");
		// This problem geometrically is equivalent to finding a point on
		// the line x+y == 1 that minimizes x + 2y.
		lp_tiny lp;
		lp_tiny_status status;
		double x[2];
		lp_tiny_init(&lp, 2, 1);
		
		lp.c[0] = 1;
		lp.c[1] = 2;
		lp.A[0] = 1;
		lp.A[1] = 1;
		lp.b[0] = 1;
		
		lp_tiny_solve(&lp, x, NULL, NULL, &status, NULL);
		printf(" Status: %s\n", lp_tiny_status_string(status));
		printf(" Solution: %g, %g\n", x[0], x[1]);
		
		lp_tiny_destroy(&lp);
	}
	return 0;
}
