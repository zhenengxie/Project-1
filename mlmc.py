import numpy
import numpy.linalg

class WeakConvergenceFailure(Exception):
    pass

def mlmc(Lmin, Lmax, N0, eps, mlmc_fn, alpha_0, beta_0, gamma):
    """
    Multi-level Monte Carlo estimation.

    (P, Nl) = mlmc(...)

    Outputs:
      P:  value
      Nl: number of samples on each level
    Inputs:
      Lmin: minimum level of refinement  >= 2
      Lmax: maximum level of refinement  >= Lmin
      N0:   initial number of samples    >  0
      eps:  desired accuracy (rms error) >  0

      alpha: weak error is  O(2^{-alpha*l})
      beta:  variance is    O(2^{-beta*l})
      gamma: sample cost is O(2^{gamma*l})  > 0

      If alpha, beta are not positive then they will be estimated.

      mlmc_fn: the user low-level routine. Its interface is
        sums = mlmc_fn(l, N)
      with inputs
        l = level
        N = number of paths
      and a numpy array of outputs
        sums[0] = sum(Y)
        sums[1] = sum(Y**2)
      where Y are iid samples with expected value
        E[P_0]            on level 0
        E[P_l - P_{l-1}]  on level l > 0
    """

    # Check arguments

    if Lmin < 2:
        raise ValueError("Need Lmin >= 2")
    if Lmax < Lmin:
        raise ValueError("Need Lmax >= Lmin")
    if N0 <= 0 or eps <= 0 or gamma <= 0:
        raise ValueError("Need N0 > 0, eps > 0, gamma > 0")

    # Initialisation

    alpha = max(0, alpha_0)
    beta  = max(0, beta_0)

    theta = 0.25

    L = Lmin

    Nl   = numpy.zeros(L+1)
    suml = numpy.zeros((2, L+1))
    dNl  = N0*numpy.ones(L+1)

    while sum(dNl) > 0:

        # update sample sums

        for l in range(0, L+1):
            if dNl[l] > 0:
                sums       = mlmc_fn(l, int(dNl[l]))
                Nl[l]      = Nl[l] + dNl[l]
                suml[0, l] = suml[0, l] + sums[0]
                suml[1, l] = suml[1, l] + sums[1]

        # compute absolute average and variance

        ml = numpy.abs(       suml[0, :]/Nl)
        Vl = numpy.maximum(0, suml[1, :]/Nl - ml**2)

        # fix to cope with possible zero values for ml and Vl
        # (can happen in some applications when there are few samples)

        for l in range(3, L+2):
            ml[l-1] = max(ml[l-1], 0.5*ml[l-2]/2**alpha)
            Vl[l-1] = max(Vl[l-1], 0.5*Vl[l-2]/2**beta)

        # use linear regression to estimate alpha, beta if not given
        if alpha_0 <= 0:
            A = numpy.ones((L, 2)); A[:, 0] = range(1, L+1)
            x = numpy.linalg.solve(A, numpy.log2(ml[1:]))
            alpha = max(0.5, -x[0])

        if beta_0 <= 0:
            A = numpy.ones((L, 2)); A[:, 0] = range(1, L+1)
            x = numpy.linalg.solve(A, numpy.log2(Vl[1:]))
            beta = max(0.5, -x[0])

        # set optimal number of additional samples

        Cl = 2**(gamma*numpy.arange(0, L+1))
        Ns = numpy.ceil( numpy.sqrt(Vl/Cl) * sum(numpy.sqrt(Vl*Cl)) / ((1-theta)*eps**2) )
        dNl = numpy.maximum(0, Ns-Nl)

        # if (almost) converged, estimate remaining error and decide
        # whether a new level is required

        if sum(dNl > 0.01*Nl) == 0:
            rem = ml[L] / (2.0**alpha - 1.0)

            if rem > numpy.sqrt(theta)*eps:
                if L == Lmax:
                    raise WeakConvergenceFailure("Failed to achieve weak convergence")
                else:
                    L = L + 1
                    Vl = numpy.append(Vl, Vl[-1] / 2.0**beta)
                    Nl = numpy.append(Nl, 0.0)
                    suml = numpy.column_stack([suml, [0, 0]])

                    Cl = 2**(gamma*numpy.arange(0, L+1))
                    Ns = numpy.ceil( numpy.sqrt(Vl/Cl) * sum(numpy.sqrt(Vl*Cl)) / ((1-theta)*eps**2) )
                    dNl = numpy.maximum(0, Ns-Nl)

    # finally, evaluate the multilevel estimator
    P = sum(suml[0,:]/Nl)

    return (P, Nl)
