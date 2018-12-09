"""
PyIBP

Implements fast Gibbs sampling for the linear-Gaussian
infinite latent feature model (IBP).

Copyright (C) 2009 David Andrzejewski (andrzeje@cs.wisc.edu)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Useful matrices
X  -  N × D  -  Observed D−dimensional features for items 1 . . . N
A  -  K × D  -  Weight matrix mapping latent features to observations

Z  -  N × K  -  Binary latent K−dimensional features for items 1 . . . N
V  -  N × K  -  Real-valued latent features (Infinite FA)
"""

import numpy as np
import scipy
import pdb
from scipy.stats import poisson

# We will be taking log(0) = -Inf, so turn off this warning
np.seterr(divide='ignore')


class IBP(object):
    """
    Implements fast Gibbs sampling for the linear-Gaussian
    infinite latent feature model (IBP)
    """

    #
    # Initialization methods
    #

    def __init__(self, data, alpha, sigma_x, sigma_a,
                 missing=None, use_v=False, init_zv=None):
        """ 
        data = NxD NumPy data matrix (should be centered)

        alpha = Fixed IBP hyperparam for OR (init,a,b) tuple where
        (a,b) are Gamma hyperprior shape and rate/inverse scale
        sigma_x = Fixed noise std OR (init,a,b) tuple (same as alpha)
        sigma_a = Fixed weight std OR (init,a,b) tuple (same as alpha)
        
        OPTIONAL ARGS
        missing = boolean/binary 'missing data' mask (1=missing entry)
        use_v = Are we using real-valued latent features? (default binary)
        init_zv = Optional initial state for the latent
        """
        # Data matrix
        self.X = data
        (self.N, self.D) = data.shape
        # IBP hyperparameter
        if type(alpha) == tuple:
            (self.alpha, self.alpha_a, self.alpha_b) = alpha
        else:
            (self.alpha, self.alpha_a, self.alpha_b) = (alpha, None, None)
        # Noise variance hyperparameter
        if type(sigma_x) == tuple:
            (self.sigma_x, self.sigma_xa, self.sigma_xb) = sigma_x
        else:
            (self.sigma_x, self.sigma_xa, self.sigma_xb) = (sigma_x, None, None)
        # Weight variance hyperparameter
        if type(sigma_a) == tuple:
            (self.sigma_a, self.sigma_aa, self.sigma_ab) = sigma_a
        else:
            (self.sigma_a, self.sigma_aa, self.sigma_ab) = (sigma_a, None, None)
        # Are we using weighted latent features?
        self.use_v = use_v

        # Do we have user-supplied initial latent feature values?
        if init_zv is None:
            # Initialze Z from IBP(alpha)
            self.init_z()
            # Initialize V from N(0,1) if necessary
            if self.use_v:
                self.init_v()
        else:
            self.ZV = init_zv
            self.K = self.ZV.shape[1]
            self.feature_counts = (self.ZV != 0).astype(np.int).sum(axis=0)
        # Sample missing data entries if necessary
        self.missing = missing
        if missing is not None:
            self.sample_x()

    def init_v(self):
        """ Init latent feature weights V according to N(0,1) """
        for (i, k) in zip(*self.ZV.nonzero()):
            self.ZV[i, k] = np.random.normal(0, 1)

    def init_z(self):
        """ Init latent features Z according to IBP(alpha) """
        Z = np.ones((0, 0))
        for i in range(1, self.N + 1):
            # Sample existing features
            # Sample dishes according to their popularity (averaged over columns)
            zi = (np.random.uniform(0, 1, (1, Z.shape[1])) <
                  (Z.sum(axis=0).astype(np.float) / i))

            # Sample new features
            num_new = poisson.rvs(self.alpha / i)
            zi = np.hstack((zi, np.ones((1, num_new))))  # For the new row, these features are all 1's

            # Add to Z matrix
            Z = np.hstack(
                (Z, np.zeros((Z.shape[0], num_new))))  # For the existing rows, they have 0's for the new features
            Z = np.vstack((Z, zi))
        self.ZV = Z
        self.K = self.ZV.shape[1]

        # Calculate initial feature counts
        self.feature_counts = (self.ZV != 0).astype(np.int).sum(axis=0)

    #
    # Convenient external methods
    #

    def full_sample(self):
        """ Do all applicable samples """
        self.sample_z()
        if self.missing is not None:
            self.sample_x()
        if self.alpha_a is not None:
            self.sample_alpha()
        if self.sigma_xa is not None:
            self.sample_sigma()

    def log_likeli(self):
        """
        Calculate log-likelihood P(X,Z)
        (or P(X,Z,V) if applicable)
        """
        liketerm = self.logPX(self.calcM(self.ZV), self.ZV)
        ibpterm = self.log_indian_buffet_process()
        if self.use_v:
            vterm = self.log_prob_v()
            return liketerm + ibpterm + vterm
        else:
            return liketerm + ibpterm

    def weights(self):
        """ Return E[A|X,Z] """
        return self.postA(self.X, self.ZV)[0]

    #
    # Actual sampling methods
    #

    def sample_v(self, k, meanA, covarA, xi, zi):
        """
        Slice sampling for feature weight V
        Implements equation 53 in http://www.david-andrzejewski.com/publications/llnl-accelerated-gibbs.pdf
        """
        # oldv = zi[0, k]
        # Log-posterior of current value
        cur_log_posterior = self.v_log_posterior(k, zi[0, k], meanA, covarA, xi, zi)
        # Vertically sample beneath this value
        curval = self.log_uniform(cur_log_posterior)
        # Initial sample from horizontal slice
        (left, right) = self.make_interval(curval, k, zi[0, k], meanA, covarA, xi, zi)
        newv = np.random.uniform(left, right)
        newval = self.v_log_posterior(k, newv, meanA, covarA, xi, zi)
        # Repeat until valid sample obtained
        while newval <= curval:
            if newv < zi[0, k]:
                left = newv
            else:
                right = newv
            newv = np.random.uniform(left, right)
            newval = self.v_log_posterior(k, newv, meanA, covarA, xi, zi)
        return newv

    def make_interval(self, u, k, v, meanA, covarA, xi, zi):
        """ Get horizontal slice sampling interval """
        w = .25
        (left, right) = (v - w, v + w)
        (leftval, rightval) = (self.v_log_posterior(k, left, meanA, covarA, xi, zi),
                               self.v_log_posterior(k, right, meanA, covarA, xi, zi))
        while leftval > u:
            left -= w
            leftval = self.v_log_posterior(k, left, meanA, covarA, xi, zi)
        while rightval > u:
            right += w
            rightval = self.v_log_posterior(k, right, meanA, covarA, xi, zi)
        return left, right

    def v_log_posterior(self, k, v, meanA, covarA, xi, zi):
        """
        For a given V, calculate the log-posterior
        P(A|X, Z, \theta)

        """
        oldv = zi[0, k]
        zi[0, k] = v
        (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
        logprior = -0.5 * (v ** 2) - 0.5 * np.log(2 * np.pi)
        loglike = self.logPxi(meanLike, covarLike, xi)
        # Restore previous value and return result
        zi[0, k] = oldv
        return logprior + loglike

    def sample_sigma(self):
        """ Sample feature/noise variances """
        # Posterior over feature weights A
        (meanA, covarA) = self.postA(self.X, self.ZV)
        # sigma_x
        variables = np.dot(self.ZV, np.dot(covarA, self.ZV.T)).diagonal()
        var_x = (np.power(self.X - np.dot(self.ZV, meanA), 2)).sum()
        var_x += self.D * variables.sum()
        n = float(self.N * self.D)
        postShape = self.sigma_xa + n / 2
        postScale = float(1) / (self.sigma_xb + var_x / 2)
        tau_x = scipy.stats.gamma.rvs(postShape, scale=postScale)
        self.sigma_x = np.sqrt(float(1) / tau_x)
        # sigma_a
        var_a = covarA.trace() * self.D + np.power(meanA, 2).sum()
        n = float(self.K * self.D)
        postShape = self.sigma_aa + n / 2
        postScale = float(1) / (self.sigma_ab + var_a / 2)
        tau_a = scipy.stats.gamma.rvs(postShape, scale=postScale)
        self.sigma_a = np.sqrt(float(1) / tau_a)
        if self.sigma_a > 100:
            pdb.set_trace()

    def sample_alpha(self):
        """ Sample alpha from conjugate posterior """
        postShape = self.alpha_a + self.feature_counts.sum()
        postScale = float(1) / (self.alpha_b + self.N)
        self.alpha = scipy.stats.gamma.rvs(postShape, scale=postScale)

    def sample_x(self):
        """ Take single sample missing data entries in X """
        # Calculate posterior mean/covar --> info
        (meanA, covarA) = self.postA(self.X, self.ZV)
        (infoA, hA) = self.toInfo(meanA, covarA)
        # Find missing observations
        xis = np.nonzero(self.missing.max(axis=1))[0]
        for i in xis:
            # Get (z,x) for this data point
            (zi, xi) = (np.reshape(self.ZV[i, :], (1, self.K)),
                        np.reshape(self.X[i, :], (1, self.D)))
            # Remove this observation
            infoA_i = self.updateInfo(infoA, zi, -1)
            hA_i = self.updateH(hA, zi, xi, -1)
            # Convert back to mean/covar
            (meanA_i, covarA_i) = self.fromInfo(infoA_i, hA_i)
            # Resample xi
            (meanXi, covarXi) = self.likelihood_xi(zi, meanA_i, covarA_i)
            newxi = np.random.normal(meanXi, np.sqrt(covarXi))
            # Replace missing features
            ks = np.nonzero(self.missing[i, :])[0]
            self.X[i, ks] = newxi[0][ks]

    def sample_z(self):
        """ Take single sample of latent features Z """
        # for each data point
        order = np.random.permutation(self.N)
        for (counter, i) in enumerate(order):
            # Initially, and later occasionally,
            # re-cacluate information directly
            if counter % 5 == 0:
                try:
                    meanA, covarA = self.postA(self.X, self.ZV)
                    infoA, hA = self.toInfo(meanA, covarA)
                except Exception as e:
                    pdb.set_trace()
                    # Get (z,x) for this data point
            (zi, xi) = (np.reshape(self.ZV[i, :], (1, self.K)),
                        np.reshape(self.X[i, :], (1, self.D)))

            # Remove this point from information
            infoA = self.updateInfo(infoA, zi, -1)
            hA = self.updateH(hA, zi, xi, -1)

            # Convert back to mean/covar
            (meanA, covarA) = self.fromInfo(infoA, hA)

            # Remove this data point from feature cts
            new_feature_counts = self.feature_counts - (self.ZV[i, :] != 0).astype(np.int)

            # Log collapsed Beta-Bernoulli terms
            log_prob_z1 = np.log(new_feature_counts)
            log_prob_z0 = np.log(self.N - new_feature_counts)

            # Find all singleton features
            singletons = [ki for ki in range(self.K) if
                          self.ZV[i, ki] != 0 and self.feature_counts[ki] == 1]
            nonsingletons = [ki for ki in range(self.K) if
                             ki not in singletons]

            # Sample for each non-singleton feature
            #
            for k in nonsingletons:
                oldz = zi[0, k]

                # z=0 case
                log_prob_0 = log_prob_z0[k]
                zi[0, k] = 0
                (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
                log_prob_0 += self.logPxi(meanLike, covarLike, xi)

                # z=1 case
                log_prob_1 = log_prob_z1[k]
                if self.use_v:
                    if oldz != 0:
                        # Use current V value
                        zi[0, k] = oldz
                        (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
                        log_prob_1 += self.logPxi(meanLike, covarLike, xi)
                    else:
                        # Sample V values from the prior to 
                        # numerically collapse/integrate
                        num_variates = 5
                        lps = np.zeros((num_variates,))
                        for variate in range(num_variates):
                            zi[0, k] = np.random.normal(0, 1)
                            (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
                            lps[variate] = self.logPxi(meanLike, covarLike, xi)
                        log_prob_1 += lps.mean()
                else:
                    zi[0, k] = 1
                    (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
                    log_prob_1 += self.logPxi(meanLike, covarLike, xi)

                # Sample Z, update feature counts
                if not self.logBern(log_prob_0, log_prob_1):
                    zi[0, k] = 0
                    if oldz != 0:
                        self.feature_counts[k] -= 1
                else:
                    if oldz == 0:
                        self.feature_counts[k] += 1
                    if self.use_v:
                        # Slice sample V from posterior if necessary
                        zi[0, k] = self.sample_v(k, meanA, covarA, xi, zi)
                if self.feature_counts[k] != ((self.ZV[:, k] != 0).astype(np.int)).sum():
                    pdb.set_trace()
                if self.feature_counts[k] > self.N:
                    pdb.set_trace()
            #
            # Sample singleton/new features using the
            # Metropolis-Hastings step described in Meeds et al
            #
            kold = len(singletons)

            # Sample from the Metropolis proposal
            knew = poisson.rvs(self.alpha / self.N)
            if self.use_v:
                vnew = np.array([np.random.normal(0, 1) for _ in range(knew)])

            # Net difference in number of singleton features
            netdiff = knew - kold

            # Contribution of singleton features to variance in x
            if self.use_v:
                prevcontrib = np.power(zi[0, singletons], 2).sum()
                newcontrib = np.power(vnew, 2).sum()
                weightdiff = newcontrib - prevcontrib
            else:
                weightdiff = knew - kold

            # Calculate the loglikelihoods
            (meanLike, covarLike) = self.likelihood_xi(zi, meanA, covarA)
            lpold = self.logPxi(meanLike, covarLike, xi)
            lpnew = self.logPxi(meanLike,
                                covarLike + weightdiff * self.sigma_a ** 2,
                                xi)
            lpaccept = min(0.0, lpnew - lpold)
            lpreject = np.log(max(1.0 - np.exp(lpaccept), 1e-100))

            if self.logBern(lpreject, lpaccept):
                # Accept the Metropolis-Hastings proposal
                if netdiff > 0:
                    # We're adding features, update ZV
                    self.ZV = np.append(self.ZV, np.zeros((self.N, netdiff)), 1)
                    if self.use_v:
                        prevNumSingletons = len(singletons)
                        self.ZV[i, singletons] = vnew[:prevNumSingletons]
                        self.ZV[i, self.K:] = vnew[prevNumSingletons:]
                    else:
                        self.ZV[i, self.K:] = 1
                    # Update feature counts m
                    self.feature_counts = np.append(self.feature_counts, np.ones(netdiff), 0)
                    # Append information matrix with 1/sigmaa^2 diag
                    infoA = np.vstack((infoA, np.zeros((netdiff, self.K))))
                    infoA = np.hstack((infoA,
                                       np.zeros((netdiff + self.K, netdiff))))
                    infoappend = (1 / self.sigma_a ** 2) * np.eye(netdiff)
                    infoA[self.K:(self.K + netdiff), self.K:(self.K + netdiff)] = infoappend
                    # only need to resize (expand) hA
                    hA = np.vstack((hA, np.zeros((netdiff, self.D))))
                    # Note that the other effects of new latent features 
                    # on (infoA,hA) (ie, the zi terms) will be counted when 
                    # this zi is added back in                    
                    self.K += netdiff
                elif netdiff < 0:
                    # We're removing features, update ZV
                    if self.use_v:
                        self.ZV[i, singletons[(-1 * netdiff):]] = vnew
                    dead = [ki for ki in singletons[:(-1 * netdiff)]]
                    self.K -= len(dead)
                    self.ZV = np.delete(self.ZV, dead, axis=1)
                    self.feature_counts = np.delete(self.feature_counts, dead)
                    # Easy to do this b/c these features did not
                    # occur in any other data points anyways...
                    infoA = np.delete(infoA, dead, axis=0)
                    infoA = np.delete(infoA, dead, axis=1)
                    hA = np.delete(hA, dead, axis=0)
                else:
                    # net difference is actually zero, just replace
                    # the latent weights of existing singletons
                    # (if applicable)
                    if self.use_v:
                        self.ZV[i, singletons] = vnew
            # Add this point back into information
            #
            zi = np.reshape(self.ZV[i, :], (1, self.K))
            infoA = self.updateInfo(infoA, zi, 1)
            hA = self.updateH(hA, zi, xi, 1)

    #
    # Output and reporting
    # 

    def sample_report(self, sampleidx):
        """ Print IBP sample status """
        print('iter %d' % sampleidx)
        print('\tcollapsed loglike = %15.3f' % self.log_likeli())
        print('\tK = %d' % self.K)
        print('\talpha   = %8.2f' % self.alpha)
        print('\tsigma_x = %8.2f' % self.sigma_x)
        print('\tsigma_a = %8.2f' % self.sigma_a)

    def weight_report(self, true_weights=None, round_off=False):
        """ Print learned weights (vs ground truth if available) """
        if true_weights is not None:
            print('\nTrue weights (A)')
            print(str(true_weights))
        print('\nLearned weights (A)')
        # Print rounded or actual weights?
        if round_off:
            print(str(self.weights().astype(np.int)))
        else:
            print(np.array_str(self.weights(), precision=2, suppress_small=True))
        print('')
        # Print V matrix if applicable
        if self.use_v:
            print('\nLatent feature weights (V)')
            print(np.array_str(self.ZV, precision=2))
            print('')
        # Print 'popularity' of latent features
        print('\nLatent feature counts (m)')
        print(np.array_str(self.feature_counts))

    def report_mean_x(self):
        meanA, covarA = self.postA(self.X, self.ZV)
        return self.ZV.dot(meanA)

    #
    # Bookkeeping and calculation methods
    #

    def log_prob_v(self):
        """ Log-likelihood of real-valued latent features V """
        lpv = -0.5 * np.power(self.ZV, 2).sum()
        return lpv - len(self.ZV.nonzero()[0]) * 0.5 * np.log(2 * np.pi)

    def log_indian_buffet_process(self):
        """ Calculate IBP prior contribution log P(Z|alpha) """
        (N, K) = self.ZV.shape
        # Need to find all unique K 'histories'
        Z = (self.ZV != 0).astype(np.int)
        Khs = {}
        for k in range(K):
            history = tuple(Z[:, k])
            Khs[history] = Khs.get(history, 0) + 1
        logp = 0
        logp += self.K * np.log(self.alpha)
        for Kh in Khs.values():
            logp -= self.logFact(Kh)
        logp -= self.alpha * sum([float(1) / i for i in range(1, N + 1)])
        for k in range(K):
            logp += self.logFact(N - self.feature_counts[k]) + self.logFact(self.feature_counts[k] - 1)
            logp -= self.logFact(N)
        if logp == float('inf'):
            pdb.set_trace()
        return logp

    def postA(self, X, Z):
        """ Mean/covar of posterior over weights A """
        M = self.calcM(Z)
        meanA = np.dot(M, np.dot(Z.T, X))
        covarA = self.sigma_x ** 2 * self.calcM(Z)
        return meanA, covarA

    def calcM(self, Z):
        """ Calculate M = (Z' * Z - (sigmax^2) / (sigmaa^2) * I)^-1 """
        return np.linalg.inv(np.dot(Z.T, Z) + (self.sigma_x ** 2)
                             / (self.sigma_a ** 2) * np.eye(self.K))

    def logPX(self, M, Z):
        """ Calculate collapsed log likelihood of data"""
        lp = -0.5 * self.N * self.D * np.log(2 * np.pi)
        lp -= (self.N - self.K) * self.D * np.log(self.sigma_x)
        lp -= self.K * self.D * np.log(self.sigma_a)
        lp -= 0.5 * self.D * np.log(np.linalg.det(np.linalg.inv(M)))
        iminzmz = np.eye(self.N) - np.dot(Z, np.dot(M, Z.T))
        lp -= (0.5 / (self.sigma_x ** 2)) * np.trace(
            np.dot(self.X.T, np.dot(iminzmz, self.X)))
        return lp

    def likelihood_xi(self, zi, meanA, covarA):
        """ Mean/covar of xi given posterior over A """
        meanXi = np.dot(zi, meanA)  # Equation 44
        covarXi = np.dot(zi, np.dot(covarA, zi.T)) + self.sigma_x ** 2  # equation 45
        return meanXi, covarXi

    def updateInfo(self, infoA, zi, addrm):
        """ Add/remove data i to/from information """
        return infoA + addrm * ((1 / self.sigma_x ** 2) * np.dot(zi.T, zi))

    def updateH(self, hA, zi, xi, addrm):
        """ Add/remove data i to/from h"""
        return hA + addrm * ((1 / self.sigma_x ** 2) * np.dot(zi.T, xi))

    #
    # Pure functions (these don't use state or additional params)
    #

    @staticmethod
    def logFact(n):
        return scipy.special.gammaln(n + 1)

    @staticmethod
    def fromInfo(infoA, hA):
        """ Calculate mean/covar from information """
        covarA = np.linalg.inv(infoA)
        meanA = np.dot(covarA, hA)
        return meanA, covarA

    @staticmethod
    def toInfo(meanA, covarA):
        """ Calculate information from mean/covar """
        infoA = np.linalg.inv(covarA)
        hA = np.dot(infoA, meanA)
        return infoA, hA

    @staticmethod
    def log_uniform(v):
        """ 
        Sample uniformly from [0, exp(v)] in the log-domain
        (derive via transform f(x)=log(x) and some calculus...)
        """
        return v + np.log(np.random.uniform(0, 1))

    @staticmethod
    def logBern(log_prob_0, log_prob_1):
        """ Bernoulli sample given log(p0) and log(p1) """
        p1 = 1 / (1 + np.exp(log_prob_0 - log_prob_1))
        return p1 > np.random.uniform(0, 1)

    @staticmethod
    def logPxi(meanLike, covarLike, xi):
        """
        Calculate log-likelihood of a single xi, given its
        mean/covar after collapsing P(A | X_{-i}, Z)
        """
        D = float(xi.shape[1])
        ll = -(D / 2) * np.log(covarLike)
        ll -= (1 / (2 * covarLike)) * np.power(xi - meanLike, 2).sum()
        return ll

    @staticmethod
    def centerData(data):
        return data - IBP.featMeans(data)

    @staticmethod
    def featMeans(data, missing=None):
        """ Replace all columns (features) with their means """
        (N, D) = data.shape
        if missing is None:
            return np.tile(data.mean(axis=0), (N, 1))
        else:
            # Sanity check on 'missing' mask
            # (ensure no totally missing data or features)
            assert (all(missing.sum(axis=0) < N) and
                    all(missing.sum(axis=1) < D))
            # Calculate column means without using the missing data
            censored = data * (np.ones((N, D)) - missing)
            censoredmeans = censored.sum(axis=0) / (N - missing.sum(axis=0))
            return np.tile(censoredmeans, (N, 1))
