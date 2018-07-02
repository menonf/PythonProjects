import numpy

'''
A random walk is a discrete process whose increments are +/-1 with equal probability.
A Brownian Motion is a continuous time series of random variables
whose increments are i.i.d. normally distributed with 0 mean and variance of 1.
'''


def brownian(timeSteps, rvNorm):
    dt = 1 / timeSteps                                          # time step or time divided by number of trade ticks
    brownianIncrements = rvNorm * numpy.sqrt(dt)
    brownianMotion = numpy.cumsum(brownianIncrements)   # brownian path
    return brownianMotion, brownianIncrements


def brownianWithDrft(mu, sigma, timeSteps, rvNorm):
    dt = 1 / timeSteps
    brownianIncrements = rvNorm * numpy.sqrt(dt)
    brownianMotion = numpy.cumsum(brownianIncrements)
    drift = mu * dt
    diffusion = sigma * brownianMotion
    brownianMotionWithDrift = drift + diffusion
    driftTerm = numpy.arange(timeSteps) * drift
    return brownianMotionWithDrift, driftTerm, brownianIncrements


def geometricBrownian1(s0, mu, sigma, w, T, n):
    t = numpy.linspace(0, 1, n+1)
    S = []
    S.append(s0)
    for i in range(1, int(n+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * w[i-1]
        s_temp = s0*numpy.exp(drift + diffusion)
        S.append(s_temp)
    return S, t
