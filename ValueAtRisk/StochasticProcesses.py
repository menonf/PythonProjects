import numpy

def brownian(timeSteps, rvNorm):
    dt = 1 / timeSteps                                  # time step or time divided by number of trade ticks
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


def geometricbrownianWithDrft(mu, sigma, timeSteps, rvNorm):
    dt = 1 / timeSteps
    brownianIncrements = rvNorm * numpy.sqrt(dt)
    brownianMotion = numpy.cumsum(brownianIncrements)
    drift = (mu - (numpy.square(sigma) * 0.5)) * dt
    diffusion = sigma * brownianMotion
    geometricBrownianMotionWithDrift = numpy.exp(drift + diffusion)
    geometricDriftTerm = numpy.exp(numpy.arange(timeSteps) * drift)
    return geometricBrownianMotionWithDrift, geometricDriftTerm, brownianIncrements
