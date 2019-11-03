import numpy as np
from sympy.solvers import solve
from sympy import Symbol


class SpotCurve(object):
    def __init__(self,):
        self.spot_rates_with_coupon_info = []
        self.time_to_maturity = []

    def SpotRate(self,time_to_maturity, yield_to_maturity):
        time_to_maturity_idx = list(i for i in range(1, time_to_maturity.count() + 1))
        yield_to_maturity = yield_to_maturity.to_list()

        spot_rate = []
        for i in range(0, len(time_to_maturity_idx)):  # calculate i-th spot rate
            summation = 0
            for j in range(0, i):
                summation = summation + yield_to_maturity[i] / (1 + spot_rate[j]) ** time_to_maturity_idx[j]
            value = ((1 + yield_to_maturity[i]) / (1 - summation)) ** (1 / time_to_maturity_idx[i]) - 1
            spot_rate.append(value)
        return spot_rate

    def SpotRateWithCouponInfo(self, par, price, maturity, coupon, frequency=2):
        coupon_npv = 0.0

        if coupon != 0.0:
            coupon_npv = self.bond_intermediate_coupon_npv(coupon, frequency, periods=np.int(maturity * frequency))

        final_flow = par + coupon / frequency if maturity % (1 / frequency) == 0.0 else par
        final_price = price - coupon_npv
        total_interest_earned = final_flow / final_price
        spot = ((1 / maturity) * np.log(total_interest_earned))
        total_interest_earned = total_interest_earned - 1
        self.spot_rates_with_coupon_info.append(spot)
        self.time_to_maturity.append(maturity)
        return spot, total_interest_earned

    def bond_intermediate_coupon_npv(self, coupon, frequency, periods):
        coupon_amt = coupon / frequency
        coupon_npv = 0

        for i in range(1, int(periods)):
            period = i / frequency
            period_spot = np.interp(period, self.time_to_maturity , self.spot_rates_with_coupon_info)  # interpolate
            discounted_value = coupon_amt / np.exp(period_spot * period)
            coupon_npv += discounted_value
        return coupon_npv

    def SpotRatesNumericalApproximation(self, yield_to_maturity, spotRates, n, verbose):
        x = Symbol('x', real=True)
        if len(spotRates) >= len(yield_to_maturity):
            print("\n\n\t+zero curve boot strapped [%d iterations]" % n)
            return
        else:
            debug_msg = ''
            for i in range(0, len(spotRates), 1):
                if i == 0:
                    debug_msg = '%2.6f/(1+%2.6f)**%d' % (yield_to_maturity[n], spotRates[i], i + 1)
                else:
                    debug_msg = debug_msg + ' +%2.6f/(1+%2.6f)**%d' % (yield_to_maturity[n], spotRates[i], i + 1)
            debug_msg = debug_msg + '+ (1+%2.6f)/(1+x)**%d-1' % (yield_to_maturity[n], n + 1)
            if verbose:
                print("-[%d] %s" % (n, debug_msg.strip()))
            rate1 = solve(eval(debug_msg), x)  # solve the expression for this iteration
            # Abs here since some solutions can be complex
            rate1 = min([np.real(np.abs(r)) for r in rate1])
            if verbose:
                print("-[%d] solution %2.6f" % (n, float(rate1)))
            spotRates.append(rate1)  # stuff the new rate in the results, will be used by the next iteration
            self.SpotRatesNumericalApproximation(yield_to_maturity, spotRates, n + 1, verbose)


class ForwardRates(object):
    def forward_rate(self, time_to_maturity, spot_rates):
        forward_rates = []
        forward_rates.append(spot_rates[1])
        for i in range(0, len(spot_rates), 1):

            if i < len(spot_rates) - 1:
                forward_rate = (((1 + spot_rates[i+1]) ** time_to_maturity[i+1]) /
                        ((1 + spot_rates[i]) ** time_to_maturity[i])) - 1
                forward_rates.append(forward_rate)
        return forward_rates




