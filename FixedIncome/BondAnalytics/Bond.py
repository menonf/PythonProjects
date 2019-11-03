import scipy.optimize as optimize


class Bond(object):
    def __init__(self, par_value, market_price, time_to_maturity, coupon_rate,  coupon_frequency, yield_to_maturity):
        self.par_value = par_value
        self.market_price = market_price
        self.time_to_maturity = time_to_maturity
        self.coupon_rate = coupon_rate
        self.coupon_frequency = coupon_frequency

        if isinstance(yield_to_maturity, (int, float, complex)):
            self.yield_to_maturity = yield_to_maturity
        else:
            self.yield_to_maturity = self.YieldToMaturity(self.market_price, 0.05)

    def ZeroCouponBondPrice(self):
        return self.par_value / (1 + self.yield_to_maturity) ** self.time_to_maturity

    def ZeroCouponYieldToMaturity(self):  # also known as spot rate or zero rate
        return ((self.par_value / self.market_price) ** (1/self.time_to_maturity)) - 1

    def BondPrice(self, YieldToMaturity):
        coupon_periods = self.time_to_maturity * self.coupon_frequency
        coupon_payments = self.par_value * self.coupon_rate
        dt = [(i + 1) / self.coupon_frequency for i in range(int(coupon_periods))]
        fair_price = sum([coupon_payments / self.coupon_frequency / (1 + YieldToMaturity / self.coupon_frequency) ** (self.coupon_frequency * t) for t in dt]) + \
                self.par_value / (1 + YieldToMaturity / self.coupon_frequency) ** (self.coupon_frequency * self.time_to_maturity)
        return fair_price

    def CurrentYield(self):
        return (self.coupon_rate * self.par_value) / self.market_price

    def YieldToMaturity(self, price, guess):
        coupon_periods = self.time_to_maturity * self.coupon_frequency
        coupon_payments = self.par_value * (self.coupon_rate / self.coupon_frequency)
        ytm_func = lambda r: -(-price + (1 - (pow(1 + r, -coupon_periods))) * (coupon_payments / r)) / (pow(1 + r, -coupon_periods)) - self.par_value
        return optimize.newton(ytm_func, guess) * self.coupon_frequency

    def Duration(self):
        coupon_periods = self.time_to_maturity * self.coupon_frequency
        ytm = self.yield_to_maturity
        pv = ((self.par_value * self.coupon_rate / self.coupon_frequency * (1 - (1 + ytm / self.coupon_frequency) ** (-coupon_periods))) / (ytm / self.coupon_frequency)) + \
                       self.par_value * (1 + (ytm / self.coupon_frequency)) ** (-coupon_periods)

        final_year = (self.time_to_maturity * self.par_value) / \
                     (pv * (1 + (ytm / self.coupon_frequency)) ** (self.coupon_frequency * self.time_to_maturity))

        summation = 0
        i = 1
        while i <= coupon_periods:
            tn = i / self.coupon_frequency
            i = i + 1
            dur = (self.par_value * self.coupon_rate * tn) / \
                  (pv * self.coupon_frequency * (1 + (ytm / self.coupon_frequency)) ** (self.coupon_frequency * tn))
            summation = summation + dur

        macaulay_duration = summation + final_year
        modified_duration = macaulay_duration / (1 + ytm/self.coupon_frequency)
        return macaulay_duration, modified_duration

    # Below is close approximation of Duration - This is a simpler calculation
    # Also known as Key Rate Duration when yield change is 0.01
    def DurationByPriceSensitivity(self, yield_change):
        ytm = self.yield_to_maturity
        ytm_minus = ytm - yield_change
        price_minus = self.BondPrice(ytm_minus)
        ytm_plus = ytm + yield_change
        price_plus = self.BondPrice(ytm_plus)
        modified_duration = (price_minus - price_plus) / (2 * self.market_price * yield_change)
        return modified_duration

    def DV01ByModifiedDuration(self):
        return (self.Duration()[1] * self.market_price) / 10000

    def DV01ByPriceSensitivity(self, yield_change):
        ytm = self.yield_to_maturity
        ytm_minus = ytm - yield_change
        price_minus = self.BondPrice(ytm_minus)
        ytm_plus = ytm + yield_change
        price_plus = self.BondPrice(ytm_plus)
        DV01 = (price_minus - price_plus) / (2 * 100)
        return DV01

    # Below is close approximation of convexity - This is a simpler calculation
    # Than the actual formula which follows tedious summation we have already seen in the Duration() function above
    def convexity(self, yield_change):
        ytm = self.yield_to_maturity
        ytm_minus = ytm - yield_change
        price_minus = self.BondPrice(ytm_minus)
        ytm_plus = ytm + yield_change
        price_plus = self.BondPrice(ytm_plus)
        convexity = (price_minus + price_plus - 2 * self.market_price) / (self.market_price * yield_change ** 2)
        return convexity
