import Bond as BD

# For Theory - https://www.wallstreetmojo.com/category/risk-management/fixed-income/
# Validate the results using - http://facweb.plattsburgh.edu/razvan.pascalau/BondForm.php

B1 = BD.Bond(par_value=100, market_price=95.0428, time_to_maturity=1.5, coupon_rate=5.75/100, coupon_frequency=2,
             yield_to_maturity='to be calculated')

print('Current Yield = ' + '{:.2%}'.format(B1.CurrentYield()))
print('Yield to Maturity = ' + '{:.2%}'.format(B1.YieldToMaturity(price=B1.market_price, guess=0.05)))
print('Macaulay Duration = ' + str(B1.Duration()[0]))
print('Modified Duration = ' + str(B1.Duration()[1]))
print('Key Rate Duration = ' + str(B1.DurationByPriceSensitivity(yield_change=0.01)))
print('DV01 Estimated by Modified Duration = ' + str(B1.DV01ByModifiedDuration()))
print('DV01 Estimated by Price Sensitivity = ' + str(B1.DV01ByPriceSensitivity(yield_change=0.01)))
print('Convexity = ' + str(B1.convexity(yield_change=0.01)))
print('Bond Fair Price at 10% YTM = ' + str(B1.BondPrice(0.1)))
