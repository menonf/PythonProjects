import YieldCurve as YC
import pandas as pd
from matplotlib import pyplot
from scipy.interpolate import splrep, splev

bond_table = pd.read_csv('E:\Repositories\PythonProjects\FixedIncome\YieldCurve\AppleBonds_101519.csv')
bond_table = bond_table.drop(['Asset Name', 'Mkt Value'], axis=1)

SC = YC.SpotCurve()
Spot1 = SC.SpotRate(bond_table['Time To Maturity'], bond_table['Yield To Maturity'])
Spot2 = []
for index, row in bond_table.iterrows():
    spot, interest = SC.SpotRateWithCouponInfo(row['Par'], row['Price'], row['Time To Maturity'], row['Coupon'] * 100,
                                               row['Coupon Frequency'])
    Spot2.append(spot)


#  --------- USES NEWTON RAPHSON - WILL TAKE A LONG TIME DEPENDING ON THE NUMBER OF BONDS -------------------------#
Spot3 = [bond_table['Yield To Maturity'].iloc[0]]  # TODO: check that this is the correct rate
SC.SpotRatesNumericalApproximation(bond_table['Yield To Maturity'], Spot3, 1, True)  # kick off the recursive code
# ----------------------------------------------------------------------------------------------------------------#


def plot_yield_curves(time_to_maturity, interpolated_spot_rates, interpolated_ytm, spot_rates, ytm):
    fig, (ax1, ax2) = pyplot.subplots(1, 2)
    fig.suptitle('Apple(AAA Rated) Yield Curve as of 10/15/2019', fontsize=12)
    ax1.plot(time_to_maturity, interpolated_spot_rates, marker='*', color='green', label='Spot Rates')
    ax1.plot(time_to_maturity, interpolated_ytm, marker='*', color='blue', label='Yield Curve')
    ax2.plot(time_to_maturity, spot_rates, marker='*', color='green', label='Spot Rates Curve')
    ax2.plot(time_to_maturity, ytm, marker='*', color='blue',
             label='Yield to Maturity')
    ax1.set_xlabel("Maturity in Years")
    ax1.legend()
    ax2.set_xlabel("Maturity in Years")
    ax2.legend()
    pyplot.ylabel("Rates (%)")
    pyplot.show()


time_to_maturity_idx = list(i for i in range(1, bond_table['Time To Maturity'].count() + 1))
bspl = splrep(bond_table['Time To Maturity'], bond_table['Yield To Maturity'], s=0.2)
bspl_y = splev(bond_table['Time To Maturity'], bspl)

bspl1 = splrep(bond_table['Time To Maturity'], Spot1, s=0.2)
bspl_y1 = splev(bond_table['Time To Maturity'], bspl1)

bspl2 = splrep(bond_table['Time To Maturity'], Spot2, s=0.2)
bspl_y2 = splev(bond_table['Time To Maturity'], bspl2)

bspl3 = splrep(bond_table['Time To Maturity'], Spot3, s=0.2)
bspl_y3 = splev(bond_table['Time To Maturity'], bspl3)

plot_yield_curves(bond_table['Time To Maturity'], bspl_y1, bspl_y, Spot1, bond_table['Yield To Maturity'])
plot_yield_curves(bond_table['Time To Maturity'], bspl_y2, bspl_y, Spot2, bond_table['Yield To Maturity'])
plot_yield_curves(bond_table['Time To Maturity'], bspl_y3, bspl_y, Spot3, bond_table['Yield To Maturity'])
