# pairs_trading
<p>
This repository contains Python code and a Jupyter notebook that examines the statistics of pairs trading.
</p>

</p>
<blockquote>
<p>
Pairs trading is an approach that takes advantage of the
mispricing between two (or more) co-moving assets, by
taking a long position in one (many) and shorting the
other(s), betting that the relationship will hold and that
prices will converge back to an equilibrium level.
</p>
<p>
<i>Definitive Guide to Pairs Trading</i> availabel from <a href="https://hudsonthames.org/">Hudson and Thames</a>
</p>
</blockquote>
<p>
Pairs trading is sometimes referred to as a statistical arbitrage trading strategy.
</p>
<blockquote>
<p>
Statistical arbitrage and pairs trading tries to solve this problem using price relativity. If two assets share the same
characteristics and risk exposures, then we can assume that their behavior would be similar as well. This has
the benefit of not having to estimate the intrinsic value of an asset but rather just if it is under or overvalued
relative to a peer(s). We only have to focus on the relationship between the two, and if the spread happens
to widen, it could be that one of the securities is overpriced, the other is underpriced, or the mispricing is a
combination of both.
</p>
<p>
<i>Definitive Guide to Pairs Trading</i> availabel from <a href="https://hudsonthames.org/">Hudson and Thames</a>
</p>
</blockquote>
<p>
Pairs trading algorithms have been reported to yield portfolios with Sharpe ratios in excess of 1.0 and returns of 10% or
higher. Pairs trading takes both long and short positions, so the portfolio tends to be market-neutral. A pairs trading portfolio
can have drawdowns, but the drawdowns should be less than a benchmark like the S&P 500 because of the market-neutral nature of the
portfolio.
</p>
<p>
Markets tend toward efficiency and many quantitative approaches fade over time as they are adopted by hedge funds. Pairs trading
goes back to the mid-1980s. Surprisingly, the approach still seems to be profitable. One reason for this could be that there are a vast
number of possible pairs and the pairs portfolio is a faction of the pairs universe. This could
leave unexploited pairs in the market. Pairs trading may also be difficult to scale to a level that would be attractive to institutional
traders, like hedge funds, so the strategy has not been arbitraged out of the market.
</p>
<p>
Mathematical finance often uses models that are based on normal distributions, constant means and standard deviations. Actual market
data is often not normally distributed and changes constantly. The statistics used to select stocks for pairs trading makes an assumption
that the pair distribution has a constant mean and standard deviation (e.g., the pairs spread is a stationary time series). This
assumption holds, at best, over a window of time. This notebook explores the statistics of the cointegrated pairs that are candidates
for pairs trading.
</p>
<p>
The statistics that predict a successful pair will not be accurate in all time periods. For the strategy to be successful, the predicition
must be right more often than not. To minimize the risk in any particular trade, this suggests that trading a larger portfolio will
be more successful than trading a small portfolio.
</p>
