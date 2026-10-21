# ADS injury rates by crash condition

Results from `scripts/eda_injury_rates.py` (2,878 ADS crashes; groups under 30 crashes excluded).

## Findings

- **Overall:** 9.8% of ADS crashes involve an injury.
- **Crash partner:** cyclist 41% (n=46) and motorcycle 26% (n=34), against about 10–11% for cars, SUVs and vans, and 1–3% for buses and heavy trucks.
- **Road type:** intersection 12%, street 10%, parking lot 2%.
- **Speed:** 55% of crashes happen at 0 mph (the ADS vehicle was stopped), with a 9% injury rate. The rate peaks at 11–25 mph (13%).
- **Weather:** little difference (clear 10%, cloudy 8%, rain 7%).

## Modeling takeaways

- Crash partner is the strongest signal but is also a reporting trigger, so models are compared with and without it.
- The other conditions vary by only a few points, so gains over the baseline may be small.

## Limitations

- Rates are among reported crashes only; there is no mileage data.
- Cyclist, motorcycle and 45+ mph groups are small, so their rates are uncertain.
- Injury severity is alleged by the reporting company, not verified.
