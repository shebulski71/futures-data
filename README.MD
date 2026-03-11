A slightly bigger run:
prefect deployment run "backfill_futures_1m/backfill-1m" \
  --param start="2015-04-01" \
  --param end="2016-01-01" \
  --param transcode_jobs=8

prefect deployment run "backfill_futures_1m/backfill-1m" \
  --param start="2015-04-01" \
  --param end="2015-07-01"

If you want a “repair” style rerun:
prefect deployment run "backfill_futures_1m/backfill-1m" \
  --param start="2015-01-01" \
  --param end="2015-04-01" \
  --param force_window=true \
  --param transcode_force=true