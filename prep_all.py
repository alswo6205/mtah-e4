import os, sys, time, gc
os.chdir(os.path.expanduser(os.environ.get("PATHCRF_DIR", "~/pathcrf")))
sys.path.insert(0, os.getcwd())
import pandas as pd
from datatools.preprocess import Preprocessor
from datatools.sportec_data import SportecData
from datatools import utils
for mid in (sys.argv[1:] or ["J03WMX","J03WN1","J03WOH","J03WOY","J03WPY","J03WQQ","J03WR9"]):
    out=f"data/sportec/tracking_processed/{mid}.parquet"
    if os.path.exists(out): continue
    t=time.time()
    events = pd.read_parquet(f"data/sportec_synced/{mid}.parquet")
    tracking = SportecData(mid, load_tracking=True).tracking.reset_index(drop=True)
    tracking[["timestamp","ball_x","ball_y"]] = tracking[["timestamp","ball_x","ball_y"]].round(2)
    proc = Preprocessor(events, tracking)
    touches = proc.detect_touches()
    touches.to_parquet(f"data/sportec/event_processed/{mid}.parquet")
    tp = utils.calculate_running_features(proc.merge_tracking_poss(touches))
    tp.to_parquet(out)
    print(mid, tp.shape, f"{time.time()-t:.0f}s", flush=True)
    del tracking, proc, tp; gc.collect()
print("PREP_DONE")
