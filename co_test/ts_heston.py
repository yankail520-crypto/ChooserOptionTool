import pandas as pd
from common.paths import get_outputs_path

df = pd.read_parquet(get_outputs_path() / "heston_params.parquet")
ok = df[df.status == "ok"]
print(len(df), "个快照，成功", len(ok))
print(ok[["rmse_iv", "v0", "kappa", "theta", "xi", "rho", "feller_ratio", "n_options"]]
      .describe().round(3).loc[["count", "mean", "min", "50%", "max"]])
print({"kappa>=24.9": int((ok.kappa >= 24.9).sum()), "kappa<=0.11": int((ok.kappa <= 0.11).sum()),
       "xi>=3.9": int((ok.xi >= 3.9).sum()), "|rho|>=0.99": int((ok.rho.abs() >= 0.99).sum()),
       "v0<=0.0011": int((ok.v0 <= 0.0011).sum()), "theta>=0.99": int((ok.theta >= 0.99).sum())})
print(df[df.status != "ok"][["date", "status"]])



import pandas as pd
from co_bsm import calibrate as cal

chain, feat = cal.load_inputs()
d = pd.Timestamp("2024-06-03")
S, r, q = cal.market_inputs(feat, d)
opts, _ = cal.select_options(chain[chain["date"] == d], S, r, q)

orig = list(cal.BOUNDS)
for kmax in (25.0, 15.0, 10.0, 5.0):
    cal.BOUNDS[1] = (0.1, kmax)
    res = cal.calibrate_snapshot(opts, S, r, q)
    print(f"kappa ≤ {kmax:>4}: RMSE {res['rmse_iv']:.3f}  kappa={res['kappa']:.2f} "
          f"xi={res['xi']:.2f} rho={res['rho']:.2f} v0={res['v0']:.3f} theta={res['theta']:.3f}")
cal.BOUNDS[:] = orig


ok = df[df.status == "ok"].copy()
ok["at_bound"] = (ok.kappa >= 24.9) | (ok.xi >= 3.9)
cols = ["date", "S", "rmse_iv", "v0", "kappa", "theta", "xi", "rho", "feller_ratio"]
print("撞边界的快照：")
print(ok[ok.at_bound][cols].round(3).to_string(index=False))
print("\nRMSE 最大的 5 个：")
print(ok.nlargest(5, "rmse_iv")[cols].round(3).to_string(index=False))
print("\nv0 最大的 3 个：")
print(ok.nlargest(3, "v0")[cols].round(3).to_string(index=False))