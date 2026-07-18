"""Network Orchestration Simulation v3 (Colab) - round-2 revision.
Paste into one Colab cell and run. Addresses R2 round-2: imperfect costed
specialist, explicit sensitivity guard, 6 policies, 20-seed CIs, p-sweep."""
import importlib, subprocess, sys
for _p,_i in [("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),
              ("scikit-learn","sklearn"),("matplotlib","matplotlib")]:
    try: importlib.import_module(_i)
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",_p])
def _in_colab():
    try: import google.colab; return True
    except ImportError: return False
IN_COLAB = _in_colab()


import json, os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
OUT = "/content/experiment_v3" if IN_COLAB else "./experiment_v3"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- generator
TASK_TYPES = ["ip_config","vlan_update","firmware_patch","firewall_rule",
              "log_rotation","user_provision","cert_renewal","backup_job",
              "policy_update","incident_response","dns_update","qos_config",
              "vpn_tunnel","ssl_renewal","intrusion_alert"]
DEVICE_TYPES = ["router","switch","firewall","server","access_point",
                "load_balancer","ids_sensor"]
PRIORITIES = ["low","medium","high","critical"]
OBS = ["task_type","device_type","priority","historical_frequency",
       "config_complexity","automation_exists","similar_past_jobs","sensitive"]

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def generate_task(rng, shift=False):
    if not shift:
        task_type = rng.choice(TASK_TYPES)
        priority = rng.choice(PRIORITIES, p=[0.4,0.35,0.2,0.05])
        historical_frequency = int(np.clip(rng.normal(50,30),0,200))
        config_complexity = int(np.clip(rng.normal(8,4),1,30))
    else:
        task_type = rng.choice(TASK_TYPES, p=np.array(
            [1,1,2,3,1,1,2,1,2,3,1,1,2,2,3],dtype=float)/26.0)
        priority = rng.choice(PRIORITIES, p=[0.25,0.35,0.3,0.1])
        historical_frequency = int(np.clip(rng.normal(35,25),0,200))
        config_complexity = int(np.clip(rng.normal(12,5),1,30))
    device_type = rng.choice(DEVICE_TYPES)
    automation_exists = rng.random()<0.45
    similar_past_jobs = int(np.clip(rng.normal(historical_frequency/5,5),0,60))
    sensitive = (task_type in ("incident_response","cert_renewal","intrusion_alert")
                 or priority=="critical")
    p_recur = sigmoid((historical_frequency-35.0)/12.0)
    will_recur = rng.random()<p_recur
    p_feasible = np.clip(0.97-0.015*config_complexity,0.50,0.97)
    feasible = rng.random()<p_feasible
    high_risk = sensitive and (rng.random()<0.85)
    if high_risk: label=2
    elif automation_exists and feasible and will_recur: label=0
    elif (not automation_exists) and feasible and will_recur: label=1
    else: label=2
    return dict(task_type=task_type,device_type=device_type,priority=priority,
                historical_frequency=historical_frequency,
                config_complexity=config_complexity,
                automation_exists=int(automation_exists),
                similar_past_jobs=similar_past_jobs,sensitive=int(sensitive),
                _will_recur=int(will_recur),_feasible=int(feasible),
                _high_risk=int(high_risk),label=label)

def encode(df, cols=None):
    e = pd.get_dummies(df[OBS], columns=["task_type","device_type","priority"])
    return e.reindex(columns=cols, fill_value=0) if cols is not None else e

print("[1/4] Training RF (same design/seed as v2) ...")
rng0 = np.random.default_rng(SEED)
df = pd.DataFrame([generate_task(rng0) for _ in range(5000)])
enc = encode(df); FEATS = list(enc.columns)
X, y = enc.values, df["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                             random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
print(f"      RF test acc: {clf.score(X_te, y_te):.4f}")

# ---------------------------------------------------------------- cost model
P = dict(TIME_MANUAL=12.0, TIME_AUTO=0.5, TIME_DEV=60.0, COST_PER_MIN=1.20,
         P_HUMAN_ERR=0.03, REWORK_MIN=30.0, P_AUTO_FAIL=0.005, FIX_MIN=15.0,
         P_FRAGILE_FAIL=0.30, FRAGILE_FIX_MIN=20.0,
         P_INCIDENT=0.10, INCIDENT_MIN=180.0,
         AI_INFRA_USD_DAY=12.0, AI_MAINT_MIN_DAY=4.3,
         THRESHOLD=5,
         P_SPEC=0.85,          # [R2-1] specialist assessment accuracy (<1)
         REVIEW_MIN=10.0,      # [R2-1] cost of one review event
         MAX_DECLINES=2,       # blacklist after this many declines
         TASKS_PER_DAY=80, DAYS=30)

POLICIES = ["manual","rule","automate_all","automate_all_guard","full_ai","proposed"]
LABELS = {"manual":"Fully manual","rule":"Rule-based router",
          "automate_all":"Automate everything (no guard)",
          "automate_all_guard":"Automate everything + guard",
          "full_ai":"Full AI (no review)",
          "proposed":"Proposed (gradual, AI + review)"}

def make_stream(seed, shift, params):
    """Pre-generate one task stream and batch-predict once (shared across
    policies: common-random-numbers pairing reduces between-policy variance).
    Task semantics are identical to the accepted round-1 design."""
    rng = np.random.default_rng(seed)
    tasks = [generate_task(rng, shift)
             for _ in range(params["DAYS"]*params["TASKS_PER_DAY"])]
    preds = clf.predict(encode(pd.DataFrame(tasks), FEATS).values)
    return tasks, preds


def run_policy(policy, params, stream, seed):
    tasks_all, preds_all = stream
    rng = np.random.default_rng(seed + 77)
    lib, fragile = set(), set()
    pending, pending_bad = Counter(), Counter()
    declines, blacklist = Counter(), set()
    minutes = 0.0; incidents = wasted = reviews = 0
    n = params["TASKS_PER_DAY"]
    for day in range(1, params["DAYS"]+1):
        lo = (day-1)*n
        tasks = tasks_all[lo:lo+n]
        preds = preds_all[lo:lo+n]
        for k, t in enumerate(tasks):
            sig = (t["task_type"], t["device_type"])
            def man():
                nonlocal minutes
                minutes += params["TIME_MANUAL"]
                if rng.random()<params["P_HUMAN_ERR"]: minutes += params["REWORK_MIN"]
            def auto():
                nonlocal minutes, incidents
                minutes += params["TIME_AUTO"]
                if t["_high_risk"] and rng.random()<params["P_INCIDENT"]:
                    minutes += params["INCIDENT_MIN"]; incidents += 1
                elif sig in fragile and rng.random()<params["P_FRAGILE_FAIL"]:
                    minutes += params["FRAGILE_FIX_MIN"]
                elif rng.random()<params["P_AUTO_FAIL"]:
                    minutes += params["FIX_MIN"]
            def dev():
                nonlocal minutes, wasted
                minutes += params["TIME_DEV"]; lib.add(sig)
                if not t["_feasible"]: fragile.add(sig)
                if not t["_will_recur"]: wasted += 1
            # [R2-2] EXPLICIT GUARD: observable-sensitive -> human, takes
            # precedence over the classifier and over the library.
            guarded = policy in ("rule","automate_all_guard","full_ai","proposed")
            if guarded and t["sensitive"]:
                man(); continue
            if policy == "manual":
                man()
            elif policy == "rule":
                if sig in lib: auto()
                elif t["automation_exists"] and t["similar_past_jobs"]>=5:
                    lib.add(sig); auto()   # register existing automation (free)
                elif (not t["automation_exists"]) and t["historical_frequency"]>=40:
                    dev(); auto()
                else: man()
            elif policy in ("automate_all","automate_all_guard"):
                if sig in lib: auto()
                else: dev(); auto()
            elif policy == "full_ai":
                pr = preds[k]
                if pr==2 and sig not in lib: man()
                elif sig in lib: auto()
                elif pr==0 and t["automation_exists"]:
                    lib.add(sig); auto()   # register existing automation (free)
                else: dev(); auto()
            elif policy == "proposed":
                pr = preds[k]
                if sig in blacklist: man(); continue
                if pr==2 and sig not in lib: man()
                elif sig in lib: auto()
                elif pr==0 and t["automation_exists"]:
                    lib.add(sig); auto()   # register existing automation (free)
                else:
                    pending[sig] += 1
                    if (not t["_will_recur"]) or (not t["_feasible"]):
                        pending_bad[sig] += 1
                    if pending[sig] >= params["THRESHOLD"]:
                        # [R2-1] imperfect, costed review of the ACCUMULATED
                        # evidence: the specialist inspects the THRESHOLD
                        # recommendations gathered for this signature; the
                        # candidate is truly bad if a majority of them were
                        # one-off or infeasible. The assessment is correct
                        # with probability P_SPEC and inverted otherwise.
                        minutes += params["REVIEW_MIN"]; reviews += 1
                        bad = 2*pending_bad[sig] > pending[sig]
                        assessed_bad = bad if rng.random()<params["P_SPEC"] else (not bad)
                        pending[sig] = 0; pending_bad[sig] = 0
                        if assessed_bad:
                            declines[sig] += 1
                            if declines[sig] >= params["MAX_DECLINES"]:
                                blacklist.add(sig)
                            man()
                        else:
                            dev(); auto()
                    else:
                        man()
    cost = minutes*params["COST_PER_MIN"]
    if policy in ("full_ai","proposed"):
        cost += params["DAYS"]*(params["AI_INFRA_USD_DAY"]
                                + params["AI_MAINT_MIN_DAY"]*params["COST_PER_MIN"])
    return dict(policy=policy, hours=minutes/60.0, cost=cost,
                incidents=incidents, wasted=wasted,
                fragile=len(fragile), reviews=reviews, library=len(lib))

# ------------------------------------------------- [R2-1] 20-seed CI runs
N_SEEDS = 20
print(f"\n[2/4] Running 6 policies x {N_SEEDS} seeds x 2 environments ...")

def ci95(a):
    a = np.asarray(a, float)
    m = a.mean()
    h = stats.t.ppf(0.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.0
    return m, h

streams = {("shift" if sh else "indist", s): make_stream(1000+s, sh, P)
           for sh in (False, True) for s in range(N_SEEDS)}
results = {}   # (env, policy) -> list of run dicts
for env in ("indist","shift"):
    for pol in POLICIES:
        results[(env,pol)] = [run_policy(pol, P, streams[(env,s)], 1000+s)
                              for s in range(N_SEEDS)]

def summarize(env):
    man_costs = np.array([r["cost"] for r in results[(env,"manual")]])
    rows = []
    for pol in POLICIES:
        runs = results[(env,pol)]
        get = lambda k: np.array([r[k] for r in runs])
        sav = (1 - get("cost")/man_costs)*100.0
        cm, ch = ci95(get("cost")); hm, hh = ci95(get("hours"))
        im, ih = ci95(get("incidents")); wm, wh = ci95(get("wasted"))
        sm, sh = ci95(sav)
        rvm, _ = ci95(get("reviews"))
        rows.append(dict(policy=pol, reviews=f"{rvm:.0f}",
                         hours=f"{hm:.1f} \u00b1 {hh:.1f}",
                         cost=f"{cm:,.0f} \u00b1 {ch:,.0f}",
                         incidents=f"{im:.1f} \u00b1 {ih:.1f}",
                         wasted=f"{wm:.1f} \u00b1 {wh:.1f}",
                         savings=("\u2014" if pol=="manual" else f"{sm:.1f} \u00b1 {sh:.1f}%"),
                         _cost=cm, _sav=sm, _sav_h=sh, _inc=im, _wst=wm))
    return pd.DataFrame(rows)

sum_in = summarize("indist"); sum_sh = summarize("shift")
print("\nIN-DISTRIBUTION (mean \u00b1 95% CI):")
print(sum_in[["policy","hours","cost","incidents","wasted","reviews","savings"]].to_string(index=False))
print("\nSHIFTED:")
print(sum_sh[["policy","hours","cost","incidents","wasted","reviews","savings"]].to_string(index=False))

# ------------------------------------------------- [R2-1] specialist sweep
print("\n[3/4] Sweeping specialist accuracy p ...")
P_GRID = [0.5,0.6,0.7,0.8,0.85,0.9,0.95,1.0]
SWEEP_SEEDS = 10
sweep = {}
for env in ("indist","shift"):
    man = np.array([r["cost"] for r in results[(env,"manual")][:SWEEP_SEEDS]])
    rows = []
    for p in P_GRID:
        pp = dict(P); pp["P_SPEC"] = p
        runs = [run_policy("proposed", pp, streams[(env,s)], 1000+s)
                for s in range(SWEEP_SEEDS)]
        costs = np.array([r["cost"] for r in runs])
        m, h = ci95((1-costs/man)*100.0)
        wst = float(np.mean([r["wasted"] for r in runs]))
        rvw = float(np.mean([r["reviews"] for r in runs]))
        rows.append((p, m, h, wst, rvw))
    sweep[env] = rows
    print(f"  {env}: " + ", ".join(f"p={p}:{m:.1f}%/w{w:.0f}/r{r:.0f}"
                                   for p,m,h,w,r in rows))

rule_in = sum_in.loc[sum_in.policy=="rule","_sav"].iloc[0]
rule_sh = sum_sh.loc[sum_sh.policy=="rule","_sav"].iloc[0]

# ---------------------------------------------------------------- figures
print("\n[4/4] Figures ...")
colors = {"manual":"#C44E52","rule":"#DD8452","automate_all":"#8172B3",
          "automate_all_guard":"#64B5CD","full_ai":"#937860","proposed":"#55A868"}

# Fig A: policy comparison with CIs (both envs, grouped bars of savings)
fig, ax = plt.subplots(figsize=(10,5.4))
pols = ["rule","automate_all","automate_all_guard","full_ai","proposed"]
xp = np.arange(len(pols)); w=0.38
si = [sum_in.loc[sum_in.policy==p,"_sav"].iloc[0] for p in pols]
ei = [sum_in.loc[sum_in.policy==p,"_sav_h"].iloc[0] for p in pols]
ss = [sum_sh.loc[sum_sh.policy==p,"_sav"].iloc[0] for p in pols]
es = [sum_sh.loc[sum_sh.policy==p,"_sav_h"].iloc[0] for p in pols]
ax.bar(xp-w/2, si, w, yerr=ei, capsize=4, label="In-distribution", color="#4C72B0")
ax.bar(xp+w/2, ss, w, yerr=es, capsize=4, label="Shifted", color="#DD8452")
for i,(a,b) in enumerate(zip(si,ss)):
    ax.text(i-w/2, a+1.2, f"{a:.1f}%", ha="center", fontsize=9)
    ax.text(i+w/2, b+1.2, f"{b:.1f}%", ha="center", fontsize=9)
ax.set_xticks(xp); ax.set_xticklabels([LABELS[p] for p in pols], rotation=12, ha="right")
ax.set_ylabel("Cost savings vs. fully manual (%)")
ax.set_title(f"Savings by policy, mean \u00b1 95% CI over {N_SEEDS} runs "
             f"(specialist accuracy p = {P['P_SPEC']})")
ax.legend(); plt.tight_layout()
plt.savefig(f"{FIG}/fig_policy_ci.png", dpi=300); plt.show()
plt.close()

# Fig B: specialist-accuracy sweep
fig, ax = plt.subplots(figsize=(8.5,5.2))
for env, lab, col in (("indist","In-distribution","#4C72B0"),
                      ("shift","Shifted","#DD8452")):
    xs=[r[0] for r in sweep[env]]; ys=[r[1] for r in sweep[env]]
    hs=[r[2] for r in sweep[env]]
    ax.errorbar(xs, ys, yerr=hs, marker="o", capsize=3, color=col,
                label=f"Proposed \u2013 {lab}")
ax.axhline(rule_in, ls="--", color="#4C72B0", alpha=0.6,
           label="Rule-based router \u2013 In-distribution")
ax.axhline(rule_sh, ls="--", color="#DD8452", alpha=0.6,
           label="Rule-based router \u2013 Shifted")
ax.axvline(P["P_SPEC"], ls=":", color="grey")
ax.set_xlabel("Specialist assessment accuracy p")
ax.set_ylabel("Cost savings vs. fully manual (%)")
ax.set_title("Effect of specialist reliability on the proposed framework")
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{FIG}/fig_specialist_sweep.png", dpi=300); plt.show()
plt.close()

# Persist
out = dict(
    parameters=P, n_seeds=N_SEEDS,
    indist=sum_in.drop(columns=[c for c in sum_in.columns if c.startswith("_")]).to_dict("records"),
    shifted=sum_sh.drop(columns=[c for c in sum_sh.columns if c.startswith("_")]).to_dict("records"),
    indist_raw={p:[r for r in results[("indist",p)]] for p in POLICIES},
    shifted_raw={p:[r for r in results[("shift",p)]] for p in POLICIES},
    sweep={env:[[p,round(m,2),round(h,2),round(w,1),round(r,1)]
                for p,m,h,w,r in rows] for env,rows in sweep.items()},
)
with open(f"{OUT}/results_v3.json","w") as f: json.dump(out,f,indent=1)
sum_in.to_csv(f"{OUT}/policy_summary_ci_indist.csv", index=False)
sum_sh.to_csv(f"{OUT}/policy_summary_ci_shifted.csv", index=False)
print("Done.")

import zipfile
zp = os.path.join(OUT,"experiment_v3_results.zip")
with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as zf:
    for fn in ["results_v3.json","policy_summary_ci_indist.csv",
               "policy_summary_ci_shifted.csv"]:
        zf.write(os.path.join(OUT,fn), arcname=fn)
    for fn in sorted(os.listdir(FIG)): zf.write(os.path.join(FIG,fn), f"figures/{fn}")
print("ZIP ->", zp)
if IN_COLAB:
    try:
        from google.colab import files; files.download(zp)
    except Exception as e: print("(use file browser)", e)
