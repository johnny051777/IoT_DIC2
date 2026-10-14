"""CRISP-DM 多元線性迴歸 + 5 種特徵選擇 (任意 CSV)。純 numpy/scikit-learn，供 Vercel Function 與本機 dev_server 共用。"""
import csv, io, os, ipaddress, socket, itertools, warnings, urllib.request, urllib.parse
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, ElasticNetCV, lasso_path
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
warnings.filterwarnings("ignore")

SEED, MAX_EX = 42, 13
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRESETS = {"startups": "startups.csv", "boston": "boston.csv"}

def fetch(url):
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("URL 必須是 http(s) 開頭的 CSV 連結")
    for ai in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(ai[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("不允許存取內部/私有網址")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=12) as r:
        data = r.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("檔案超過 5MB 限制")
    return data.decode("utf-8-sig", errors="replace")

def parse(text, target=None):
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("CSV 內容為空")
    head, body = [h.strip() for h in rows[0]], rows[1:]
    if len(body) < 15:
        raise ValueError(f"資料列數太少 (僅 {len(body)} 列)，多元迴歸至少需要 15 列以上")
    
    col = {h: [(r[i].strip() if i < len(r) else "") for r in body] for i, h in enumerate(head)}
    
    def num(v):
        o = np.full(len(v), np.nan)
        for i, s in enumerate(v):
            try: o[i] = float(s)
            except ValueError: pass
        return o

    target = (target or "").strip() or head[-1]
    if target not in col:
        raise ValueError(f"找不到目標欄位 '{target}'；可用欄位包括: {', '.join(head)}")
    
    y = num(col[target])
    if np.isnan(y).mean() > 0.5:
        raise ValueError(f"目標欄位 '{target}' 不是有效的數值型態")
    
    keep, names, cols, notes = ~np.isnan(y), [], [], []
    stats = {}

    # Target stats
    valid_y = y[keep]
    stats[target] = {
        "min": round(float(np.min(valid_y)), 2),
        "max": round(float(np.max(valid_y)), 2),
        "mean": round(float(np.mean(valid_y)), 2),
        "median": round(float(np.median(valid_y)), 2),
        "std": round(float(np.std(valid_y)), 2)
    }

    for h in head:
        if h == target:
            continue
        v, a = col[h], num(col[h])
        filled = [s for s in v if s and s.upper() not in ("NA", "NAN", "NULL")]
        if filled and np.isfinite(a).sum() >= 0.85 * len(filled): # 數值特徵
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            a = np.where(np.isnan(a), med, a)
            if a.std() > 0:
                names.append(h)
                cols.append(a)
                stats[h] = {
                    "min": round(float(np.min(a)), 2),
                    "max": round(float(np.max(a)), 2),
                    "mean": round(float(np.mean(a)), 2),
                    "median": round(float(np.median(a)), 2),
                    "std": round(float(np.std(a)), 2)
                }
        else: # 類別特徵 -> one-hot (drop first)
            cats = sorted(set(v))
            if 2 <= len(cats) <= 10:
                for c in cats[1:]:
                    feat_name = f"{h}_{c}"
                    arr = np.array([float(s == c) for s in v])
                    names.append(feat_name)
                    cols.append(arr)
                    stats[feat_name] = {
                        "min": 0.0, "max": 1.0,
                        "mean": round(float(np.mean(arr)), 2),
                        "median": float(np.median(arr)),
                        "std": round(float(np.std(arr)), 2)
                    }
                notes.append(f"{h}: One-Hot 編碼 ({len(cats)} 類: {', '.join(cats)})")
            else:
                notes.append(f"{h}: 已略過 (非連續數值且類別過多或單一)")

    if len(names) < 2:
        raise ValueError("特徵欄位不足，至少需要 2 個有效數值特徵進行多元線性迴歸")

    X = np.column_stack(cols)[keep]
    raw_preview = {
        "headers": head,
        "rows": body[:8],
        "total_rows": len(body),
        "stats": stats
    }
    return names, X, y[keep], target, notes, raw_preview

def r4(a):
    return np.round(np.asarray(a, float), 4).tolist()

def analyze(text, target=None):
    F, X, y, target, notes, raw_preview = parse(text, target)
    p = len(F)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    
    folds = [(Xtr[a], ytr[a], Xtr[b], ytr[b]) for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xtr)]
    
    def cv(idx):
        idx = list(idx)
        rm, r2 = [], []
        for A, ya, B, yb in folds:
            w = np.linalg.lstsq(np.c_[np.ones(len(A)), A[:, idx]], ya, rcond=None)[0]
            e = yb - np.c_[np.ones(len(B)), B[:, idx]] @ w
            rm.append(np.sqrt((e ** 2).mean()))
            denom = ((yb - yb.mean()) ** 2).sum()
            r2.append(1 - (e ** 2).sum() / (denom if denom > 1e-12 else 1e-12))
        return float(np.mean(rm)), float(np.mean(r2))

    def parsimonious(d, tol=0.01):
        b = min(d.values())
        return min(k for k, v in d.items() if v <= b * (1 + tol))

    SEL, curves = {}, {}

    # 1. Lasso (CV + 1-SE)
    la = LassoCV(cv=5, random_state=SEED).fit(Xs, ytr)
    m = la.mse_path_.mean(1)
    se = la.mse_path_.std(1) / np.sqrt(la.mse_path_.shape[1])
    i = int(np.argmin(m))
    a1 = la.alphas_[m <= m[i] + se[i]].max()
    lc = Lasso(alpha=a1).fit(Xs, ytr).coef_
    SEL["Lasso (1-SE Rule)"] = [j for j in range(p) if abs(lc[j]) > 1e-8] or [int(np.argmax(np.abs(la.coef_)))]
    al, co, _ = lasso_path(Xs, ytr, alphas=la.alphas_)
    lasso_p = {"x": r4(np.log10(al)[::2]), "coefs": [r4(c[::2]) for c in co], "sel": float(np.log10(a1))}

    # 2. Forward SFS
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select="auto", tol=0.005, direction="forward", scoring="r2", cv=5).fit(Xs, ytr)
    SEL["Forward SFS"] = [int(j) for j in np.where(sfs.get_support())[0]]

    # 3. RFE (RFECV)
    rf = RFECV(LinearRegression(), step=1, cv=5, scoring="neg_root_mean_squared_error").fit(Xs, ytr)
    SEL["RFE (RFECV)"] = [int(j) for j in np.where(rf.support_)[0]]
    curves["RFE"] = {"k": list(range(1, p + 1)), "rmse": r4(-rf.cv_results_["mean_test_score"])}

    # 4. ElasticNet Ranking
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, random_state=SEED).fit(Xs, ytr)
    rank = [int(j) for j in np.argsort(-np.abs(en.coef_))]
    encv = {k: cv(rank[:k])[0] for k in range(1, p + 1)}
    SEL["ElasticNet Ranking"] = rank[:parsimonious(encv)]
    curves["ElasticNet Ranking"] = {"k": list(encv), "rmse": r4(list(encv.values()))}

    # 5. Exhaustive Search
    pool = list(range(p)) if p <= MAX_EX else rank[:MAX_EX]
    ex = []
    for k in range(1, len(pool) + 1):
        for s in itertools.combinations(pool, k):
            ex.append((s, k, *cv(s)))
    best = min(e[2] for e in ex)
    SEL["Exhaustive Search"] = list(min((e for e in ex if e[2] <= best * 1.01), key=lambda e: (e[1], e[2]))[0])
    top = sorted(ex, key=lambda e: (e[2], e[1]))[:10]

    # Baseline
    SEL = {"All Features (Baseline)": list(range(p)), **SEL}

    res, pred, coefs, models = [], {}, {}, {}
    sd = Xtr.std(0)

    for nm, idx in SEL.items():
        mdl = LinearRegression().fit(Xtr[:, idx], ytr)
        pt = mdl.predict(Xte[:, idx])
        k, n = len(idx), len(yte)
        rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        denom = ((yte - yte.mean()) ** 2).sum()
        r2 = 1.0 - ((yte - pt) ** 2).sum() / (denom if denom > 1e-12 else 1e-12)
        c = cv(idx)
        adj_r2 = float(1 - (1 - r2) * (n - 1) / max(n - k - 1, 1))
        
        res.append(dict(
            method=nm,
            k=k,
            features=[F[j] for j in idx],
            train_rmse=rm(ytr, mdl.predict(Xtr[:, idx])),
            test_rmse=rm(yte, pt),
            test_r2=float(r2),
            adj_r2=adj_r2,
            cv_rmse=c[0],
            cv_r2=c[1]
        ))
        pred[nm] = {"actual": r4(yte), "pred": r4(pt)}
        coefs[nm] = {F[j]: float(w * sd[j]) for j, w in zip(idx, mdl.coef_)}
        models[nm] = {
            "intercept": round(float(mdl.intercept_), 4),
            "coefs": {F[j]: round(float(w), 6) for j, w in zip(idx, mdl.coef_)},
            "features": [F[j] for j in idx]
        }

    cols = F + [target]
    Z = np.column_stack([X, y])
    C = np.corrcoef(Z.T)
    order = np.argsort(-np.abs(C[-1, :-1]))[:15].tolist() + [p]

    return dict(
        info=dict(
            rows=len(y),
            n_train=len(ytr),
            n_test=len(yte),
            target=target,
            features=F,
            notes=notes,
            exhaustive_pool=[F[j] for j in pool],
            exhaustive_total=len(ex)
        ),
        preview=raw_preview,
        corr={"names": [cols[j] for j in order], "z": r4(C[np.ix_(order, order)])},
        results=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in res],
        matrix=[[int(j in idx) for j in range(p)] for idx in SEL.values()],
        pred=pred,
        coefs=coefs,
        models=models,
        lasso=lasso_p,
        curves=curves,
        exhaustive={
            "k": [e[1] for e in ex],
            "rmse": r4([e[2] for e in ex]),
            "r2": r4([e[3] for e in ex]),
            "top": [
                {"features": [F[j] for j in e[0]], "k": e[1], "cv_rmse": round(e[2], 4), "cv_r2": round(e[3], 4)}
                for e in top
            ]
        }
    )

def handle(payload):
    try:
        if not isinstance(payload, dict):
            payload = {}
        ds = payload.get("dataset", "startups")
        target = (payload.get("target") or "").strip() or None
        
        if payload.get("csv_data"):
            text = payload["csv_data"]
        elif ds in PRESETS:
            preset_file = os.path.join(DATA, PRESETS[ds])
            if not os.path.exists(preset_file):
                raise FileNotFoundError(f"找不到預設資料檔案: {PRESETS[ds]}")
            text = open(preset_file, encoding="utf-8-sig").read()
        else:
            url = (payload.get("url") or "").strip()
            if not url:
                raise ValueError("請提供有效的 CSV URL 或是選擇預設資料集")
            text = fetch(url)
            
        return 200, analyze(text, target)
    except Exception as e:
        return 400, {"error": f"{type(e).__name__}: {str(e)}"}
