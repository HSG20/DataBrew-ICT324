# DataBrew | ICT324 | French Bakery Daily Sales
# Reformatted from the original experimental script; computational logic unchanged.
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys, math, time, itertools, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('Bakery sales.csv')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name('DataBrew_bakery_results')
OUT.mkdir(exist_ok=True, parents=True)
SUP = 0.01
CONF = 0.2
SEED = 42

# Apriori: generate candidates, prune infrequent subsets and intersect receipt IDs.
def apriori(baskets, m):
    # Build a product-to-receipt index for efficient co-occurrence counting.
    tids = defaultdict(set)
    for i, b in enumerate(baskets):
        for p in b:
            tids[p].add(i)
    level = {(p,): s for p, s in tids.items() if len(s) >= m}
    out = {k: len(v) for k, v in level.items()}
    while len(level) > 1:
        keys = sorted(level)
        new = {}
        k = len(keys[0])
        for i, x in enumerate(keys):
            for y in keys[i + 1:]:
                if x[:-1] != y[:-1]:
                    break
                z = x + (y[-1],)
                if k > 1 and any((s not in level for s in itertools.combinations(z, k))):
                    continue
                s = level[x] & level[y]
                if len(s) >= m:
                    new[z] = s
        out.update({k: len(v) for k, v in new.items()})
        level = new
    return out

# FP-tree node storing item counts and links to parent and child nodes.
class Node:
    __slots__ = ('item', 'count', 'parent', 'children')

    def __init__(self, item=None, parent=None):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children = {}

# FP-Growth: build compressed trees and recursively mine conditional pattern bases.
def fp_growth(baskets, m):
    out = {}

    def mine(paths, prefix):
        counts = Counter()
        for items, w in paths:
            for p in items:
                counts[p] += w
        counts = {p: n for p, n in counts.items() if n >= m}
        if not counts:
            return
        root = Node()
        header = defaultdict(list)
        for items, w in paths:
            node = root
            for p in sorted((p for p in items if p in counts), key=lambda p: (-counts[p], p)):
                child = node.children.get(p)
                if child is None:
                    child = Node(p, node)
                    node.children[p] = child
                    header[p].append(child)
                child.count += w
                node = child
        for p in sorted(counts, key=lambda p: (counts[p], p)):
            pattern = prefix + (p,)
            out[tuple(sorted(pattern))] = counts[p]
            conditional = []
            for node in header[p]:
                parent = node.parent
                path = []
                while parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional.append((path, node.count))
            if conditional:
                mine(conditional, pattern)
    mine([(b, 1) for b in baskets], ())
    return out

# Generate association rules from frequent itemsets; retain confidence >= 20% and lift > 1.
def rules(freq, n, group):
    rows = []
    for items, count in freq.items():
        for size in range(1, len(items)):
            for ant in itertools.combinations(items, size):
                cons = tuple((i for i in items if i not in ant))
                conf = count / freq[ant]
                lift = conf / (freq[cons] / n)
                if conf >= CONF and lift > 1:
                    rows.append(dict(group=group, antecedent=' + '.join(ant), consequent=' + '.join(cons), support=count / n, confidence=conf, lift=lift, cooccurrences=count, antecedent_count=freq[ant], consequent_count=freq[cons], antecedent_size=len(ant), consequent_size=len(cons)))
    return pd.DataFrame(rows, columns=['group', 'antecedent', 'consequent', 'support', 'confidence', 'lift', 'cooccurrences', 'antecedent_count', 'consequent_count', 'antecedent_size', 'consequent_size'])

# End-to-end data audit, preparation, clustering and mining.
def main():
    d = pd.read_csv(SOURCE)
    d['price'] = pd.to_numeric(d.unit_price.str.replace('€', '', regex=False).str.replace(',', '.', regex=False).str.strip())
    audit = dict(raw_rows=len(d), raw_tickets=int(d.ticket_number.nunique()), raw_products=int(d.article.nunique()), raw_multi_product_tickets=int((d.groupby('ticket_number').article.nunique() > 1).sum()), missing_cells=int(d.isna().sum().sum()), exact_duplicate_rows=int(d.duplicated().sum()), same_values_excluding_row_id=int(d.drop(columns='Unnamed: 0').duplicated().sum()), negative_quantity_rows=int((d.Quantity < 0).sum()), zero_price_rows=int((d.price == 0).sum()), first_date=d.date.min(), last_date=d.date.max(), ticket_date_conflicts=int((d.groupby('ticket_number').date.nunique() > 1).sum()), ticket_time_conflicts=int((d.groupby('ticket_number').time.nunique() > 1).sum()))
    assert audit['missing_cells'] == 0 and audit['ticket_date_conflicts'] == audit['ticket_time_conflicts'] == 0
    # Exclude tickets containing negative quantities, then remove remaining nonpositive purchases.
    neg = d.loc[d.Quantity < 0, 'ticket_number'].unique()
    a = d.loc[~d.ticket_number.isin(neg)].copy()
    audit.update(excluded_return_tickets=len(neg), excluded_return_ticket_rows=len(d) - len(a), excluded_nonpositive_price_quantity_rows=int(((a.price <= 0) | (a.Quantity <= 0)).sum()))
    a = a.loc[(a.price > 0) & (a.Quantity > 0)].copy()
    a['spend'] = a.Quantity * a.price
    a['hour'] = pd.to_datetime(a.time, format='%H:%M').dt.hour
    b = a.groupby('ticket_number', sort=True).agg(date=('date', 'first'), hour=('hour', 'first'), units=('Quantity', 'sum'), basket_size=('article', 'nunique'), spend=('spend', 'sum'), items=('article', lambda x: tuple(sorted(set(x)))))
    audit.update(clean_rows=len(a), clean_tickets=len(b), clean_products=int(a.article.nunique()), clean_multi_product_tickets=int((b.basket_size > 1).sum()), clean_avg_distinct_items=float(b.basket_size.mean()), clean_avg_units=float(b.units.mean()), clean_avg_spend=float(b.spend.mean()))
    print('AUDIT', json.dumps(audit), flush=True)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    # Reduce skew using log1p on size and spending, then standardize all features.
    x = StandardScaler().fit_transform(np.column_stack([b.hour, np.log1p(b.basket_size), np.log1p(b.spend)]))
    sample = np.random.default_rng(SEED).choice(len(b), size=5000, replace=False)
    scores = []
    models = {}
    # Fit K-Means on all valid receipts; evaluate silhouette on a fixed 5,000-receipt sample.
    for k in range(2, 9):
        m = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        score = silhouette_score(x[sample], m.labels_[sample])
        models[k] = m
        scores.append(dict(k=k, wcss=m.inertia_, silhouette=score))
        print('K', scores[-1], flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / 'kmeans_validation.csv', index=False)
    selected = int(scores.loc[scores.silhouette.idxmax(), 'k'])
    b['cluster'] = models[selected].labels_
    order = b.groupby('cluster').hour.mean().sort_values().index
    b['cluster'] = b.cluster.map({c: i for i, c in enumerate(order)})
    profile = b.groupby('cluster').agg(tickets=('hour', 'size'), mean_hour=('hour', 'mean'), median_hour=('hour', 'median'), min_hour=('hour', 'min'), max_hour=('hour', 'max'), mean_distinct_products=('basket_size', 'mean'), mean_units=('units', 'mean'), mean_spend_eur=('spend', 'mean'), multi_product_share=('basket_size', lambda x: (x > 1).mean()))
    profile['share'] = profile.tickets / len(b)
    profile.to_csv(OUT / 'cluster_profiles.csv')
    print('SELECTED', selected, '\n', profile.to_string(), flush=True)
    # Benchmark both mining methods three times and check exact itemset/count equivalence.
    all_rules = []
    summary = []
    for name, subset in [('Global', b)] + [(f'Cluster {i}', g) for i, g in b.groupby('cluster')]:
        baskets = subset['items'].tolist()
        n = len(baskets)
        min_count = math.ceil(SUP * n - 1e-12)
        times = {}
        outputs = {}
        for algo, func in [('Apriori', apriori), ('FP-Growth', fp_growth)]:
            tt = []
            for _ in range(3):
                t = time.perf_counter()
                f = func(baskets, min_count)
                tt.append(time.perf_counter() - t)
            outputs[algo] = f
            times[algo] = float(np.median(tt))
        assert outputs['Apriori'] == outputs['FP-Growth'], name
        f = outputs['Apriori']
        r = rules(f, n, name)
        all_rules.append(r)
        summary.append(dict(group=name, tickets=n, min_count=min_count, frequent_itemsets=len(f), rules=len(r), avg_support=r.support.mean(), avg_confidence=r.confidence.mean(), avg_lift=r.lift.mean(), apriori_seconds=times['Apriori'], fpgrowth_seconds=times['FP-Growth'], identical_itemsets=True))
        print('ARM', summary[-1], flush=True)
        print(r[(r.antecedent_size == 1) & (r.consequent_size == 1)].sort_values(['lift', 'support'], ascending=False).head(6).to_string(index=False), flush=True)
    r = pd.concat(all_rules, ignore_index=True)
    r.to_csv(OUT / 'association_rules.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'association_summary.csv', index=False)
    # Build a product-to-receipt index for efficient co-occurrence counting.
    tids = defaultdict(set)
    for i, items in enumerate(b['items']):
        for item in items:
            tids[item].add(i)

    def count(items):
        return len(set.intersection(*(tids[p] for p in items)))
    # Compare each segmented rule with its global equivalent, even below global thresholds.
    cache = {}
    matched = []
    for row in r[r.group != 'Global'].itertuples(index=False):
        a = tuple(row.antecedent.split(' + '))
        c = tuple(row.consequent.split(' + '))
        key = (a, c)
        if key not in cache:
            ac = count(a)
            cc = count(c)
            joint = count(a + c)
            gc = joint / ac
            gl = gc / (cc / len(b))
            cache[key] = (joint / len(b), gc, gl)
        gs, gc, gl = cache[key]
        matched.append(dict(group=row.group, antecedent=row.antecedent, consequent=row.consequent, support=row.support, confidence=row.confidence, lift=row.lift, global_support=gs, global_confidence=gc, global_lift=gl, confidence_change=row.confidence - gc, lift_change=row.lift - gl, cooccurrences=row.cooccurrences, antecedent_count=row.antecedent_count, antecedent_size=row.antecedent_size, consequent_size=row.consequent_size))
    matched = pd.DataFrame(matched)
    matched.to_csv(OUT / 'matched_rules.csv', index=False)
    print('MATCHED descriptive:\n', matched.groupby('group').agg(rules=('confidence', 'size'), avg_global_confidence=('global_confidence', 'mean'), avg_cluster_confidence=('confidence', 'mean'), avg_conf_change=('confidence_change', 'mean'), avg_lift_change=('lift_change', 'mean')).to_string(), flush=True)
    print('DONE', flush=True)
if __name__ == '__main__':
    main()
