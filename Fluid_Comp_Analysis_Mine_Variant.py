#%%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
# %%

#GOAL: To determine reasonable relationships to classify diamond forming fluids using the elements that can be measured in twinning wisps by LA-ICP-MS 

# Plots of Majors, Fluid Mobile, High Field Strength and And Melt compatible Elements Ree Elements 
#%%
path = "Weiss_et_al-RiMG-Global_Diamond_HDF_compilation_Aug_2021.xlsx"
sheet = "HDF_Limited_Col"


df = pd.read_excel(io=path,sheet_name=sheet)

fix, ax = plt.subplots()

sns.scatterplot(df, y = "Zr/La", x= "Pb/U", hue = "Locality ", ax = ax)
# ax.set_xbound(0,.00x0002)
# ax.set_ybound(0,20)
plt.yscale('log')
plt.xscale('log')


#%%
# Ba/Ce Vs LA/Ce is pretty good for differentiating Saline Fluids as is La/Ce Pb/U
# Th/U Is also Pretty useful for this
fig, ax = plt.subplots()
sns.scatterplot(df, y = "La/Ce", x= "Ba/Ce", hue = "Locality ", ax = ax)
# ax.set_xbound(0,.000002)

plt.yscale('log')
plt.xscale('log')
#ax.set_ybound(0.1,1)
# %%

#Mg/Al  vs Zr/La or Rb/Sr are Good at discriminating between Carbonatites and Silicic 
#Y/La or Zr/Sr
#Al/La vs Zr/La Is very good
# 

#" low ionic potential (e.g. Na, K and Rb) are generally considered to be mobile, whereas high-field-strength elements (HFSEs; e.g. Ti, Zr and Th) and REEs largely remain immobile (e.g. Pearce and Cann, 1973;Rollinson, 1993). Indeed, when several elements are plotted against Zr, the HFSEs and REEs ca"


#%%
fig, ax = plt.subplots()
sns.scatterplot(df, y = "Nb/La", x= "Zr/Sr", hue = "Locality ", ax = ax)
#ax.set_xbound(0,100)
# ax.set_ybound(0,20)
plt.yscale('log')
plt.xscale('log')


# %% Build log-ratio features from element concentrations
# log(element/Ce) from the concentration columns, missing kept as NaN (see hdf_compilation.py:
# the Excel ratio columns store a missing numerator as 0, and log(x + 1) squashes ratios below 1)
from hdf_compilation import LABEL, has_trace_data, log_ratios, make_classifier

target = "Locality "
# Localities with fewer than 10 rows with trace data are dropped: too few to cross-validate
min_rows = 10
features = log_ratios(df)
has_data = has_trace_data(features) & df[target].notna()
counts = df.loc[has_data, target].value_counts()
plot_order = counts[counts >= min_rows].index.tolist()
keep = has_data & df[target].isin(plot_order)
X = features[keep].to_numpy()
y = df.loc[keep, target].to_numpy()
groups = df.loc[keep, "Reference"].fillna("unknown").to_numpy()
hdf_type = df.loc[keep, [LABEL]].fillna("unknown")
counts

#%% Clustering Algorithms
from sklearn import mixture
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Missing values are filled with the column median, then each log ratio is scaled to unit variance
X_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)

n_components = 8
gmm = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=1,
    )

gmm.fit(X_scaled)
y_pred = gmm.predict(X_scaled)

plot_cols = ["Rb/Sr", "Zr/La", "Ba/Ce", "La/Ce"]
Classification_Comp = pd.DataFrame({"Locality": y, "Clusters": y_pred}, index=df.index[keep])
Classification_Comp = Classification_Comp.join(df[plot_cols].where(df[plot_cols] > 0))
# %%
cmap = sns.color_palette("hls", n_components)
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "Clusters", ax = ax, palette= cmap)

plt.yscale('log')
plt.xscale('log')



fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Rb/Sr", x= "Zr/La", hue = "Locality", ax = ax)

plt.yscale('log')
plt.xscale('log')

#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Clusters", ax = ax, palette=cmap)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Locality", ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%
sns.catplot(Classification_Comp, x  = "Locality", y = "Clusters" , alpha = 0.2, size = 10, order = plot_order )
plt.xticks(rotation=30)

pivot_table = Classification_Comp.pivot_table(index='Locality', columns='Clusters', aggfunc='size', fill_value=0)
pivot_table
#%% Classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder

# Balanced logistic regression on the log ratios (pipeline in hdf_compilation.make_classifier).
# An anisotropic-RBF Gaussian process classifier, used before 10/7/25, scored on a single split.
classifier = make_classifier()

# Baseline that knows only the fluid type. Localities differ in which fluid types were sampled (Ekati and
# Diavik are mostly saline, Internationalnaya mostly carbonatitic), so fluid type alone predicts some of them.
# Beating chance alone does not show a compositional fingerprint; the trace-element model has to
# beat this baseline.
baseline = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(class_weight="balanced", max_iter=5000)
)

stratified = StratifiedKFold(5, shuffle=True, random_state=0)
y_pred_strat = cross_val_predict(classifier, X, y, cv=stratified)
y_pred_baseline = cross_val_predict(baseline, hdf_type, y, cv=stratified)
print(f"n = {len(y)}, {len(plot_order)} classes, chance balanced accuracy = {1 / len(plot_order):.2f}")
print(f"balanced accuracy, stratified 5-fold, trace-element ratios: {balanced_accuracy_score(y, y_pred_strat):.3f}")
print(f"balanced accuracy, stratified 5-fold, fluid type only:      {balanced_accuracy_score(y, y_pred_baseline):.3f}")

# No group-by-paper score here: Nyurbinskaya, Ebelyakh and Ekati each come from a single paper, so
# holding out papers removes whole localities. Locality is therefore confounded with lab and method,
# and the stratified score above is an upper bound on how well a new lab's data would classify.
# Partial check: the localities studied by more than one paper, scored with whole papers held out.
papers_per_locality = pd.Series(groups).groupby(y).nunique()
multi_paper = np.isin(y, papers_per_locality[papers_per_locality > 1].index)
y_multi, groups_multi = y[multi_paper], groups[multi_paper]
y_pred_multi_group = cross_val_predict(classifier, X[multi_paper], y_multi, cv=GroupKFold(5), groups=groups_multi)
y_pred_multi_strat = cross_val_predict(classifier, X[multi_paper], y_multi, cv=stratified)
print(f"multi-paper localities {sorted(set(y_multi))}, n = {multi_paper.sum()}, chance {1 / len(set(y_multi)):.2f}")
print(f"  balanced accuracy, stratified 5-fold:     {balanced_accuracy_score(y_multi, y_pred_multi_strat):.3f}")
print(f"  balanced accuracy, group-by-paper 5-fold: {balanced_accuracy_score(y_multi, y_pred_multi_group):.3f}")

#%% Confusion matrix: rows are the actual Locality, columns the predicted Locality
test_pivot_table = pd.DataFrame(
    confusion_matrix(y, y_pred_strat, labels=plot_order), index=plot_order, columns=plot_order
)
test_pivot_table
# %%


classifier.fit(X, y)
y_all_pred = classifier.predict(X)
#%%
fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = y_all_pred, hue_order=plot_order, ax = ax)

plt.yscale('log')
plt.xscale('log')

fig, ax = plt.subplots()
sns.scatterplot(Classification_Comp, y = "Ba/Ce", x= "La/Ce", hue = "Locality", hue_order=plot_order, ax = ax)

plt.yscale('log')
plt.xscale('log')
# %%
