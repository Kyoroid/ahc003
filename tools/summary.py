#%%
import optuna

study = optuna.load_study(
    study_name="ahc003",
    storage="sqlite:///ahc003.db"
)
# %%
print(study.trials_dataframe())
# %%
optuna.visualization.plot_optimization_history(study)
# %%
optuna.visualization.plot_contour(study)
# %%
