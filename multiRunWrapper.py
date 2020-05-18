import options as op
import numpy as np
from utilities import store_results

def multiRunWrapper(modelTrainerFunction, modelName):
    options = op.read()

    num_of_runs = range(options['num_runs'])
    run_results = {
                        "train_seen_domains" : [],
                        "valid_seen_domains" : [],
                        "test_seen_domains": [],
                        "train_unseen_domains": [],
                        "val+test_unseen_domains" : []
                    }
    for run_id in num_of_runs:
        print(str(modelName) + " - starting run " + str(run_id))
        current_res = modelTrainerFunction(run_id)
        for key, obj in run_results.items():
            obj.append(current_res[key])

    #### calculating average and std of runs, store final results
    final_results = {
                        "train_seen_domains" : {"mean": np.mean(run_results["train_seen_domains"]), "std":np.std(run_results["train_seen_domains"])},
                        "valid_seen_domains" :  {"mean": np.mean(run_results["valid_seen_domains"]), "std":np.std(run_results["valid_seen_domains"])},
                        "test_seen_domains":  {"mean": np.mean(run_results["test_seen_domains"]), "std":np.std(run_results["test_seen_domains"])},
                        "train_unseen_domains":  {"mean": np.mean(run_results["train_unseen_domains"]), "std":np.std(run_results["train_unseen_domains"])},
                        "val+test_unseen_domains" :  {"mean": np.mean(run_results["val+test_unseen_domains"]), "std":np.std(run_results["val+test_unseen_domains"])}
                    }
    print("################################ FINAL AVERAGE RESULTS ###################################################")
    print(final_results)
    store_results(final_results, modelName, "average", options)