import random
import propulate
import mlflow

def loss_fn(params):
    mlflow.log_metric(loss)
    mlflow.log_param(params['x'])
    mlflow.log_param(params['y'])
    loss = params["x"] ** 2 + params["y"] ** 2
    return loss

limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
rng=random.Random()
checkpoint_path="/tmp/pcheckpoints"
propulate.utils.set_logger_config()
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_PER_NODE"])

propagator = propulate.utils.get_default_propagator(
        pop_size=10,
        limits=limits,
        rng=rng,
        )

propulator = propulate.Propulator(
        loss_fn=loss_fn,
        propagator=propagator,
        rng=rng,
        generations=generations,
        checkpoint_path=checkpoint_path,
        )

propulator.propulate()
