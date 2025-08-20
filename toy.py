import random
import propulate
import mlflow

def loss_fn(params):
    mlflow.set_experiment("toy_experiment")
    with mlflow.start_run(run_name=f"run_{params.rank}_{params.generation}"):
        mlflow.log_param("x", params['x'])
        mlflow.log_param("y", params['y'])
        mlflow.log_param("generation", params.generation)

        loss = params["x"] ** 2 + params["y"] ** 2

        mlflow.log_metric("loss", loss, step=0)
    return loss

limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12)}
generations = 100
rng=random.Random()
checkpoint_path="/tmp/pcheckpoints"
propulate.utils.set_logger_config()

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
