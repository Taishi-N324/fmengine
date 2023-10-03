import torch
from fmengine.utils import logger_rank0
from fmengine.utils.monitor import rank0_log
import deepspeed

def speed_monitor(elapsed_time, current_step, current_loss, configs, engine, model_config):
    if current_step % configs["trainer"]["log_steps"] == 0:
        logger_rank0.info(f"Step={current_step:>6}, Loss={current_loss.item():.4f}")
        logger_rank0.info(f"{elapsed_time:.2f} s/step")
        logger_rank0.info(
            f"{configs['deepspeed']['train_batch_size'] * configs['trainer']['max_seq_len']/elapsed_time:.2f} tokens/s"
        )
        torch.cuda.empty_cache()


def wandb_monitor(elapsed_time, current_step, current_loss, configs, engine, model_config):
    tps = (
        configs["deepspeed"]["train_batch_size"]
        * configs["trainer"]["max_seq_len"]
        / elapsed_time
    )

    # TODO Check if this is the correct way to check the presence or absence of deepspeed activation_checkpointing
    checkpoint_activations_factor = 3
    if deepspeed.checkpointing.is_configured() == True :
        checkpoint_activations_factor = 4

    flops_per_iteration: float = (
        24
        * checkpoint_activations_factor
        * configs["deepspeed"]["train_batch_size"]
        * configs["trainer"]["max_seq_len"]
        * model_config.num_hidden_layers
        * (model_config.hidden_size**2)
    ) * (
        1.0
        + (configs["trainer"]["max_seq_len"] / (6.0 * model_config.hidden_size))
        + (model_config.vocab_size / (16.0 * model_config.num_hidden_layers * model_config.hidden_size))
    )
    
    tflops = flops_per_iteration / (elapsed_time * (10**12))

    rank0_log(
        {
            "loss": current_loss.item(),
            "lr": engine.optimizer.param_groups[0]["lr"],
            "step": current_step,
            "tokens_per_second": tps,
            "step_time": elapsed_time,
            "tflops": tflops,
        }
    )
