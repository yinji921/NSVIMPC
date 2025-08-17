import ast
from common import *

# CUDA
import pycuda.autoinit

global drv
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class CudaModule(PrintBase):
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        param_names = [
            "samples",
            "horizon",
            "control_dim",
            "state_dim",
            "raceline_len",
            "dt",
            "cc_ratio",
            "zero_ref_ctrl_ratio",
            "alfa",
            "beta",
            "obstacle_radius",
            "cuda_filename",
            "r1",
            "r2",
            "curand_kernel_n",
            "use_raceline",
            "obstacle_radius",
        ]
        self.config_to_attribute_by_name(config_data, section_name, param_names)

        with open(self.cuda_filename, "r") as f:
            code = f.read()

        cuda_code_macros = {
            "SAMPLE_COUNT": self.samples,
            "HORIZON": self.horizon,
            "CONTROL_DIM": self.control_dim,
            "STATE_DIM": self.state_dim,
            "RACELINE_LEN": self.raceline_len,
            "DT": self.dt,
            "CC_RATIO": self.cc_ratio,
            "ZERO_REF_CTRL_RATIO": self.zero_ref_ctrl_ratio,
            "MAX_V": 3.0,
            "R1": self.r1,
            "R2": self.r2,
        }
        cuda_code_macros.update({"CURAND_KERNEL_N": self.curand_kernel_n})
        cuda_code_macros.update({"alfa": self.alfa})
        cuda_code_macros.update({"beta": self.beta})
        cuda_code_macros.update(
            {"use_raceline": "true" if self.use_raceline > 0.5 else "false"}
        )
        cuda_code_macros.update({"obstacle_radius": self.obstacle_radius})
        mod = SourceModule(code % cuda_code_macros, no_extern_c=True)
        self.mod = mod

        self.K = self.samples
        threads_per_block = 512
        if self.K < 512:
            # if K is small only employ one grid
            self.cuda_block_size = (self.K, 1, 1)
            self.cuda_grid_size = (1, 1)
        else:
            # employ multiple grid,
            self.cuda_block_size = (512, 1, 1)
            self.cuda_grid_size = (ceil(self.K / float(threads_per_block)), 1)
        self.print_info(
            "cuda block size %d, grid size %d"
            % (self.cuda_block_size[0], self.cuda_grid_size[0])
        )

        self.cuda_init_curand_kernel = mod.get_function("init_curand_kernel")
        self.cuda_generate_random_var = mod.get_function("generate_random_normal")
        self.cuda_evaluate_control_sequence = mod.get_function(
            "evaluate_control_sequence"
        )

        seed = np.int32(int(time() * 10000))
        self.cuda_init_curand_kernel(seed, block=(1024, 1, 1), grid=(1, 1, 1))

        self.rand_vals = np.zeros(self.K * self.T * self.m, dtype=np.float32)
        self.device_rand_vals = drv.to_device(self.rand_vals)

        print_info(
            "registers used each kernel in eval_ctrl= %d"
            % self.cuda_evaluate_control_sequence.num_regs
        )
        assert (
            int(self.cuda_evaluate_control_sequence.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert (
            int(self.cuda_init_curand_kernel.num_regs * self.cuda_block_size[0])
            <= 65536
        )
        assert (
            int(self.cuda_generate_random_var.num_regs * self.cuda_block_size[0])
            <= 65536
        )

        self.discretized_raceline = discretized_raceline.astype(np.float32)
        self.discretized_raceline = self.discretized_raceline.flatten()
        self.device_discretized_raceline = drv.to_device(self.discretized_raceline)

        sleep(0.01)

    # read names from config data and set them as attribute
    def config_to_attribute_by_name(self, config_data, section_name, names):
        for name in names:
            setattr(
                self,
                name,
                np.asarray(ast.literal_eval(config_data.get(section_name, name))),
            )
