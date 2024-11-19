import functools
from typing import Any, Callable, Sequence, Tuple

import flax 
from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any
Init = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
Conv = functools.partial(nn.Conv, use_bias=True, kernel_init=Init, bias_init=nn.initializers.zeros)

class WideBlock(nn.Module):
    filters: int
    conv: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    strides_helper = jnp.zeros(strides)

    @nn.compact
    def __call__(self, x,):
        residual = x
        x = self.act(x)
        x = self.conv(self.filters, (3, 3), padding=[(1,1), (1,1)])(x)
        x = self.act(x)
        x = self.conv(self.filters, (3, 3), self.strides, padding=[(1,1), (1,1)])(x)

        if residual.shape != x.shape or self.strides_helper.shape != (1, 1):
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)

        return residual + x

class WideResNet(nn.Module):
    num_classes: int
    depth: int
    widen_factor: int
    act: Callable = functools.partial(nn.leaky_relu, negative_slope=0.2)
    block_cls: ModuleDef = WideBlock

    @nn.compact
    def __call__(self, x):
        conv = Conv

        n = (self.depth-4)//6
        k = self.widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        x = conv(nStages[0], (3, 3), (1, 1), padding=[(1,1), (1,1)])(x)    
    
        def _wide_layer(x, planes, num_blocks, stride):
            strides = [stride] + [(1, 1)]*(num_blocks-1)
            for stride in strides:
                x = self.block_cls(filters=planes, strides=stride, act=self.act, conv=conv)(x)
            return x

        x = _wide_layer(x, nStages[1], n, (1, 1))
        x = _wide_layer(x, nStages[2], n, (2, 2))
        x = _wide_layer(x, nStages[3], n, (2, 2))  
    
        x = self.act(x)

        x = nn.avg_pool(x, window_shape=(8, 8))
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.num_classes, kernel_init=Init)(x)
    
        return x

WRN = functools.partial(WideResNet, num_classes=10, depth=28, widen_factor=10)

def load_pretrained_model(path):
    params_dict = WRN().init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

    with open(path, "rb") as f:
        params = flax.serialization.from_bytes(params_dict, f.read())

    return params

def log_prob(images, params):
    logits = WRN().apply(params, images)
    lse = jax.scipy.special.logsumexp(logits, axis=1)
    return lse.sum(), lse
log_prob_grad = jax.jit(jax.grad(log_prob, argnums=0, has_aux=True))

def log_joint_prob(images, labels, params):
    logits = WRN().apply(params, images)
    logits_y = logits[jnp.arange(logits.shape[0]), labels]
    return logits_y.sum(), logits_y
log_joint_prob_grad = jax.jit(jax.grad(log_joint_prob, argnums=0, has_aux=True))

def get_grads_wrt_input(params, x, y=None):
    x = jnp.array(x)
    if y is not None:
        grad, log_p = log_joint_prob_grad(x, jnp.array(y), params)
    else:
        grad, log_p = log_prob_grad(x, params)
    return log_p, grad