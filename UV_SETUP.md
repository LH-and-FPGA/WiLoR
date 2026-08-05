# 使用 uv 配置 WiLoR 环境

有效栈：Ubuntu Linux x86_64、Python 3.10、PyTorch 2.7.1 / torchvision 0.22.1 /
CUDA 12.8。CUDA 12.8 是硬要求——本工作站的 RTX 5090 和 RTX PRO 4000 Blackwell
都是 sm_120，更早的 CUDA 构建跑不起来。

仓库自带的 `requirements.txt` 不带版本约束，且缺少两个实际必需的包（见下文
"元数据差异"）。以 `pyproject.toml` + `uv.lock` 为准。

## 宿主机：使用 .venv

在仓库根目录：

    uv sync --frozen
    source .venv/bin/activate

不设置 `UV_PROJECT_ENVIRONMENT` 时，uv 默认使用项目根目录的 `.venv`。也可以不
激活，直接：

    uv run --frozen python demo.py --img_folder demo_img --out_folder demo_out

## Docker：使用 .venv-docker

宿主机创建的 venv 内含绝对解释器路径，挂载进容器后通常不可用，因此容器里要用
独立环境：

    UV_PROJECT_ENVIRONMENT=.venv-docker uv sync --frozen
    source .venv-docker/bin/activate

容器中运行命令时要保留同一个变量，否则 `uv run` 会转回 `.venv`：

    UV_PROJECT_ENVIRONMENT=.venv-docker uv run --frozen python -V

**本工作站的容器已经预设了 `UV_PROJECT_ENVIRONMENT=.venv-docker`**（还有
`UV_LINK_MODE=copy`）。所以在容器里直接 `uv sync --frozen` 建出来的就是
`.venv-docker`，不需要手动加变量；反过来，如果你以为自己在建 `.venv` 却发现
它没出现，那说明你在容器里。

`.venv` 和 `.venv-docker` 都已加入 `.gitignore`，`uv.lock` 应提交到 Git。

## 解释器必须装在持久路径

这台机器上唯一的 Python 3.10 曾经是 uv 装在 `/tmp/rl-test3/graspgen-runtime/`
下的（`/root/.local/bin/python3.10` 只是指过去的软链）。**/tmp 会被清空**——
之前 `/tmp/isoswarm-wilor-env` 整个环境就是这么没的。venv 只记录基础解释器的
路径，解释器一没，venv 立刻报废。

因此解释器已改装到持久磁盘：

    export UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
    uv python install 3.10

建议把这一行写进 shell 配置。重建环境时显式指定：

    UV_PYTHON_INSTALL_DIR=/workspace/.uv-python uv sync --frozen

可以这样确认当前 venv 的基础解释器不在 /tmp：

    head -1 .venv-docker/pyvenv.cfg
    # 期望：home = /workspace/.uv-python/cpython-3.10-linux-x86_64-gnu/bin

## 可选依赖

默认 `uv sync` 只装 PyTorch 推理所需的包。额外的按需同步：

    # TensorRT / ONNX / 量化：构建和运行加速引擎
    uv sync --frozen --extra accel

    # 主干蒸馏（distill/，见 distill/README.md）
    uv sync --frozen --extra distill

    # 上游 Gradio demo 和 hydra 训练入口
    uv sync --frozen --extra demo

    # 全部
    uv sync --frozen --all-extras

`accel` 约 12 GB（TensorRT + CUDA 运行时轮子占大头）。只做 PyTorch 推理的话
不需要它。

## 权重不由 uv 管理

Python 依赖由 uv 锁定，模型权重不写入 lock，且 `pretrained_models/` 与
`mano_data/` 已在 `.gitignore` 中：

    pretrained_models/detector.pt          51 MB
    pretrained_models/wilor_final.ckpt    2.4 GB
    mano_data/MANO_RIGHT.pkl             3.6 MB   （单独授权，须从 MANO 官网获取）
    mano_data/mano_mean_params.npz

## 已知的元数据差异

`pyproject.toml` 里有两处 `override-dependencies`，都是实测得出的，不要按包自己
声明的版本去"修正"：

1. **pyrender 0.1.45 元数据钉 `pyopengl==3.1.0`**，但 3.1.0 起不来无头渲染用的
   EGL 后端。实际可用的是 3.1.10。
2. **nvidia-modelopt 0.45.0 声明 `torch>=2.8`**，但本项目用到的是它的 ONNX
   量化路径，已在 torch 2.7.1 上验证可用（INT8/FP8 PTQ 引擎构建通过）。
   不要为了满足 modelopt 而放宽 torch 版本——2.8 没有在这块硬件上对
   `wilor_final.ckpt` 验证过。

另外，`requirements.txt` 缺两个实际必需的包，`pyproject.toml` 已补上：

- **`dill`**：`pretrained_models/detector.pt` 是用 dill 序列化的，没有它
  `torch.load` 会报 `ModuleNotFoundError: No module named 'dill'`。旧环境是靠
  别的包间接带进来的，所以上游没注意到。
- **`pyopengl`**：只作为 pyrender 的传递依赖存在，但版本必须被显式约束（见上）。

`xtcocotools` 在 `requirements.txt` 里但推理路径用不到，已省略。

### chumpy 需要 pip 才能构建

`chumpy` 只有 git 源（PyPI 最新是 0.70 且仅有 sdist；两个已验证环境里装的都是
git commit `580566eafc9ac68b2614b64d6f7aaa84eebb70da` 的 0.71）。它的 `setup.py`
在构建期 `import pip`，而 uv 的隔离构建环境不带 pip，冷缓存机器上会失败：

    ModuleNotFoundError: No module named 'pip'

`pyproject.toml` 已用 uv 的声明式修法解决：

    [tool.uv.extra-build-dependencies]
    chumpy = ["pip"]

**注意这个问题只在冷缓存时出现**——uv 缓存里一旦有构建好的 chumpy 就不会重建，
所以在老机器上测不出来，换新机器才暴露。已用独立的空 `UV_CACHE_DIR` 做过正反
对照验证。

## 验证

    CUDA_VISIBLE_DEVICES=0 PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      uv run --frozen python -c "
    import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))
    import cv2, smplx, pytorch_lightning, yacs, timm, ultralytics, chumpy, dill, pyrender
    import tensorrt, torchao, onnxruntime, modelopt.onnx.quantization
    print('OK')"

在 RTX PRO 4000 Blackwell 上的参考基准（主干 batch 2、fp32+TF32）：**17.4 ms**。
偏离超过约 5% 说明环境有问题。
