import os
from pathlib import Path

from game_agent import models as shared


Q2_XL = {
    'label':'Qwen3.8 27B UD-Q2_K_XL / RTX5050 + RX580 / post-encode front51-54 residency (8K, Q8 KV, cache OFF) / GCN medium + FA BR8 + IQ3S TPB16 / MTP OFF',
    'model':'Qwen3.8-27B-UD-Q2_K_XL.gguf',
    'projector':'mmproj-Qwen3.8-27B-Q8_0.gguf',
    'image_min_tokens':1024,'image_max_tokens':1024,
}
IQ2_S = dict(shared.MODEL_PRESETS['qwen38_27b_iq2_s'],
             label='Qwen3.8 27B IQ2_S + Q8 vision / RTX5050 + RX580 FFN9 + head')
MODEL_PRESETS = {**shared.MODEL_PRESETS,'qwen38_27b_iq2_s':IQ2_S,'qwen38_27b_q2_k_xl':Q2_XL}
VULKAN_BACKEND = Path(__file__).resolve().parent/'data'/'runtimes'/'llama-b11000-vulkan'/'ggml-vulkan.dll'
Q2_MEDIUM_BACKEND = VULKAN_BACKEND.parent.with_name('llama-b11000-vulkan-gcn-medium')/'ggml-vulkan.dll'
Q2_BR8_BACKEND = VULKAN_BACKEND.parent.with_name('llama-b11000-vulkan-gcn-medium-br8')/'ggml-vulkan.dll'
Q2_TPB16_BACKEND = VULKAN_BACKEND.parent.with_name('llama-b11000-vulkan-gcn-br8-tpb16')/'ggml-vulkan.dll'
Q2_FRONT_RUNTIME = VULKAN_BACKEND.parent.with_name('llama-b11000-front-postencode')


def uses_front_residency(model,projector,context_tokens,cache_ram_mib):
    return (model_preset(model) is Q2_XL and Path(projector).name.casefold()==Q2_XL['projector'].casefold()
            and context_tokens==8192 and cache_ram_mib==0)


def model_preset(model):
    if Path(model).name.casefold()==Q2_XL['model'].casefold():
        return Q2_XL
    if Path(model).name.casefold()==IQ2_S['model'].casefold():
        return IQ2_S
    return shared.model_preset(model)


def model_label(model):
    preset = model_preset(model)
    return preset['label'] if preset else Path(model).name


def apply_model_preset(config,name,directory):
    if name!='qwen38_27b_q2_k_xl':
        return shared.apply_model_preset(config,name,directory)
    paths = {key:str(Path(directory)/Q2_XL[key]) for key in ('model','projector')}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return dict(config,**paths,image_min_tokens=1024,image_max_tokens=1024)


def uses_dual_gpu(model,projector):
    preset = model_preset(model)
    return preset is Q2_XL or (preset is IQ2_S and Path(projector).name.casefold()==IQ2_S['projector'].casefold())


def model_server_options(model,projector,image_max_tokens):
    if not uses_dual_gpu(model,projector):
        return shared.model_server_options(model,projector,image_max_tokens)
    if Path(projector).name.casefold()!=Q2_XL['projector'].casefold():
        raise ValueError('Selected model requires its matching projector: '+Q2_XL['projector'])
    if image_max_tokens>1024:
        raise ValueError('This dual-GPU profile supports image tokens up to 1024')
    iq2s = model_preset(model) is IQ2_S
    override = (r'^(output\.weight|blk\.(5[5-9]|6[0-3])\.ffn_(gate|up|down)\.weight)$=Vulkan0'
            if iq2s else r'^token_embd\.weight$=CPU,^blk\.51\.ffn_(gate|up|down)\.weight$=Vulkan0'
            r',^blk\.55\.(ffn_(gate|up|down)|post_attention_norm)\.weight$=Vulkan0')
    return ['-c','8192','-ngl','all','-b','512','-ub','128','-ctk','q8_0','-ctv','q8_0',
            '--fit-target','128','-t','12','-tb','12' if iq2s else '6','--no-host',
            '--device','CUDA0,Vulkan0' if iq2s else 'CUDA0,Vulkan0,CUDA0,Vulkan0',
            '--split-mode','layer','--tensor-split','1,0' if iq2s else '52,3,1,10',
            '--mmproj-device','CUDA0','--override-tensor',override]+([] if iq2s else ['--spec-type','none'])


def server_environment(model,projector):
    if not uses_dual_gpu(model,projector):
        return None
    q2xl = model_preset(model) is Q2_XL
    backend = Q2_TPB16_BACKEND if q2xl else VULKAN_BACKEND
    if not backend.is_file():
        raise FileNotFoundError(Path(model).name+' requires the b11000 Vulkan backend: '+str(backend))
    environment = dict(os.environ,GGML_BACKEND_PATH=str(backend),CUDA_SCALE_LAUNCH_QUEUES='4x')
    if q2xl:
        for key in ('GGML_VK_GCN_LARGE_TILE','GGML_VK_GCN_PROBE','LLAMA_MTP_IMAGE_PACKED',
                    'GGML_VK_GCN_FA_Q8_DIRECT','GGML_VK_GCN_FA_MASK_OPT','GGML_VK_PERF_LOGGER',
                    'GGML_VK_PERF_LOGGER_CONCURRENT','GGML_VK_PERF_LOGGER_FREQUENCY',
                    'GGML_VK_PERF_LOGGER_DETAILS','GGML_SCHED_COPY_TRACE','GGML_VK_GCN_FA_BR4',
                    'GGML_VK_GCN_IQ3_ROWS4','GGML_VK_GCN_DOWN_SMALL','GGML_VK_GCN_IQ3_DMMV'):
            environment.pop(key,None)
        environment['GGML_VK_GCN_MEDIUM_TILE'] = '1'
        environment['GGML_VK_GCN_FA_BR8'] = '1'
        environment['GGML_VK_GCN_IQ3_TPB16'] = '1'
        environment['GGML_VK_VISIBLE_DEVICES'] = '0'
        environment['GGML_VK_ALLOW_GRAPHICS_QUEUE'] = '1'
        environment['GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM'] = '1'
    return environment