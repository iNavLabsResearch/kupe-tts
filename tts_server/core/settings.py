from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass(slots=True)
class ServerSettings:
    bind_host: str
    bind_port: int
    trust_proxy_headers: bool
    forwarded_allow_ips: str


@dataclass(slots=True)
class RuntimeSettings:
    model_id: str
    model_type: str
    weight_dtype: str
    executor_type: str


@dataclass(slots=True)
class VoiceSettings:
    default_voice: str
    auto_profiles: bool
    configured_profiles: list[str]


@dataclass(slots=True)
class BatchingSettings:
    max_workers: int
    max_batch_size: int
    timeout_ms: float


@dataclass(slots=True)
class SettingsBundle:
    server: ServerSettings
    runtime: RuntimeSettings
    voice: VoiceSettings
    batching: BatchingSettings


def load_settings() -> SettingsBundle:
    return SettingsBundle(
        server=ServerSettings(
            bind_host=config.BIND_HOST,
            bind_port=config.BIND_PORT,
            trust_proxy_headers=config.TRUST_PROXY_HEADERS,
            forwarded_allow_ips=config.FORWARDED_ALLOW_IPS,
        ),
        runtime=RuntimeSettings(
            model_id=config.MODEL_ID,
            model_type=config.MODEL_TYPE,
            weight_dtype=config.WEIGHT_DTYPE,
            executor_type=config.EXECUTOR_TYPE,
        ),
        voice=VoiceSettings(
            default_voice=config.DEFAULT_VOICE,
            auto_profiles=config.VOICE_PROFILES_AUTO,
            configured_profiles=list(config.VOICE_PROFILES),
        ),
        batching=BatchingSettings(
            max_workers=config.MAX_WORKERS,
            max_batch_size=config.MAX_BATCH_SIZE,
            timeout_ms=config.BATCH_TIMEOUT_MS,
        ),
    )

