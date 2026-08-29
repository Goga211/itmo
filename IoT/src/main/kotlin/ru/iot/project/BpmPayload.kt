package ru.iot.project

import com.fasterxml.jackson.annotation.JsonProperty

data class BpmPayload(
    @JsonProperty("bpm_window") val bpmWindow: Float,
    @JsonProperty("bpm_buffer") val bpmBuffer: Int,
    @JsonProperty("ir") val ir: Long,
    @JsonProperty("samples") val samples: Int
)
