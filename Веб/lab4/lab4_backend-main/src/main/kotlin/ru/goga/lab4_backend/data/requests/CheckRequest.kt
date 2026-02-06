package ru.goga.lab4_backend.data.requests

import com.fasterxml.jackson.annotation.JsonProperty


data class CheckRequest(
//    @field: Min(-3) @field:Max(5)
    @JsonProperty("x") val x: Float,

//    @field:Min(-3) @field:Max(5)
    @JsonProperty("y") val y: Float,

//    @field:Min(-3) @field:Max(5)
    @JsonProperty("r") val r: Float
)
