package ru.goga.lab4_backend.data.responses

data class RefreshTokenResponse(
    var accessToken: String? = "",
    var error: String = ""
)
