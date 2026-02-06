package ru.mikhail.lab3.dto

import java.sql.Timestamp

data class ResponseData(
    val x: Float,
    val y: Float,
    val r: Float,
    val result: Boolean,
    val executionTime: Long,
    val nowTime: Timestamp
) {
    override fun toString(): String {
        return "Data(x=$x, y=$y, r=$r, result=$result, executionTime=$executionTime, nowTime=$nowTime)"
    }
}
