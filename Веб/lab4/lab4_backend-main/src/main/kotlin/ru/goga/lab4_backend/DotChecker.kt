package ru.goga.lab4_backend

import kotlin.math.abs

object DotChecker {

    fun checkDot(x: Float, y: Float, r: Float): Boolean {
        if (r == 0f) return false

        val R = abs(r)

        return checkTriangle(x, y, R) ||
                checkRectangle(x, y, R) ||
                checkQuarterCircle(x, y, R)
    }

    private fun checkTriangle(x: Float, y: Float, r: Float): Boolean {
        if (x <= 0f && y >= 0f) {
            val withinX = x >= -r / 2f && x <= 0f
            val withinY = y >= 0f && y <= r / 2f
            val underLine = y <= x + r / 2f
            return withinX && withinY && underLine
        }
        return false
    }


    private fun checkRectangle(x: Float, y: Float, r: Float): Boolean {
        if (x <= 0f && y <= 0f) {
            val withinX = x >= -r && x <= 0f
            val withinY = y >= -r / 2f && y <= 0f
            return withinX && withinY
        }
        return false
    }


    private fun checkQuarterCircle(x: Float, y: Float, r: Float): Boolean {
        if (x >= 0f && y <= 0f) {
            val radius = r / 2f
            return x * x + y * y <= radius * radius
        }
        return false
    }
}
