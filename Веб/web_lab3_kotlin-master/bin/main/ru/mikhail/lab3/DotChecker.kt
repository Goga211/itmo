package ru.mikhail.lab3

object DotChecker {

    fun checkDot(x: Float, y: Float, r: Float): Boolean {
        return checkFirstQuarter(x, y, r) || checkSecondQuarter(x, y, r) ||
                checkThirdQuarter(x, y, r)
    }

    private fun checkFirstQuarter(x: Float, y: Float, r: Float): Boolean {
        return if (x >= 0 && y >= 0) {
            // Треугольник со сторонами R/2
            y <= (-x + r / 2)
        } else {
            false
        }
    }

    private fun checkSecondQuarter(x: Float, y: Float, r: Float): Boolean {
        return if (x <= 0 && y >= 0) {
            // Четверть круга: x^2 + y^2 <= R^2
            x * x + y * y <= r * r
        } else {
            false
        }
    }

    private fun checkThirdQuarter(x: Float, y: Float, r: Float): Boolean {
        return if (x <= 0 && y <= 0) {
            // Квадрат
            x >= -r && y >= -r
        } else {
            false
        }
    }
}
