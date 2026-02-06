package ru.goga.lab2;

public class DotChecker {

    static boolean checkDot(float x, int y, int r) {
        return checkFirstQuarter(x, y, r)
                || checkSecondQuarter(x, y, r)
                || checkThirdQuarter(x, y, r)
                || checkFourthQuarter();
    }

    // 1 четверть: четверть окружности радиусом R/2
    static boolean checkFirstQuarter(float x, int y, int r) {
        if (x >= 0 && y >= 0) {
            double radius = r / 2.0;
            return (x * x + y * y <= radius * radius);
        }
        return false;
    }

    // 2 четверть: треугольник с вершинами (0,0), (-R/2,0), (0,R/2)
    static boolean checkSecondQuarter(float x, int y, int r) {
        if (x <= 0 && y >= 0) {
            return (y <= (r / 2.0 + x) && x >= -r / 2.0);
        }
        return false;
    }

    // 3 четверть: квадрат [-R,0] x [-R,0]
    static boolean checkThirdQuarter(float x, int y, int r) {
        if (x <= 0 && y <= 0) {
            return (x >= -r && y >= -r);
        }
        return false;
    }

    // 4 четверть: пусто
    static boolean checkFourthQuarter() {
        return false;
    }
}
