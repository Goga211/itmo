public class CheckPointPos {
    public static boolean isPointInside(double x, double y, double r) {
        return (isPointSecondQuarter(x, y, r) || isPointThirdQuarter(x, y, r) || isPointFourthQuarter(x, y, r));
    }

    // 2 четверть: четверть круга радиуса r/2
    private static boolean isPointSecondQuarter(double x, double y, double r) {
        if (x <= 0.0D && y >= 0.0D) {
            double radius = r / 2.0D;
            return (x * x + y * y <= radius * radius);
        }
        return false;
    }

    // 3 четверть: треугольник от -r/2 по X до -r/2 по Y
    private static boolean isPointThirdQuarter(double x, double y, double r) {
        if (x <= 0.0D && y <= 0.0D) {
            double halfR = r / 2.0D;
            // Треугольник с вершинами: (0,0), (-r/2,0), (0,-r/2)
            return (x >= -halfR && y >= -halfR && y >= x);
        }
        return false;
    }

    // 4 четверть: прямоугольник от -r/2 по Y до r по X
    private static boolean isPointFourthQuarter(double x, double y, double r) {
        if (x >= 0.0D && y <= 0.0D) {
            double halfR = r / 2.0D;
            return (x <= r && y >= -halfR);
        }
        return false;
    }
}
