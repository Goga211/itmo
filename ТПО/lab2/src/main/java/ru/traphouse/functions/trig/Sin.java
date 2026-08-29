package ru.traphouse.functions.trig;

public class Sin {
    private final double epsilon;

    public Sin(double epsilon) {
        this.epsilon = epsilon;
    }

    public double calculate(double x) {
        x = x % (2 * Math.PI);
        if (x > Math.PI) x -= 2 * Math.PI;
        if (x < -Math.PI) x += 2 * Math.PI;

        double result = 0;
        double term = x;
        int n = 1;

        while (Math.abs(term) > epsilon) {
            result += term;
            term *= -x * x / ((2.0 * n) * (2.0 * n + 1));
            n++;
        }

        return result;
    }
}
