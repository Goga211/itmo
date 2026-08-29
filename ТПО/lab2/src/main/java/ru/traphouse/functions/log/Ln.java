package ru.traphouse.functions.log;

public class Ln {
    private final double epsilon;

    public Ln(double epsilon) {
        this.epsilon = epsilon;
    }

    public double calculate(double x) {
        if (x <= 0) {
            throw new ArithmeticException("ln(x) is undefined for x <= 0, got: " + x);
        }

        // ln(x) = 2 * Σ (1/(2k+1)) * ((x-1)/(x+1))^(2k+1), converges for all x > 0
        double t = (x - 1.0) / (x + 1.0);
        double t2 = t * t;
        double result = 0;
        double term = t;
        int k = 0;

        do {
            result += term / (2 * k + 1);
            term *= t2;
            k++;
        } while (Math.abs(term / (2 * k + 1)) > epsilon);

        return 2 * result;
    }
}
