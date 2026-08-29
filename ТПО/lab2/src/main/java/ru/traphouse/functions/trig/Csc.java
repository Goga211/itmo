package ru.traphouse.functions.trig;

public class Csc {
    private final Sin sin;

    public Csc(Sin sin) {
        this.sin = sin;
    }

    // csc(x) = 1 / sin(x)
    public double calculate(double x) {
        double sinVal = sin.calculate(x);
        if (Math.abs(sinVal) < 1e-10) {
            throw new ArithmeticException("csc(x) is undefined: sin(x) = 0 at x = " + x);
        }
        return 1.0 / sinVal;
    }
}
