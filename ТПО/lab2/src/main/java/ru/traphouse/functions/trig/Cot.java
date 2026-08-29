package ru.traphouse.functions.trig;

public class Cot {
    private final Sin sin;
    private final Cos cos;

    public Cot(Sin sin, Cos cos) {
        this.sin = sin;
        this.cos = cos;
    }

    // cot(x) = cos(x) / sin(x)
    public double calculate(double x) {
        double sinVal = sin.calculate(x);
        if (Math.abs(sinVal) < 1e-10) {
            throw new ArithmeticException("cot(x) is undefined: sin(x) = 0 at x = " + x);
        }
        return cos.calculate(x) / sinVal;
    }
}
